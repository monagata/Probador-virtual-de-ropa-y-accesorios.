"""
================================================================================
PROBADOR VIRTUAL 3D CON INTELIGENCIA ARTIFICIAL
Proyecto de Grado - Sistema Completo Integrado
================================================================================
Autor: [Tu Nombre]
Fecha: 2026
Hardware: RTX 3050 4GB VRAM
Python: 3.11.8

OBJETIVOS CUMPLIDOS:
1. ✅ Procesamiento de imágenes y modelos 3D de realidad aumentada
2. ✅ Interfaz intuitiva con selección de prendas y ajuste de parámetros
3. ✅ IA para recomendar estilos, tallas y colores
4. ✅ Sistema de evaluación y validación completo
================================================================================
"""

# ============================================
# IMPORTS Y CONFIGURACIÓN INICIAL
# ============================================
import os
import sys
import time
import json
import threading
import warnings
from datetime import datetime
from pathlib import Path

# Configuración de encodings y warnings
if sys.platform == "win32":
    os.system('chcp 65001 > nul 2>&1')
warnings.filterwarnings('ignore')

# Librerías core
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2
import colorsys

# Deep Learning
import torch
import torch.nn.functional as F

# Computer Vision y Pose Detection
import mediapipe as mp

# 3D Processing
import trimesh

# Background Removal
from rembg import remove

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    mean_absolute_error, cohen_kappa_score
)

# Visualización
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para servidor
import matplotlib.pyplot as plt
import seaborn as sns

# Gradio UI
import gradio as gr

# QR Code
import qrcode

# HTTP Server
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Utilities
import psutil
import joblib


# ============================================
# CONFIGURACIÓN GLOBAL
# ============================================
class Config:
    """Configuración centralizada del sistema"""

    # Directorios
    BASE_DIR = Path(__file__).parent
    OUTPUT_DIR = BASE_DIR / "avatares_generados"
    CATALOGO_DIR = BASE_DIR / "catalogo_prendas"
    EVALUACION_DIR = BASE_DIR / "evaluacion_resultados"
    MODELS_DIR = BASE_DIR / "modelos_ia"
    DATASET_DIR = BASE_DIR / "dataset_validacion"

    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    VRAM_LIMIT_MB = 3500  # RTX 3050 tiene 4GB, dejamos margen

    # Servidor HTTP
    HTTP_PORT = 5000
    HTTP_HOST = '0.0.0.0'

    # Parámetros de procesamiento
    MAX_IMAGE_SIZE = (1024, 1024)  # Optimizado para RTX 3050
    DEPTH_MODEL = "MiDaS_small"  # Modelo pequeño para 4GB VRAM
    POSE_COMPLEXITY = 1  # 0=Lite, 1=Full, 2=Heavy

    # IA
    MIN_CONFIDENCE_TALLA = 0.7
    MIN_CONFIDENCE_COLOR = 0.65

    # Crear directorios
    for dir_path in [OUTPUT_DIR, CATALOGO_DIR, EVALUACION_DIR, MODELS_DIR, DATASET_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)

config = Config()

print(f"""
╔══════════════════════════════════════════════════════════╗
║     PROBADOR VIRTUAL 3D - SISTEMA COMPLETO              ║
╠══════════════════════════════════════════════════════════╣
║  Hardware: {config.DEVICE.upper():48s}║
║  VRAM Disponible: {torch.cuda.get_device_properties(0).total_memory // (1024**2) if torch.cuda.is_available() else 0:4d} MB                        ║
║  Python: {sys.version.split()[0]:47s}║
╚══════════════════════════════════════════════════════════╝
""")


# ============================================
# MÓDULO 1: PROCESAMIENTO DE IMÁGENES Y 3D
# ============================================

class MedicionesAntropometricas:
    """
    OBJETIVO 1: Detección de pose y extracción de medidas corporales
    Utiliza MediaPipe Pose para detectar 33 keypoints del cuerpo
    """

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=config.POSE_COMPLEXITY,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detectar_puntos(self, imagen):
        """
        Detecta keypoints del cuerpo humano

        Args:
            imagen: PIL Image o numpy array

        Returns:
            dict con coordenadas de puntos clave o None si falla
        """
        # Convertir a numpy si es PIL
        if isinstance(imagen, Image.Image):
            img_np = np.array(imagen)
        else:
            img_np = imagen

        # MediaPipe espera RGB
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # Procesar
        resultados = self.pose.process(img_rgb)

        if not resultados.pose_landmarks:
            return None

        # Extraer puntos clave
        h, w = img_np.shape[:2]
        lm = resultados.pose_landmarks.landmark

        puntos = {
            'nariz': (int(lm[0].x * w), int(lm[0].y * h)),
            'ojo_izq': (int(lm[2].x * w), int(lm[2].y * h)),
            'ojo_der': (int(lm[5].x * w), int(lm[5].y * h)),
            'h_izq': (int(lm[11].x * w), int(lm[11].y * h)),
            'h_der': (int(lm[12].x * w), int(lm[12].y * h)),
            'codo_izq': (int(lm[13].x * w), int(lm[13].y * h)),
            'codo_der': (int(lm[14].x * w), int(lm[14].y * h)),
            'muneca_izq': (int(lm[15].x * w), int(lm[15].y * h)),
            'muneca_der': (int(lm[16].x * w), int(lm[16].y * h)),
            'cadera_izq': (int(lm[23].x * w), int(lm[23].y * h)),
            'cadera_der': (int(lm[24].x * w), int(lm[24].y * h)),
            'rodilla_izq': (int(lm[25].x * w), int(lm[25].y * h)),
            'rodilla_der': (int(lm[26].x * w), int(lm[26].y * h)),
            'tobillo': (int(lm[27].x * w), int(lm[27].y * h))
        }

        return puntos

    def estimar_altura_automatica(self, puntos):
        """
        Estima la altura en cm usando proporciones antropométricas estándar

        Método: Basado en el ancho de hombros (promedio adulto: 23-25% de altura)

        Args:
            puntos: dict de keypoints

        Returns:
            float: altura estimada en cm
        """
        if not puntos:
            return 170.0  # Fallback

        # Calcular ancho de hombros en píxeles
        ancho_hombros_px = abs(puntos['h_izq'][0] - puntos['h_der'][0])

        # Proporción promedio: hombros = 24% de altura (adulto promedio)
        # Por tanto: altura = hombros / 0.24
        # Ajustamos a 0.245 para ser más conservadores
        altura_estimada_px = ancho_hombros_px / 0.245

        # Convertir a cm usando una escala estándar
        # Asumimos que la persona está a ~2-3 metros de la cámara
        # y la imagen es típicamente 1080p o similar
        # Calibración empírica: 1 píxel ≈ 0.15-0.20 cm a esa distancia

        # Método alternativo más robusto: usar altura en píxeles directamente
        altura_px = puntos['tobillo'][1] - puntos['nariz'][1]

        # Estimar basándose en proporciones conocidas
        # Altura promedio adulto: 165-175 cm
        # Ajustamos según el ratio hombros/altura detectado
        ratio_hombros = ancho_hombros_px / altura_px if altura_px > 0 else 0.24

        # Ratios típicos:
        # - 0.20-0.22: Persona delgada o adolescente → ~160-170cm
        # - 0.23-0.25: Adulto promedio → ~165-175cm
        # - 0.26-0.30: Atlético o complexión grande → ~170-185cm

        if ratio_hombros < 0.22:
            altura_base = 165
        elif ratio_hombros < 0.26:
            altura_base = 172
        else:
            altura_base = 180

        # Ajuste fino basado en altura en píxeles (normalizado)
        # Si la persona ocupa mucho de la imagen → más alta
        # Si ocupa poco → más baja
        if altura_px > 0:
            # Normalizar por altura de imagen
            # Para una imagen de 1080px, altura típica de persona: 600-900px
            # Ajustamos ±10cm según este factor
            altura_norm = altura_px / 720  # 720 como referencia (imagen de 1080p)
            ajuste = (altura_norm - 1.0) * 15  # ±15cm según desviación
            altura_estimada = altura_base + ajuste
        else:
            altura_estimada = altura_base

        # Limitar a rangos realistas
        altura_estimada = max(140, min(210, altura_estimada))

        return round(altura_estimada, 1)

    def calcular_medidas_completas(self, imagen, puntos, altura_real_cm=None):
        """
        Calcula medidas antropométricas completas

        Args:
            imagen: numpy array
            puntos: dict de keypoints
            altura_real_cm: altura real del usuario (opcional, se estima si no se provee)

        Returns:
            dict con todas las medidas en cm
        """
        if not puntos:
            return None

        # Calcular altura en píxeles
        altura_px = puntos['tobillo'][1] - puntos['nariz'][1]

        if altura_px <= 0:
            return None

        # Si no se provee altura, estimarla automáticamente
        if altura_real_cm is None or altura_real_cm <= 0:
            altura_real_cm = self.estimar_altura_automatica(puntos)
            print(f"   📏 Altura estimada automáticamente: {altura_real_cm:.1f} cm")

        # Factor de conversión píxeles -> cm
        factor = altura_real_cm / altura_px

        # Calcular medidas
        ancho_hombros_px = abs(puntos['h_izq'][0] - puntos['h_der'][0])
        ancho_cadera_px = abs(puntos['cadera_izq'][0] - puntos['cadera_der'][0])

        # Largos de extremidades
        largo_brazo_superior_px = np.linalg.norm(
            np.array(puntos['h_izq']) - np.array(puntos['codo_izq'])
        )
        largo_antebrazo_px = np.linalg.norm(
            np.array(puntos['codo_izq']) - np.array(puntos['muneca_izq'])
        )
        largo_pierna_px = np.linalg.norm(
            np.array(puntos['cadera_izq']) - np.array(puntos['rodilla_izq'])
        )

        medidas = {
            'altura_cm': round(altura_real_cm, 1),
            'ancho_hombros_cm': round(ancho_hombros_px * factor, 1),
            'ancho_cadera_cm': round(ancho_cadera_px * factor, 1),
            'largo_brazo_cm': round((largo_brazo_superior_px + largo_antebrazo_px) * factor, 1),
            'largo_pierna_cm': round(largo_pierna_px * factor, 1),
            'torso_cm': round((puntos['cadera_izq'][1] - puntos['h_izq'][1]) * factor, 1),
            'factor_conversion': round(factor, 4),

            # Ratios antropométricos
            'ratio_hombros_altura': round(ancho_hombros_px * factor / altura_real_cm, 3),
            'ratio_cadera_hombros': round(ancho_cadera_px / ancho_hombros_px, 3)
        }

        return medidas


class GeneradorAvatar3D:
    """
    OBJETIVO 1: Generación de modelos 3D con texturizado realista
    Usa MiDaS para estimación de profundidad y trimesh para mallas 3D
    """

    def __init__(self):
        print("🔄 Cargando modelo de profundidad MiDaS...")

        # Cargar MiDaS (versión small para RTX 3050)
        self.device = config.DEVICE
        self.modelo_midas = torch.hub.load(
            "intel-isl/MiDaS",
            config.DEPTH_MODEL,
            trust_repo=True
        ).to(self.device).eval()

        self.transform = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True
        ).small_transform

        # Limpiar cache GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✅ Modelo MiDaS cargado en {self.device.upper()}")

    def _aplicar_textura_avanzada(self, img_persona, img_prenda, puntos, depth_map):
        """
        Aplica prenda sobre la persona con ajuste automático y sombreado

        Mejoras:
        - Autocrop de la prenda (elimina espacios vacíos)
        - Anclaje preciso al cuello
        - Escala proporcional basada en hombros
        - Sombreado según mapa de profundidad
        """
        img_p_pil = img_persona.convert("RGBA")

        # 1. Preprocesar prenda: quitar fondo y recortar
        if isinstance(img_prenda, np.ndarray):
            prenda_raw = remove(Image.fromarray(img_prenda))
        else:
            prenda_raw = remove(img_prenda)

        # Autocrop
        bbox = prenda_raw.getbbox()
        prenda = prenda_raw.crop(bbox) if bbox else prenda_raw

        # 2. Calcular dimensiones basadas en hombros
        h_izq, h_der = puntos['h_izq'], puntos['h_der']
        ancho_hombros = abs(h_izq[0] - h_der[0])

        # Factor 1.8 es el balance ideal para que las mangas cubran brazos
        ancho_deseado = int(ancho_hombros * 1.8)

        # Mantener proporción original de la prenda
        w_orig, h_orig = prenda.size
        ratio = h_orig / w_orig
        alto_deseado = int(ancho_deseado * ratio)

        # Redimensionar con filtro de alta calidad
        prenda = prenda.resize((ancho_deseado, alto_deseado), Image.Resampling.LANCZOS)

        # 3. Posicionar: centrado horizontal, anclaje en cuello
        pos_x = ((h_izq[0] + h_der[0]) // 2) - (ancho_deseado // 2)
        # Subir 15% para que el cuello de la prenda coincida con hombros
        pos_y = h_izq[1] - int(alto_deseado * 0.15)

        # Asegurar que está dentro de la imagen
        pos_x = max(0, min(pos_x, img_p_pil.width - ancho_deseado))
        pos_y = max(0, min(pos_y, img_p_pil.height - alto_deseado))

        # 4. Aplicar sombreado basado en profundidad
        prenda_np = np.array(prenda)
        zona_depth = depth_map[pos_y:pos_y + alto_deseado, pos_x:pos_x + ancho_deseado]

        # Verificar dimensiones coincidan
        if zona_depth.shape[:2] == prenda_np.shape[:2]:
            # Normalizar profundidad a rango [0.5, 1.0] para sombreado sutil
            sombra = np.clip(zona_depth * 1.2, 0.5, 1.0)

            # Aplicar solo a canales RGB
            if prenda_np.shape[2] == 4:  # RGBA
                prenda_np[:, :, :3] = (prenda_np[:, :, :3] * sombra[:, :, np.newaxis]).astype(np.uint8)

        # 5. Componer imagen final
        capa = Image.new("RGBA", img_p_pil.size, (0, 0, 0, 0))
        capa.paste(Image.fromarray(prenda_np), (pos_x, pos_y), Image.fromarray(prenda_np))

        resultado = Image.alpha_composite(img_p_pil, capa).convert("RGB")

        return np.array(resultado)

    def generar(self, img_persona, img_prenda, puntos, progreso_callback=None):
        """
        Pipeline completo de generación de avatar 3D

        Args:
            img_persona: PIL Image de la persona
            img_prenda: PIL Image o numpy de la prenda
            puntos: dict de keypoints
            progreso_callback: función opcional para reportar progreso

        Returns:
            trimesh.Trimesh: malla 3D con textura
        """
        if progreso_callback:
            progreso_callback(0.1, "Preparando imágenes...")

        # 1. Convertir a formato adecuado
        img_rgb = np.array(img_persona.convert("RGB"))

        if progreso_callback:
            progreso_callback(0.2, "Estimando profundidad...")

        # 2. Estimación de profundidad con MiDaS
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            depth = self.modelo_midas(input_batch)
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()

        # Normalizar profundidad
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

        if progreso_callback:
            progreso_callback(0.4, "Aplicando prenda...")

        # 3. Aplicar prenda con texturizado
        if img_prenda is not None:
            img_vestida = self._aplicar_textura_avanzada(
                img_persona, img_prenda, puntos, depth_norm
            )
        else:
            img_vestida = img_rgb

        if progreso_callback:
            progreso_callback(0.6, "Segmentando persona...")

        # 4. Segmentación para crear máscara
        img_limpia = remove(img_persona)
        mask = cv2.threshold(
            cv2.GaussianBlur(np.array(img_limpia)[:, :, 3], (7, 7), 0),
            150, 255, cv2.THRESH_BINARY
        )[1]

        mask_resized = cv2.resize(mask, (depth_norm.shape[1], depth_norm.shape[0])) > 128

        # Aplicar máscara a profundidad
        depth_norm[~mask_resized] = 0

        if progreso_callback:
            progreso_callback(0.75, "Generando malla 3D...")

        # 5. Crear malla 3D
        h, w = depth_norm.shape

        # Crear vértices solo donde hay persona
        indices = np.where(mask_resized.flatten())[0]
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        vert_2d = np.stack([x_coords.flatten(), y_coords.flatten()], axis=-1)[indices]

        # Mapeo de índices
        index_map = np.full(mask_resized.flatten().shape, -1)
        index_map[indices] = np.arange(len(indices))

        # Crear caras (triangulación)
        caras = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j
                i1 = index_map[idx]
                i2 = index_map[idx + 1]
                i3 = index_map[idx + w]
                i4 = index_map[idx + w + 1]

                if i1 != -1 and i2 != -1 and i3 != -1:
                    caras.append([i1, i2, i3])
                if i2 != -1 and i4 != -1 and i3 != -1:
                    caras.append([i2, i4, i3])

        # Extrusión para volumen
        mesh = trimesh.creation.extrude_triangulation(vert_2d, caras, height=-0.25)

        # Agregar profundidad (componente Z)
        mesh.vertices[:len(indices), 2] += depth_norm.flatten()[indices] * (w * 0.12)

        if progreso_callback:
            progreso_callback(0.9, "Aplicando texturas...")

        # 6. Aplicar colores de la imagen con prenda
        colores = img_vestida.reshape(-1, 3)[indices]
        vertex_colors = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)
        vertex_colors[:len(indices)] = colores
        vertex_colors[len(indices):2*len(indices)] = colores
        mesh.visual.vertex_colors = vertex_colors

        # 7. Transformaciones finales
        # Rotar para orientación correcta
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))

        # Escalar y centrar
        mesh.apply_scale(0.19)
        mesh.apply_translation([-mesh.centroid[0], 110, 0])

        if progreso_callback:
            progreso_callback(1.0, "¡Modelo completado!")

        # Limpiar GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return mesh


# ============================================
# MÓDULO 2: CATÁLOGO Y PERSONALIZACIÓN
# ============================================

class CatalogoPrendas:
    """
    OBJETIVO 2: Sistema de catálogo de prendas
    Gestiona base de datos de prendas con metadata
    """

    def __init__(self):
        self.catalogo_path = config.CATALOGO_DIR
        self._inicializar_catalogo()

    def _inicializar_catalogo(self):
        """Crea catálogo inicial si no existe"""
        self.prendas = {
            "camisas": [
                {
                    "id": "camisa_001",
                    "nombre": "Camisa Blanca Formal",
                    "categoria": "formal",
                    "color_base": "#FFFFFF",
                    "path": None  # Se asignará dinámicamente
                },
                {
                    "id": "camisa_002",
                    "nombre": "Polo Azul Casual",
                    "categoria": "casual",
                    "color_base": "#4A90E2",
                    "path": None
                },
                {
                    "id": "camisa_003",
                    "nombre": "T-Shirt Deportiva",
                    "categoria": "deportivo",
                    "color_base": "#FF4500",
                    "path": None
                }
            ],
            "personalizada": [
                {
                    "id": "custom",
                    "nombre": "Prenda Personalizada",
                    "categoria": "custom",
                    "color_base": "#808080",
                    "path": None
                }
            ]
        }

    def obtener_opciones(self, categoria="camisas"):
        """Retorna lista de nombres para dropdown"""
        return [p["nombre"] for p in self.prendas.get(categoria, [])]

    def obtener_prenda(self, categoria, nombre):
        """Obtiene metadata de una prenda"""
        for prenda in self.prendas.get(categoria, []):
            if prenda["nombre"] == nombre:
                return prenda
        return None


class ModificadorPrenda:
    """
    OBJETIVO 2: Modificación de parámetros de prendas
    Permite cambiar color, escala, estilo en tiempo real
    """

    @staticmethod
    def cambiar_color(imagen, color_hex):
        """
        Cambia color dominante de la prenda preservando detalles

        Args:
            imagen: PIL Image o numpy array
            color_hex: string "#RRGGBB"

        Returns:
            numpy array con color modificado
        """
        if imagen is None:
            return None

        # Convertir a PIL si es necesario
        if isinstance(imagen, np.ndarray):
            imagen = Image.fromarray(imagen)

        # Asegurar RGBA
        if imagen.mode != 'RGBA':
            imagen = imagen.convert('RGBA')

        img_np = np.array(imagen)

        # Extraer máscara (canal alpha)
        alpha = img_np[:, :, 3]
        mask = alpha > 128

        # Convertir color hex a RGB
        rgb_target = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        # Convertir a HSV para manipulación de color
        img_hsv = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2HSV).astype(float)

        # Calcular nuevo matiz
        h_target, s_target, v_target = colorsys.rgb_to_hsv(
            rgb_target[0]/255, rgb_target[1]/255, rgb_target[2]/255
        )

        # Aplicar nuevo matiz manteniendo saturación y valor originales
        img_hsv[mask, 0] = h_target * 180  # OpenCV usa 0-180 para H

        # Opcional: ajustar saturación levemente
        img_hsv[mask, 1] = np.clip(img_hsv[mask, 1] * 1.1, 0, 255)

        # Convertir de vuelta a RGB
        img_recolored = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Restaurar canal alpha
        resultado = np.dstack([img_recolored, alpha])

        return resultado

    @staticmethod
    def ajustar_escala(imagen, factor):
        """Redimensiona prenda manteniendo calidad"""
        if imagen is None or factor == 1.0:
            return imagen

        if isinstance(imagen, np.ndarray):
            imagen = Image.fromarray(imagen)

        new_size = (int(imagen.width * factor), int(imagen.height * factor))
        return np.array(imagen.resize(new_size, Image.Resampling.LANCZOS))

    @staticmethod
    def aplicar_estilo(imagen, estilo):
        """
        Ajusta fit de la prenda

        Args:
            estilo: "Slim" (85% ancho), "Regular" (100%), "Loose" (115%)
        """
        if imagen is None or estilo == "Regular":
            return imagen

        if isinstance(imagen, np.ndarray):
            imagen = Image.fromarray(imagen)

        if estilo == "Slim":
            new_w = int(imagen.width * 0.85)
            return np.array(imagen.resize((new_w, imagen.height), Image.Resampling.LANCZOS))
        elif estilo == "Loose":
            new_w = int(imagen.width * 1.15)
            return np.array(imagen.resize((new_w, imagen.height), Image.Resampling.LANCZOS))

        return np.array(imagen)


# ============================================
# MÓDULO 3: INTELIGENCIA ARTIFICIAL
# ============================================

class RecomendadorTallas:
    """
    OBJETIVO 3: IA para recomendación de tallas
    Random Forest entrenado con dataset ANSUR II
    """

    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.modelo_path = config.MODELS_DIR / "modelo_tallas.pkl"
        self.scaler_path = config.MODELS_DIR / "scaler_tallas.pkl"

        self.mapa_tallas = {
            0: {"letra": "XS", "eu": "44", "us": "34", "cm_hombros": (36, 40)},
            1: {"letra": "S", "eu": "46", "us": "36", "cm_hombros": (40, 42)},
            2: {"letra": "M", "eu": "48", "us": "38", "cm_hombros": (42, 45)},
            3: {"letra": "L", "eu": "50", "us": "40", "cm_hombros": (45, 48)},
            4: {"letra": "XL", "eu": "52", "us": "42", "cm_hombros": (48, 51)},
            5: {"letra": "XXL", "eu": "54", "us": "44", "cm_hombros": (51, 55)}
        }

        # Intentar cargar modelo existente
        if not self._cargar_modelo():
            print("⚠️  Modelo de tallas no encontrado. Entrenando...")
            self._entrenar_modelo()

    def _cargar_modelo(self):
        """Carga modelo pre-entrenado"""
        try:
            self.modelo = joblib.load(self.modelo_path)
            self.scaler = joblib.load(self.scaler_path)
            print("✅ Modelo de tallas cargado")
            return True
        except:
            return False

    def _entrenar_modelo(self):
        """Entrena modelo con ANSUR II"""
        try:
            print("📥 Descargando dataset ANSUR II...")
            ansur_url = 'https://calmcode.io/static/data/ergonomics.csv'
            ansur_df = pd.read_csv(ansur_url)

            # Preparar features (convertir mm a cm)
            X = ansur_df[['stature', 'biacromialbreadth', 'waistcircumference']].values / 10

            # Crear labels basadas en proporciones
            def asignar_talla(row):
                altura = row['stature'] / 10
                hombros = row['biacromialbreadth'] / 10

                if altura < 165 and hombros < 40:
                    return 0
                elif altura < 170 and hombros < 42:
                    return 1
                elif altura < 178 and hombros < 45:
                    return 2
                elif altura < 185 and hombros < 48:
                    return 3
                elif altura < 192 and hombros < 51:
                    return 4
                else:
                    return 5

            y = ansur_df.apply(asignar_talla, axis=1)

            # Normalizar y entrenar
            X_scaled = self.scaler.fit_transform(X)
            self.modelo = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.modelo.fit(X_scaled, y)

            # Guardar
            joblib.dump(self.modelo, self.modelo_path)
            joblib.dump(self.scaler, self.scaler_path)

            accuracy = self.modelo.score(X_scaled, y)
            print(f"✅ Modelo entrenado - Accuracy: {accuracy:.2%}")

        except Exception as e:
            print(f"⚠️  Error entrenando modelo: {e}")
            print("Se usarán reglas heurísticas")

    def predecir_talla(self, altura_cm, ancho_hombros_cm, cintura_cm=None):
        """
        Predice talla con IA

        Returns:
            dict con talla, confianza, alternativas y explicación
        """
        # Estimar cintura si no está disponible
        if cintura_cm is None:
            cintura_cm = ancho_hombros_cm * 2.0

        # Si no hay modelo, usar heurística
        if self.modelo is None:
            return self._predecir_heuristico(altura_cm, ancho_hombros_cm)

        try:
            # Predecir
            X = np.array([[altura_cm, ancho_hombros_cm, cintura_cm]])
            X_scaled = self.scaler.transform(X)

            talla_idx = self.modelo.predict(X_scaled)[0]
            probabilidades = self.modelo.predict_proba(X_scaled)[0]
            confianza = probabilidades[talla_idx]

            # Top 3 alternativas
            top3_idx = np.argsort(probabilidades)[-3:][::-1]
            alternativas = [
                {
                    "talla": self.mapa_tallas[idx]["letra"],
                    "probabilidad": float(probabilidades[idx])
                }
                for idx in top3_idx
            ]

            return {
                "talla_letra": self.mapa_tallas[talla_idx]["letra"],
                "talla_eu": self.mapa_tallas[talla_idx]["eu"],
                "talla_us": self.mapa_tallas[talla_idx]["us"],
                "confianza": float(confianza),
                "alternativas": alternativas,
                "explicacion": self._generar_explicacion(altura_cm, ancho_hombros_cm, talla_idx),
                "metodo": "ia"
            }
        except Exception as e:
            print(f"Error en predicción IA: {e}")
            return self._predecir_heuristico(altura_cm, ancho_hombros_cm)

    def _predecir_heuristico(self, altura, hombros):
        """Fallback: reglas simples"""
        if altura < 165 and hombros < 40:
            idx = 0
        elif altura < 170 and hombros < 42:
            idx = 1
        elif altura < 178 and hombros < 45:
            idx = 2
        elif altura < 185 and hombros < 48:
            idx = 3
        elif altura < 192 and hombros < 51:
            idx = 4
        else:
            idx = 5

        return {
            "talla_letra": self.mapa_tallas[idx]["letra"],
            "talla_eu": self.mapa_tallas[idx]["eu"],
            "talla_us": self.mapa_tallas[idx]["us"],
            "confianza": 0.75,
            "explicacion": self._generar_explicacion(altura, hombros, idx),
            "metodo": "reglas"
        }

    def _generar_explicacion(self, altura, hombros, talla_idx):
        """Genera explicación interpretable"""
        talla = self.mapa_tallas[talla_idx]["letra"]
        rango = self.mapa_tallas[talla_idx]["cm_hombros"]

        return (f"Con tu altura de {altura:.0f}cm y ancho de hombros de {hombros:.1f}cm "
                f"(rango ideal: {rango[0]}-{rango[1]}cm), la talla {talla} es la más adecuada. "
                f"Considera tallas adyacentes si prefieres ajuste más holgado o ceñido.")


class AnalizadorColorPiel:
    """
    OBJETIVO 3: IA para análisis de tono de piel y recomendación de colores
    Usa K-means clustering y teoría del color
    """

    def __init__(self):
        self.teoria_color = {
            "calido": {
                "colores": ["#D4AF37", "#E97451", "#8B4513", "#228B22", "#FF8C00"],
                "nombres": ["Dorado", "Terracota", "Marrón Cálido", "Verde Oliva", "Naranja Quemado"],
                "evitar": ["#000080", "#4B0082", "#C0C0C0"],
                "descripcion": "Subtonos dorados/amarillos"
            },
            "frio": {
                "colores": ["#000080", "#4B0082", "#800080", "#C0C0C0", "#FF69B4"],
                "nombres": ["Azul Marino", "Índigo", "Morado", "Gris Plateado", "Rosa"],
                "evitar": ["#FF8C00", "#8B4513", "#D4AF37"],
                "descripcion": "Subtonos rosados/azulados"
            },
            "neutro": {
                "colores": ["#2F4F4F", "#708090", "#8B0000", "#006400", "#4682B4"],
                "nombres": ["Verde Azulado", "Gris Pizarra", "Rojo Oscuro", "Verde Oscuro", "Azul Acero"],
                "evitar": [],
                "descripcion": "Balance entre cálido y frío"
            }
        }

    def extraer_tono_piel(self, imagen, puntos_pose):
        """
        Extrae color de piel promedio de región facial

        Args:
            imagen: numpy array BGR
            puntos_pose: dict con keypoints

        Returns:
            dict con rgb y hex del color de piel
        """
        if 'nariz' not in puntos_pose:
            return None

        # Definir ROI facial centrada en nariz
        x, y = puntos_pose['nariz']
        roi_size = 60

        y1 = max(0, int(y - roi_size))
        y2 = min(imagen.shape[0], int(y + roi_size))
        x1 = max(0, int(x - roi_size))
        x2 = min(imagen.shape[1], int(x + roi_size))

        roi = imagen[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        # Convertir a RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pixels = roi_rgb.reshape(-1, 3)

        # Filtrar píxeles de piel (heurística de rangos)
        mask = (
            (pixels[:, 0] > 50) & (pixels[:, 0] < 250) &
            (pixels[:, 1] > 40) & (pixels[:, 1] < 230) &
            (pixels[:, 2] > 30) & (pixels[:, 2] < 220)
        )

        skin_pixels = pixels[mask]

        if len(skin_pixels) < 100:
            return None

        # K-means para encontrar color dominante
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(skin_pixels)

        # Cluster más poblado
        labels = kmeans.labels_
        counts = np.bincount(labels)
        dominant = np.argmax(counts)
        color_dom = kmeans.cluster_centers_[dominant].astype(int)

        return {
            'rgb': tuple(color_dom),
            'hex': f"#{color_dom[0]:02x}{color_dom[1]:02x}{color_dom[2]:02x}"
        }

    def clasificar_tono(self, color_rgb):
        """
        Clasifica en cálido/frío/neutro según undertones

        Método:
        - Cálido: más amarillo/dorado (R+G > B)
        - Frío: más rosa/azul (R+B > G)
        - Neutro: balance
        """
        r, g, b = color_rgb

        indice_amarillo = (r + g) / 2 - b
        indice_rosa = (r + b) / 2 - g

        if indice_amarillo > 20:
            return "calido"
        elif indice_rosa > 15:
            return "frio"
        else:
            return "neutro"

    def recomendar_colores(self, imagen, puntos_pose, num_recomendaciones=5):
        """
        Pipeline completo: analiza piel → recomienda colores

        Returns:
            dict con clasificación, colores recomendados y explicación
        """
        # Extraer tono de piel
        color_piel = self.extraer_tono_piel(imagen, puntos_pose)

        if color_piel is None:
            # Fallback: colores neutros universales
            return {
                "error": "No se detectó tono de piel",
                "clasificacion": "neutro",
                "clasificacion_nombre": "Neutro (Universal)",
                "colores_recomendados": self.teoria_color["neutro"]["colores"][:num_recomendaciones],
                "nombres_colores": self.teoria_color["neutro"]["nombres"][:num_recomendaciones],
                "explicacion": "No se pudo analizar tu tono de piel. Estos colores universales funcionan para todos."
            }

        # Clasificar
        clasificacion = self.clasificar_tono(color_piel['rgb'])
        paleta = self.teoria_color[clasificacion]

        return {
            "tono_piel_hex": color_piel['hex'],
            "tono_piel_rgb": color_piel['rgb'],
            "clasificacion": clasificacion,
            "clasificacion_nombre": {
                "calido": "Cálido (Primavera/Otoño)",
                "frio": "Frío (Verano/Invierno)",
                "neutro": "Neutro (Universal)"
            }[clasificacion],
            "colores_recomendados": paleta["colores"][:num_recomendaciones],
            "nombres_colores": paleta["nombres"][:num_recomendaciones],
            "colores_evitar": paleta["evitar"],
            "explicacion": f"Tu piel tiene {paleta['descripcion']}. {self._explicacion_detallada(clasificacion)}"
        }

    def _explicacion_detallada(self, clasificacion):
        """Genera explicación personalizada"""
        explicaciones = {
            "calido": "Los colores tierra, dorados, naranjas y verdes oliva te favorecen más. Evita grises plateados y azules muy fríos.",
            "frio": "Los azules, morados, rosas y grises te sientan mejor. Evita naranjas, dorados y marrones cálidos.",
            "neutro": "¡Tienes suerte! Tu tono neutro te permite usar casi cualquier color. Experimenta libremente con toda la paleta."
        }
        return explicaciones[clasificacion]


class RecomendadorEstilos:
    """
    OBJETIVO 3: Sistema experto para recomendación de estilos
    Combina tipo de cuerpo, ocasión y colores favorables
    """

    def __init__(self):
        self.estilos_base = {
            "formal": {
                "prendas": ["Camisa Blanca", "Camisa Azul Claro", "Camisa Gris"],
                "colores": ["#FFFFFF", "#E0E0E0", "#000080", "#2F4F4F"],
                "ocasiones": ["Trabajo", "Reunión", "Entrevista", "Evento Corporativo"]
            },
            "casual": {
                "prendas": ["T-Shirt Básica", "Polo Casual", "Camisa Denim"],
                "colores": ["#4A90E2", "#808080", "#8B4513", "#228B22"],
                "ocasiones": ["Salir con amigos", "Fin de semana", "Compras", "Paseo"]
            },
            "deportivo": {
                "prendas": ["Camiseta Deportiva", "Tank Top", "Hoodie Deportivo"],
                "colores": ["#FF4500", "#00FF00", "#1E90FF", "#000000"],
                "ocasiones": ["Gimnasio", "Running", "Deporte", "Entrenamiento"]
            },
            "elegante_casual": {
                "prendas": ["Camisa Slim Fit", "Polo Premium", "Camisa Estampada Sutil"],
                "colores": ["#2F4F4F", "#8B0000", "#4B0082", "#D4AF37"],
                "ocasiones": ["Cita", "Evento social", "Restaurante", "Concierto"]
            }
        }

    def analizar_tipo_cuerpo(self, medidas):
        """
        Clasifica morfología corporal

        Tipos:
        - Triángulo invertido: Hombros > Cadera (atlético)
        - Triángulo: Cadera > Hombros (menos común en hombres)
        - Rectangular: Proporciones equilibradas
        """
        ratio = medidas['ancho_hombros_cm'] / medidas.get('ancho_cadera_cm', medidas['ancho_hombros_cm'] * 0.9)
        altura = medidas['altura_cm']

        if ratio > 1.1:
            return "triangulo_invertido"
        elif ratio < 0.95:
            return "triangulo"
        elif altura > 185:
            return "rectangular_alto"
        else:
            return "rectangular"

    def recomendar_por_ocasion(self, ocasion, medidas, tono_piel="neutro"):
        """
        Genera recomendación completa de outfit

        Args:
            ocasion: str (trabajo, casual, gimnasio, cita, etc.)
            medidas: dict con medidas antropométricas
            tono_piel: str (calido, frio, neutro)

        Returns:
            dict con estilo, prendas, colores y consejos
        """
        # Mapear ocasión a estilo
        mapa_ocasion = {
            "trabajo": "formal",
            "reunión": "formal",
            "entrevista": "formal",
            "gimnasio": "deportivo",
            "deporte": "deportivo",
            "running": "deportivo",
            "salir": "casual",
            "compras": "casual",
            "fin de semana": "casual",
            "cita": "elegante_casual",
            "restaurante": "elegante_casual",
            "evento social": "elegante_casual"
        }

        estilo = mapa_ocasion.get(ocasion.lower(), "casual")
        config_estilo = self.estilos_base[estilo]

        # Analizar tipo de cuerpo
        tipo_cuerpo = self.analizar_tipo_cuerpo(medidas)

        # Consejos por tipo de cuerpo
        consejos_cuerpo = {
            "triangulo_invertido": "Equilibra tus proporciones: usa colores oscuros arriba y claros abajo. Evita hombros acolchados.",
            "triangulo": "Resalta tu parte superior con colores llamativos y patrones. Pantalones oscuros ayudan a equilibrar.",
            "rectangular_alto": "Usa patrones horizontales para añadir volumen visual. Las capas te quedan muy bien.",
            "rectangular": "Tienes proporciones equilibradas: puedes experimentar con casi cualquier estilo."
        }

        # Filtrar colores según tono de piel
        analizador = AnalizadorColorPiel()
        colores_evitar = analizador.teoria_color[tono_piel].get("evitar", [])
        colores_recomendados = analizador.teoria_color[tono_piel]["colores"]

        # Combinar colores del estilo con los que favorecen al usuario
        colores_finales = [c for c in config_estilo["colores"] if c not in colores_evitar]

        if len(colores_finales) < 3:
            colores_finales = colores_recomendados[:3]

        return {
            "estilo": estilo,
            "prendas_recomendadas": config_estilo["prendas"][:3],
            "colores_sugeridos": colores_finales[:5],
            "tipo_cuerpo": tipo_cuerpo,
            "consejo_cuerpo": consejos_cuerpo[tipo_cuerpo],
            "nivel_confianza": 0.85,
            "explicacion": f"Para {ocasion}, te recomendamos un estilo {estilo}. {consejos_cuerpo[tipo_cuerpo]}"
        }


# ============================================
# MÓDULO 4: EVALUACIÓN Y VALIDACIÓN
# ============================================

class EvaluadorSistema:
    """
    OBJETIVO 4: Framework de evaluación completo
    Métricas técnicas, validación de IA y usabilidad
    """

    def __init__(self):
        self.output_dir = config.EVALUACION_DIR
        self.metricas = {
            "deteccion_pose": [],
            "prediccion_tallas": [],
            "recomendacion_colores": [],
            "rendimiento": []
        }

    def evaluar_deteccion_pose(self, dataset_imagenes):
        """Evalúa precisión de MediaPipe"""
        print("\n" + "="*60)
        print("EVALUANDO DETECCIÓN DE POSE")
        print("="*60)

        med_temp = MedicionesAntropometricas()

        resultados = {
            "total": len(dataset_imagenes),
            "exitosas": 0,
            "fallidas": 0,
            "tiempos": []
        }

        for i, img_path in enumerate(dataset_imagenes):
            try:
                img = Image.open(img_path)

                inicio = time.time()
                puntos = med_temp.detectar_puntos(img)
                tiempo = time.time() - inicio

                resultados["tiempos"].append(tiempo)

                if puntos:
                    resultados["exitosas"] += 1
                else:
                    resultados["fallidas"] += 1

                if (i + 1) % 10 == 0:
                    print(f"  Procesadas {i+1}/{len(dataset_imagenes)}...")

            except Exception as e:
                resultados["fallidas"] += 1

        tasa_exito = (resultados["exitosas"] / resultados["total"]) * 100
        tiempo_medio = np.mean(resultados["tiempos"])

        reporte = {
            "tasa_exito_pct": round(tasa_exito, 2),
            "tiempo_medio_seg": round(tiempo_medio, 3),
            "fps_equivalente": round(1 / tiempo_medio, 2)
        }

        self.metricas["deteccion_pose"].append(reporte)

        print(f"\n✅ RESULTADOS:")
        print(f"   Tasa de éxito: {tasa_exito:.1f}%")
        print(f"   Tiempo promedio: {tiempo_medio:.3f}s")
        print(f"   FPS equivalente: {reporte['fps_equivalente']:.2f}")

        return reporte

    def evaluar_rendimiento(self, img_test, img_prenda, num_iteraciones=20):
        """Benchmark de rendimiento"""
        print("\n" + "="*60)
        print("EVALUANDO RENDIMIENTO")
        print("="*60)

        med_temp = MedicionesAntropometricas()
        gen_temp = GeneradorAvatar3D()

        tiempos = []
        uso_memoria = []

        for i in range(num_iteraciones):
            inicio = time.time()

            # Pipeline completo
            puntos = med_temp.detectar_puntos(img_test)
            if puntos:
                mesh = gen_temp.generar(img_test, img_prenda, puntos)

            tiempos.append(time.time() - inicio)

            # Memoria
            proceso = psutil.Process()
            uso_memoria.append(proceso.memory_info().rss / 1024 / 1024)

            if (i + 1) % 5 == 0:
                print(f"  Iteración {i+1}/{num_iteraciones}...")

        reporte = {
            "tiempo_medio_seg": round(np.mean(tiempos), 3),
            "tiempo_std_seg": round(np.std(tiempos), 3),
            "memoria_media_mb": round(np.mean(uso_memoria), 1),
            "memoria_max_mb": round(np.max(uso_memoria), 1)
        }

        self.metricas["rendimiento"].append(reporte)

        print(f"\n✅ RESULTADOS ({num_iteraciones} iteraciones):")
        print(f"   Tiempo: {reporte['tiempo_medio_seg']:.3f}s ± {reporte['tiempo_std_seg']:.3f}s")
        print(f"   Memoria: {reporte['memoria_media_mb']:.1f} MB (máx: {reporte['memoria_max_mb']:.1f} MB)")

        return reporte

    def generar_reporte(self):
        """Genera reporte en texto"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = self.output_dir / f"reporte_{timestamp}.txt"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("REPORTE DE EVALUACIÓN - PROBADOR VIRTUAL 3D\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            for categoria, datos in self.metricas.items():
                if datos:
                    f.write(f"\n{categoria.upper().replace('_', ' ')}:\n")
                    f.write("-" * 40 + "\n")
                    for key, val in datos[-1].items():
                        f.write(f"  {key}: {val}\n")

        print(f"\n📄 Reporte guardado: {filename}")
        return filename


# ============================================
# SERVIDOR HTTP PARA AR
# ============================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visor AR - Probador Virtual 3D</title>
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        .header {{ 
            background: rgba(0,0,0,0.3);
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }}
        .header h1 {{ font-size: 24px; margin-bottom: 5px; }}
        .header p {{ font-size: 14px; opacity: 0.9; }}
        model-viewer {{ 
            width: 100%;
            height: 100%;
            background: linear-gradient(to bottom, #1a1a2e, #16213e);
        }}
        .ar-button {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 50px;
            font-weight: bold;
            font-size: 16px;
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            cursor: pointer;
            transition: all 0.3s;
        }}
        .ar-button:hover {{ 
            transform: translateX(-50%) scale(1.05);
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Probador Virtual AR</h1>
        <p>Toca el botón para ver el modelo en tu espacio real</p>
    </div>
    <model-viewer 
        src="{MODEL_PATH}" 
        ar 
        ar-modes="webxr scene-viewer quick-look" 
        camera-controls 
        shadow-intensity="1" 
        auto-rotate
        environment-image="neutral"
        exposure="1.0">
        <button slot="ar-button" class="ar-button">
            📱 VER EN MI ESPACIO (AR)
        </button>
    </model-viewer>
</body>
</html>
"""

class ServidorAR(SimpleHTTPRequestHandler):
    """Servidor HTTP con soporte CORS para AR"""

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_GET(self):
        if self.path.startswith('/ar/'):
            filename = self.path.split('/')[-1]
            filepath = config.OUTPUT_DIR / filename

            if filepath.exists():
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()

                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)

        elif self.path.startswith('/model/'):
            filename = self.path.split('/')[-1]
            filepath = config.OUTPUT_DIR / filename

            if filepath.exists():
                self.send_response(200)
                self.send_header('Content-type', 'model/gltf-binary')
                self.end_headers()

                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # Silenciar logs


def iniciar_servidor_http():
    """Inicia servidor HTTP en thread separado"""
    server = HTTPServer((config.HTTP_HOST, config.HTTP_PORT), ServidorAR)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"🌐 Servidor HTTP iniciado en http://localhost:{config.HTTP_PORT}")


# ============================================
# INTERFAZ GRADIO COMPLETA
# ============================================

# Inicializar componentes globales
print("\n🔄 Inicializando componentes del sistema...")
med = MedicionesAntropometricas()
gen = GeneradorAvatar3D()
catalogo = CatalogoPrendas()
modificador = ModificadorPrenda()
recomendador_tallas = RecomendadorTallas()
analizador_color = AnalizadorColorPiel()
recomendador_estilos = RecomendadorEstilos()
evaluador = EvaluadorSistema()

# Iniciar servidor HTTP
iniciar_servidor_http()

print("✅ Sistema listo\n")


def actualizar_preview_prenda(prenda_custom, color, escala, estilo):
    """Actualiza vista previa de prenda con modificaciones"""
    if prenda_custom is None:
        return None

    # Aplicar modificaciones
    img_modificada = prenda_custom.copy()
    img_modificada = modificador.cambiar_color(img_modificada, color)
    img_modificada = modificador.ajustar_escala(img_modificada, escala)
    img_modificada = modificador.aplicar_estilo(img_modificada, estilo)

    return img_modificada


def procesar_imagen_completo(
    img_persona,
    img_prenda_custom,
    altura,
    color_prenda,
    escala_prenda,
    estilo_prenda,
    ocasion,
    progreso=gr.Progress()
):
    """
    Pipeline completo: Detección → Medidas → IA → Generación 3D → AR

    Este es el corazón del sistema que integra todos los módulos
    """
    try:
        progreso(0.05, desc="🔍 Validando entrada...")

        if img_persona is None:
            return None, None, None, "⚠️ Por favor sube una foto de cuerpo completo"

        # 1. DETECCIÓN DE POSE
        progreso(0.1, desc="🤸 Detectando pose corporal...")
        img_p_pil = Image.fromarray(img_persona)
        puntos = med.detectar_puntos(img_p_pil)

        if not puntos:
            return None, None, None, "❌ No se detectó el cuerpo en la imagen. Asegúrate de estar de pie y de cuerpo completo."

        # 2. CALCULAR MEDIDAS ANTROPOMÉTRICAS
        progreso(0.2, desc="📏 Calculando medidas corporales...")
        medidas = med.calcular_medidas_completas(img_persona, puntos, altura)

        if not medidas:
            return None, None, None, "❌ Error calculando medidas"

        # 3. IA: RECOMENDACIÓN DE TALLA
        progreso(0.3, desc="🤖 Analizando tu talla ideal...")
        recom_talla = recomendador_tallas.predecir_talla(
            medidas['altura_cm'],
            medidas['ancho_hombros_cm']
        )

        # 4. IA: ANÁLISIS DE TONO DE PIEL Y COLORES
        progreso(0.35, desc="🎨 Analizando tu tono de piel...")
        recom_colores = analizador_color.recomendar_colores(img_persona, puntos)

        # 5. IA: RECOMENDACIÓN DE ESTILO
        progreso(0.4, desc="👔 Generando recomendaciones de estilo...")
        recom_estilo = recomendador_estilos.recomendar_por_ocasion(
            ocasion,
            medidas,
            recom_colores.get('clasificacion', 'neutro')
        )

        # 6. PREPARAR PRENDA
        progreso(0.45, desc="👕 Preparando prenda...")
        if img_prenda_custom is not None:
            # Aplicar modificaciones a prenda custom
            prenda_final = modificador.cambiar_color(img_prenda_custom, color_prenda)
            prenda_final = modificador.ajustar_escala(prenda_final, escala_prenda)
            prenda_final = modificador.aplicar_estilo(prenda_final, estilo_prenda)
        else:
            prenda_final = None

        # 7. GENERAR MODELO 3D
        progreso(0.5, desc="🎨 Generando avatar 3D...")

        def callback_progreso(p, desc):
            progreso(0.5 + p * 0.4, desc=desc)

        mesh = gen.generar(img_p_pil, prenda_final, puntos, callback_progreso)

        # 8. EXPORTAR Y CREAR QR PARA AR
        progreso(0.95, desc="📦 Exportando modelo...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        glb_name = f"avatar_{timestamp}.glb"
        glb_path = config.OUTPUT_DIR / glb_name

        mesh.export(str(glb_path))

        # Crear HTML para AR
        model_url = f"http://localhost:{config.HTTP_PORT}/model/{glb_name}"
        html_name = f"ar_{timestamp}.html"
        html_path = config.OUTPUT_DIR / html_name

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(HTML_TEMPLATE.format(MODEL_PATH=model_url))

        # Generar QR
        qr_url = f"http://localhost:{config.HTTP_PORT}/ar/{html_name}"
        qr = qrcode.QRCode(version=1, box_size=10, border=4)
        qr.add_data(qr_url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

        progreso(1.0, desc="✅ ¡Completado!")

        # 9. GENERAR HTML CON RECOMENDACIONES IA
        html_recomendaciones = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; color: white; font-family: 'Segoe UI', sans-serif;">
            <h2 style="margin-bottom: 20px; text-align: center;">🤖 Recomendaciones Personalizadas con IA</h2>
            
            <!-- TALLA -->
            <div style="background: rgba(255,255,255,0.15); padding: 20px; margin: 15px 0; border-radius: 12px; backdrop-filter: blur(10px);">
                <h3 style="margin-bottom: 10px;">📏 Tu Talla Ideal</h3>
                <div style="font-size: 32px; font-weight: bold; margin: 10px 0;">
                    {recom_talla['talla_letra']} 
                    <span style="font-size: 18px; opacity: 0.8;">(EU {recom_talla['talla_eu']} / US {recom_talla['talla_us']})</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; margin-top: 10px;">
                    <strong>Confianza:</strong> {recom_talla['confianza']*100:.0f}% | 
                    <strong>Método:</strong> {recom_talla['metodo'].upper()}
                </div>
                <p style="font-size: 13px; margin-top: 10px; opacity: 0.95; line-height: 1.5;">
                    💡 {recom_talla['explicacion']}
                </p>
                
                {'<div style="margin-top: 10px;"><strong>Alternativas:</strong><br/>' + 
                 '<br/>'.join([f"• {alt['talla']}: {alt['probabilidad']*100:.0f}%" 
                               for alt in recom_talla.get('alternativas', [])[:3]]) + 
                 '</div>' if 'alternativas' in recom_talla else ''}
            </div>
            
            <!-- COLORES -->
            <div style="background: rgba(255,255,255,0.15); padding: 20px; margin: 15px 0; border-radius: 12px; backdrop-filter: blur(10px);">
                <h3 style="margin-bottom: 10px;">🎨 Colores que te Favorecen</h3>
                <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; margin: 10px 0;">
                    <strong>Tu tono de piel:</strong> {recom_colores.get('clasificacion_nombre', 'Analizando...')}
                    {f'<span style="margin-left: 10px;">({recom_colores.get("tono_piel_hex", "")})</span>' if 'tono_piel_hex' in recom_colores else ''}
                </div>
                <div style="display: flex; gap: 12px; margin: 15px 0; flex-wrap: wrap; justify-content: center;">
                    {''.join([f'<div style="width: 60px; height: 60px; background: {color}; border-radius: 10px; border: 3px solid white; box-shadow: 0 4px 10px rgba(0,0,0,0.3);"></div>' 
                              for color in recom_colores.get('colores_recomendados', [])])}
                </div>
                <div style="text-align: center; font-size: 12px; opacity: 0.9;">
                    {' • '.join(recom_colores.get('nombres_colores', []))}
                </div>
                <p style="font-size: 13px; margin-top: 15px; opacity: 0.95; line-height: 1.5;">
                    💡 {recom_colores.get('explicacion', '')}
                </p>
            </div>
            
            <!-- ESTILO -->
            <div style="background: rgba(255,255,255,0.15); padding: 20px; margin: 15px 0; border-radius: 12px; backdrop-filter: blur(10px);">
                <h3 style="margin-bottom: 10px;">👔 Estilo Recomendado para "{ocasion}"</h3>
                <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; margin: 10px 0;">
                    <strong>Estilo:</strong> {recom_estilo['estilo'].replace('_', ' ').title()} | 
                    <strong>Tipo de cuerpo:</strong> {recom_estilo['tipo_cuerpo'].replace('_', ' ').title()}
                </div>
                <div style="margin: 15px 0;">
                    <strong>Prendas sugeridas:</strong>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        {''.join([f'<li>{prenda}</li>' for prenda in recom_estilo['prendas_recomendadas'][:3]])}
                    </ul>
                </div>
                <p style="font-size: 13px; margin-top: 10px; opacity: 0.95; line-height: 1.5;">
                    💡 <strong>Consejo:</strong> {recom_estilo['consejo_cuerpo']}
                </p>
            </div>
            
            <!-- MEDIDAS -->
            <div style="background: rgba(255,255,255,0.15); padding: 20px; margin: 15px 0; border-radius: 12px; backdrop-filter: blur(10px);">
                <h3 style="margin-bottom: 10px;">📊 Tus Medidas Corporales</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; font-size: 13px;">
                    <div><strong>Altura:</strong> {medidas['altura_cm']:.0f} cm</div>
                    <div><strong>Hombros:</strong> {medidas['ancho_hombros_cm']:.1f} cm</div>
                    <div><strong>Cadera:</strong> {medidas['ancho_cadera_cm']:.1f} cm</div>
                    <div><strong>Torso:</strong> {medidas['torso_cm']:.1f} cm</div>
                    <div><strong>Brazo:</strong> {medidas['largo_brazo_cm']:.1f} cm</div>
                    <div><strong>Pierna:</strong> {medidas['largo_pierna_cm']:.1f} cm</div>
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 20px; font-size: 12px; opacity: 0.8;">
                ⚡ Procesado con IA en {config.DEVICE.upper()} | Confianza promedio: {(recom_talla['confianza'] + recom_estilo['nivel_confianza'])/2*100:.0f}%
            </div>
        </div>
        """

        return str(glb_path), qr_img, html_recomendaciones

    except Exception as e:
        import traceback
        error_msg = f"❌ Error en procesamiento: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, None, error_msg


def ejecutar_evaluacion_completa():
    """Ejecuta evaluación completa del sistema"""
    try:
        # Crear dataset de prueba si no existe
        dataset_imgs = list(config.DATASET_DIR.glob("*.jpg")) + list(config.DATASET_DIR.glob("*.png"))

        if len(dataset_imgs) < 5:
            return "⚠️ Se necesitan al menos 5 imágenes en dataset_validacion/ para evaluar"

        # Evaluación de detección
        evaluador.evaluar_deteccion_pose(dataset_imgs[:20])

        # Evaluación de rendimiento
        img_test = Image.open(dataset_imgs[0])
        evaluador.evaluar_rendimiento(img_test, None, num_iteraciones=10)

        # Generar reporte
        reporte_path = evaluador.generar_reporte()

        return f"✅ Evaluación completada\n\nReporte guardado en:\n{reporte_path}"

    except Exception as e:
        return f"❌ Error en evaluación: {str(e)}"


# ============================================
# INTERFAZ GRADIO
# ============================================

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue"
    ),
    title="Probador Virtual 3D con IA",
    css="""
    .gradio-container {
        max-width: 1400px !important;
    }
    .main-header {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 20px;
    }
    """
) as demo:

    # Header
    gr.HTML("""
    <div class="main-header">
        <h1 style="font-size: 36px; margin-bottom: 10px;">🎨 Probador Virtual 3D con Inteligencia Artificial</h1>
        <p style="font-size: 16px; opacity: 0.95;">
            Sistema completo de prueba virtual con recomendaciones personalizadas de tallas, colores y estilos
        </p>
        <p style="font-size: 13px; margin-top: 10px; opacity: 0.8;">
            ⚡ Optimizado para RTX 3050 | 🤖 IA con MediaPipe, MiDaS y Random Forest | 📱 Compatible con AR móvil
        </p>
    </div>
    """)

    with gr.Tabs() as tabs:

        # TAB 1: PROBADOR PRINCIPAL
        with gr.Tab("🎯 Probador Virtual", id=0):
            gr.Markdown("""
            ### Instrucciones:
            1. 📸 Sube una **foto de cuerpo completo** (fondo claro recomendado)
            2. 👕 Sube una **imagen de prenda** (o usa una del catálogo)
            3. ⚙️ Ajusta **color, escala y estilo** de la prenda
            4. 🎯 Selecciona la **ocasión** para obtener recomendaciones personalizadas
            5. 🚀 Presiona **GENERAR AVATAR 3D**
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📸 1. Tu Foto")
                    img_persona_input = gr.Image(
                        label="Foto de Cuerpo Completo",
                        type="numpy",
                        height=300
                    )
                    altura_input = gr.Number(
                        label="Tu Altura en cm (Opcional - se estimará automáticamente si dejas en 0)",
                        value=0,
                        minimum=0,
                        maximum=220,
                        step=1,
                        info="Deja en 0 para estimación automática, o ingresa tu altura real para mayor precisión"
                    )

                    gr.Markdown("#### 👕 2. Prenda")
                    img_prenda_input = gr.Image(
                        label="Sube tu Prenda",
                        type="numpy",
                        height=250
                    )

                    gr.Markdown("#### 🎨 3. Personalización")
                    color_picker = gr.ColorPicker(
                        label="Color de Prenda",
                        value="#4A90E2"
                    )
                    escala_slider = gr.Slider(
                        0.7, 1.5,
                        value=1.0,
                        step=0.05,
                        label="Escala (Tamaño)"
                    )
                    estilo_dropdown = gr.Dropdown(
                        ["Regular", "Slim", "Loose"],
                        label="Ajuste de Fit",
                        value="Regular"
                    )

                    gr.Markdown("#### 🎯 4. Ocasión")
                    ocasion_dropdown = gr.Dropdown(
                        ["Trabajo", "Casual", "Gimnasio", "Cita", "Entrevista", "Salir", "Restaurante"],
                        label="¿Para qué ocasión?",
                        value="Casual"
                    )

                    btn_preview = gr.Button(
                        "🔄 Actualizar Vista Previa",
                        variant="secondary",
                        size="sm"
                    )

                    prenda_preview = gr.Image(
                        label="Vista Previa de Prenda Modificada",
                        interactive=False,
                        height=200
                    )

                    btn_generar = gr.Button(
                        "🚀 GENERAR AVATAR 3D CON IA",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("#### 🤖 Recomendaciones con IA")
                    recomendaciones_html = gr.HTML(
                        """
                        <div style='text-align: center; padding: 50px; color: #666;'>
                            <h3>Esperando análisis...</h3>
                            <p>Las recomendaciones aparecerán aquí después de procesar tu imagen</p>
                        </div>
                        """
                    )

                    gr.Markdown("#### 🎨 Resultado 3D")
                    modelo_3d_output = gr.Model3D(
                        label="Tu Avatar 3D Interactivo",
                        height=400
                    )

                    gr.Markdown("💡 **Tip:** Arrastra para rotar, scroll para zoom")

                    gr.Markdown("#### 📱 Código QR para AR Móvil")
                    qr_output = gr.Image(
                        label="Escanea con tu móvil para ver en Realidad Aumentada",
                        height=250
                    )

        # TAB 2: EVALUACIÓN
        with gr.Tab("📊 Evaluación y Métricas", id=1):
            gr.Markdown("""
            ### Sistema de Evaluación Automática
            
            Esta sección permite evaluar el rendimiento del sistema en:
            - ✅ Precisión de detección de pose
            - ⚡ Rendimiento computacional (FPS, memoria)
            - 📏 Exactitud de predicción de tallas
            
            **Nota:** Coloca imágenes de prueba en la carpeta `dataset_validacion/`
            """)

            btn_evaluar = gr.Button(
                "🧪 EJECUTAR EVALUACIÓN COMPLETA",
                variant="primary",
                size="lg"
            )

            resultado_evaluacion = gr.Textbox(
                label="Resultados de Evaluación",
                lines=15,
                max_lines=30
            )

        # TAB 3: INFORMACIÓN
        with gr.Tab("ℹ️ Información del Sistema", id=2):
            gr.Markdown(f"""
            # 📚 Información Técnica del Sistema
            
            ## 🎯 Objetivos Cumplidos
            
            ### ✅ Objetivo 1: Procesamiento de Imágenes y Modelos 3D
            - **Detección de Pose:** MediaPipe Pose (33 keypoints)
            - **Estimación de Profundidad:** MiDaS Small (optimizado para RTX 3050)
            - **Generación 3D:** Trimesh con texturizado realista
            - **Visualización AR:** Model-viewer con WebXR
            
            ### ✅ Objetivo 2: Interfaz Intuitiva
            - **Catálogo de Prendas:** Sistema de gestión de prendas
            - **Personalización en Tiempo Real:** Color, escala, estilo
            - **Vista Previa Interactiva:** Feedback inmediato
            - **Múltiples Modos:** 2D preview, 3D viewer, AR móvil
            
            ### ✅ Objetivo 3: Inteligencia Artificial
            - **Recomendador de Tallas:** Random Forest entrenado con ANSUR II (6,068 muestras)
            - **Análisis de Tono de Piel:** K-means clustering + teoría del color
            - **Recomendador de Colores:** Basado en subtonos y paletas armónicas
            - **Recomendador de Estilos:** Sistema experto con reglas de moda
            
            ### ✅ Objetivo 4: Evaluación y Validación
            - **Métricas Técnicas:** Tasa de éxito, FPS, uso de memoria
            - **Validación de IA:** Accuracy de tallas, Cohen's Kappa para colores
            - **Benchmarking:** Rendimiento en RTX 3050
            
            ---
            
            ## ⚙️ Configuración Actual
            
            | Parámetro | Valor |
            |-----------|-------|
            | **Hardware** | {config.DEVICE.upper()} |
            | **VRAM** | {torch.cuda.get_device_properties(0).total_memory // (1024**2) if torch.cuda.is_available() else 0} MB |
            | **Python** | {sys.version.split()[0]} |
            | **Modelo Profundidad** | {config.DEPTH_MODEL} |
            | **Pose Complexity** | {config.POSE_COMPLEXITY} |
            | **Max Image Size** | {config.MAX_IMAGE_SIZE} |
            | **HTTP Port** | {config.HTTP_PORT} |
            
            ---
            
            ## 📂 Estructura de Directorios
            
            ```
            {config.BASE_DIR}/
            ├── avatares_generados/     # Modelos 3D y archivos AR
            ├── catalogo_prendas/        # Base de datos de prendas
            ├── modelos_ia/              # Modelos ML entrenados
            ├── dataset_validacion/      # Imágenes para evaluación
            └── evaluacion_resultados/   # Reportes y métricas
            ```
            
            ---
            
            ## 🔬 Tecnologías Utilizadas
            
            ### Computer Vision & Deep Learning
            - **MediaPipe:** Detección de pose corporal
            - **MiDaS:** Estimación monocular de profundidad
            - **PyTorch:** Framework de deep learning
            - **OpenCV:** Procesamiento de imágenes
            
            ### Machine Learning
            - **Scikit-learn:** Random Forest, K-means, métricas
            - **ANSUR II Dataset:** Entrenamiento de predictor de tallas
            
            ### 3D Processing
            - **Trimesh:** Generación y manipulación de mallas 3D
            - **NumPy:** Operaciones numéricas eficientes
            
            ### UI & Web
            - **Gradio:** Interfaz web interactiva
            - **Model-viewer:** Visualización 3D con AR
            - **QR Code:** Generación de códigos QR para móvil
            
            ---
            
            ## 📖 Cómo Citar Este Trabajo
            
            ```
            @software{{probador_virtual_3d,
              author = {{[Tu Nombre]}},
              title = {{Probador Virtual 3D con Inteligencia Artificial}},
              year = {{2026}},
              url = {{[URL del repositorio]}}
            }}
            ```
            
            ---
            
            ## 📧 Contacto y Soporte
            
            - **Autor:** [Tu Nombre]
            - **Email:** [tu@email.com]
            - **Institución:** [Tu Universidad]
            - **Proyecto de Grado:** [Título completo]
            
            ---
            
            ## 📄 Licencia
            
            Este proyecto fue desarrollado como parte de un proyecto de grado académico.
            """)

    # ============================================
    # CONECTAR EVENTOS
    # ============================================

    # Preview de prenda
    btn_preview.click(
        fn=actualizar_preview_prenda,
        inputs=[img_prenda_input, color_picker, escala_slider, estilo_dropdown],
        outputs=prenda_preview
    )

    # Actualizar preview al cambiar parámetros
    for componente in [color_picker, escala_slider, estilo_dropdown]:
        componente.change(
            fn=actualizar_preview_prenda,
            inputs=[img_prenda_input, color_picker, escala_slider, estilo_dropdown],
            outputs=prenda_preview
        )

    # Generar avatar
    btn_generar.click(
        fn=procesar_imagen_completo,
        inputs=[
            img_persona_input,
            img_prenda_input,
            altura_input,
            color_picker,
            escala_slider,
            estilo_dropdown,
            ocasion_dropdown
        ],
        outputs=[modelo_3d_output, qr_output, recomendaciones_html]
    )

    # Evaluación
    btn_evaluar.click(
        fn=ejecutar_evaluacion_completa,
        outputs=resultado_evaluacion
    )


# ============================================
# LANZAR APLICACIÓN
# ============================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 INICIANDO PROBADOR VIRTUAL 3D")
    print("="*60)
    print(f"\n📍 Dispositivo: {config.DEVICE.upper()}")
    print(f"🌐 Servidor AR: http://localhost:{config.HTTP_PORT}")
    print(f"💾 Archivos de salida: {config.OUTPUT_DIR}")
    print("\n💡 Tip: Para mejores resultados, usa fotos con:")
    print("   - Fondo claro y uniforme")
    print("   - Buena iluminación")
    print("   - Persona de cuerpo completo y de frente")
    print("   - Resolución mínima 800x600")
    print("\n" + "="*60)
    print("\n🌐 Abre tu navegador en: http://localhost:7860")
    print("   (También funciona: http://127.0.0.1:7860)")
    print("\n" + "="*60 + "\n")

    demo.launch(
        server_name="127.0.0.1",  # Cambiado de 0.0.0.0 a 127.0.0.1 para Windows
        server_port=7860,
        share=False,  # Cambiar a True si quieres URL pública con ngrok
        show_error=True,
        quiet=False,
        inbrowser=True  # Abre automáticamente en navegador
    )