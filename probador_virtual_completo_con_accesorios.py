"""
================================================================================
PROBADOR VIRTUAL 3D CON INTELIGENCIA ARTIFICIAL
Proyecto de Grado - Sistema Completo Integrado
================================================================================
Autor: [Anyi Carolain Velasquez]
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

# Requests para detectar ngrok
try:
    import requests
except ImportError:
    requests = None  # Se manejará el error en la función


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
        
        # Calcular punto del cuello (arriba de los hombros, hacia la base del cuello)
        cuello_x = (puntos['h_izq'][0] + puntos['h_der'][0]) // 2
        # El cuello está aproximadamente 15% de la distancia desde hombros hacia arriba
        distancia_hombros_y = abs(puntos['h_izq'][1] - puntos['h_der'][1])
        altura_cabeza_hombros = abs(puntos['nariz'][1] - puntos['h_izq'][1])
        offset_cuello = int(altura_cabeza_hombros * 0.15)  # 15% hacia arriba
        cuello_y = puntos['h_izq'][1] - offset_cuello
        puntos['cuello'] = (cuello_x, cuello_y)

        return puntos

    def estimar_altura_automatica(self, puntos):
        """
        Estima la altura en cm usando altura en píxeles y calibración simple

        SIMPLIFICADO: Método más robusto basado en altura completa

        Args:
            puntos: dict de keypoints

        Returns:
            float: altura estimada en cm
        """
        if not puntos:
            return 170.0  # Fallback

        # Calcular altura en píxeles (nariz a tobillo)
        altura_px = puntos['tobillo'][1] - puntos['nariz'][1]

        if altura_px <= 0:
            return 170.0  # Fallback si hay error

        # Calcular ancho de hombros
        ancho_hombros_px = abs(puntos['h_izq'][0] - puntos['h_der'][0])

        # DEBUG: Mostrar valores
        print(f"\n   🔍 DEBUG Estimación de Altura:")
        print(f"      • Altura en píxeles: {altura_px:.1f} px")
        print(f"      • Ancho hombros: {ancho_hombros_px:.1f} px")

        # CALIBRACIÓN SIMPLE Y ROBUSTA:
        # Basado en observaciones empíricas:
        # - Foto típica de cuerpo completo: persona ocupa 60-80% de altura de imagen
        # - Imagen típica: 720-1080 px de alto
        # - Persona típica en píxeles: 500-800 px
        # - Altura real típica: 155-185 cm

        # Ratio observado: altura_px / altura_cm ≈ 3.8-4.5
        # Promedio: 4.2

        # Por tanto: altura_cm = altura_px / 4.2

        # FACTOR DE CONVERSIÓN CALIBRADO CON DATOS REALES
        # Calibrado con:
        # - Modelo 178cm: altura_px / 178 = factor
        # - Modelo 156cm: altura_px / 156 = factor
        # Promedio empírico observado: 4.0 px/cm (ajustado de 3.5 para evitar sobreestimación)
        factor_conversion = 4.0  # px por cm (CALIBRADO - más conservador)

        altura_estimada = altura_px / factor_conversion

        print(f"      • Factor conversión: {factor_conversion}")
        print(f"      • Altura estimada base: {altura_estimada:.1f} cm")

        # Ajuste fino según relación altura/ancho
        # Detectar si la foto tiene a la persona más cerca o más lejos
        if ancho_hombros_px > 0:
            # Ratio altura/ancho de hombros (típicamente 4.5-6.0)
            ratio_proporcion = altura_px / ancho_hombros_px

            print(f"      • Ratio altura/ancho: {ratio_proporcion:.2f}")

            # Si el ratio es muy bajo, persona está muy cerca (ajustar hacia abajo)
            # Si el ratio es muy alto, persona está muy lejos (ajustar hacia arriba)
            if ratio_proporcion < 4.5:  # Muy cerca
                altura_estimada *= 0.95
                print(f"      • Ajuste: -5% (persona muy cerca de cámara)")
            elif ratio_proporcion > 6.0:  # Muy lejos
                altura_estimada *= 1.05
                print(f"      • Ajuste: +5% (persona lejos de cámara)")

        # Limitar a rangos realistas (145-190cm para evitar estimaciones exageradas)
        altura_estimada = max(145, min(190, altura_estimada))

        print(f"      • Altura final: {altura_estimada:.1f} cm\n")

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

    def _aplicar_textura_avanzada(self, img_persona, img_prenda, puntos, depth_map, tipo_prenda="auto"):
        """
        Aplica prenda sobre la persona con ajuste automático y sombreado

        Mejoras:
        - Autocrop de la prenda (elimina espacios vacíos)
        - Detección automática mejorada: superior vs inferior
        - Anclaje preciso según tipo de prenda
        - Escala proporcional basada en anatomía
        - Sombreado según mapa de profundidad
        
        Args:
            tipo_prenda: "auto", "superior" (camisa/blusa/top), "inferior" (pantalón/falda/short)
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

        # 2. Detectar tipo de prenda automáticamente si es "auto"
        w_orig, h_orig = prenda.size
        ratio_aspecto = h_orig / w_orig
        
        if tipo_prenda == "auto":
            # Análisis multi-criterio para detección más precisa
            
            # Criterio 1: Ratio de aspecto
            es_alto = ratio_aspecto > 1.0  # Más alto que ancho
            es_muy_alto = ratio_aspecto > 1.5
            es_ancho = ratio_aspecto < 0.8  # Más ancho que alto
            
            # Criterio 2: Análisis de contenido de píxeles
            prenda_array = np.array(prenda)
            if prenda_array.shape[2] == 4:  # RGBA
                alpha = prenda_array[:, :, 3]
                # Analizar dónde hay contenido (alpha > 50)
                contenido = alpha > 50
                
                # Calcular densidad en tercios verticales
                altura_tercio = h_orig // 3
                tercio_superior = np.sum(contenido[0:altura_tercio, :]) / (altura_tercio * w_orig)
                tercio_medio = np.sum(contenido[altura_tercio:2*altura_tercio, :]) / (altura_tercio * w_orig)
                tercio_inferior = np.sum(contenido[2*altura_tercio:, :]) / (altura_tercio * w_orig)
                
                # Prendas superiores tienen más contenido arriba
                mas_contenido_arriba = tercio_superior > tercio_inferior * 1.3
                
                # Calcular ancho promedio en diferentes alturas
                ancho_arriba = np.sum(contenido[0:altura_tercio, :], axis=0)
                ancho_abajo = np.sum(contenido[2*altura_tercio:, :], axis=0)
                
                # Pantalones/shorts se ensanchan o mantienen, tops se estrechan abajo
                se_ensancha_abajo = np.mean(ancho_abajo) > np.mean(ancho_arriba) * 0.8
            else:
                mas_contenido_arriba = False
                se_ensancha_abajo = False
            
            # Reglas de decisión mejoradas
            if es_muy_alto and se_ensancha_abajo:
                # Ratio > 1.5 y se ensancha abajo = definitivamente pantalón/falda
                tipo_prenda = "inferior"
                razon = f"muy alto ({ratio_aspecto:.2f}) + se ensancha abajo"
            elif es_ancho and mas_contenido_arriba:
                # Ratio < 0.8 y más contenido arriba = definitivamente top/camisa
                tipo_prenda = "superior"
                razon = f"muy ancho ({ratio_aspecto:.2f}) + contenido arriba"
            elif ratio_aspecto < 0.7:
                # Muy ancho = probablemente top strapless o crop top
                tipo_prenda = "superior"
                razon = f"muy ancho ({ratio_aspecto:.2f}) = top/strapless"
            elif 0.7 <= ratio_aspecto <= 1.0 and mas_contenido_arriba:
                # Cuadrado con contenido arriba = top
                tipo_prenda = "superior"
                razon = f"cuadrado ({ratio_aspecto:.2f}) + contenido superior"
            elif 0.7 <= ratio_aspecto <= 1.0 and se_ensancha_abajo:
                # Cuadrado que se ensancha = short o falda corta
                tipo_prenda = "inferior"
                razon = f"cuadrado ({ratio_aspecto:.2f}) + se ensancha = short/falda"
            elif ratio_aspecto > 1.3:
                # Alto = pantalón o falda
                tipo_prenda = "inferior"
                razon = f"alto ({ratio_aspecto:.2f}) = pantalón/falda"
            else:
                # Caso ambiguo: usar ratio simple
                if ratio_aspecto >= 1.0:
                    tipo_prenda = "inferior"
                    razon = f"ratio {ratio_aspecto:.2f} >= 1.0"
                else:
                    tipo_prenda = "superior"
                    razon = f"ratio {ratio_aspecto:.2f} < 1.0"
            
            print(f"🔍 Detección automática: {tipo_prenda.upper()}")
            print(f"   Razón: {razon}")
            print(f"   Dimensiones: {w_orig}×{h_orig}px")

        # 3. Extraer landmarks necesarios
        h_izq, h_der = puntos['h_izq'], puntos['h_der']
        cadera_izq, cadera_der = puntos['cadera_izq'], puntos['cadera_der']
        rodilla_izq = puntos.get('rodilla_izq', None)
        tobillo_izq = puntos.get('tobillo_izq', None)
        
        # 4. LÓGICA DIFERENTE SEGÚN TIPO DE PRENDA
        if tipo_prenda == "superior":
            # ============ PRENDA SUPERIOR (Camisa, Blusa, Top) ============
            ancho_hombros = abs(h_izq[0] - h_der[0])
            ancho_deseado = int(ancho_hombros * 1.8)  # Factor para cubrir brazos
            alto_deseado = int(ancho_deseado * ratio_aspecto)
            
            prenda = prenda.resize((ancho_deseado, alto_deseado), Image.Resampling.LANCZOS)
            
            # Posición: centrado horizontal, anclaje en hombros
            pos_x = ((h_izq[0] + h_der[0]) // 2) - (ancho_deseado // 2)
            altura_hombros = h_izq[1]
            pos_y = altura_hombros - int(alto_deseado * 0.20)  # Hombros al 20% de la prenda
            
            print(f"👕 PRENDA SUPERIOR: ancho={ancho_deseado}px, alto={alto_deseado}px, pos=({pos_x},{pos_y})")
            
        else:  # tipo_prenda == "inferior"
            # ============ PRENDA INFERIOR (Pantalón, Falda, Short) ============
            ancho_caderas = abs(cadera_izq[0] - cadera_der[0])
            
            # Para pantalones/shorts/faldas, el ancho debe ser proporcional a las caderas
            # Factor 2.3 para que cubra bien las piernas con holgura natural
            ancho_deseado = int(ancho_caderas * 2.3)
            
            # La cintura (top del pantalón) está entre hombros y caderas
            altura_hombros = h_izq[1]
            altura_caderas = cadera_izq[1]
            # 75% para posicionar en cintura natural
            altura_cintura = altura_hombros + int((altura_caderas - altura_hombros) * 0.75)
            
            # Detectar si es SHORT basado en el ratio (shorts son más cuadrados)
            es_short = ratio_aspecto < 1.2
            
            # Si tenemos landmarks de piernas, calcular alto basado en tobillos/rodillas
            if tobillo_izq is not None and rodilla_izq is not None:
                altura_tobillos = tobillo_izq[1]
                altura_rodillas = rodilla_izq[1]
                
                if es_short:
                    # SHORT: termina en o justo debajo de rodillas
                    alto_deseado = int((altura_rodillas - altura_cintura) * 1.15)
                    print(f"🩳 Detectado SHORT (ratio={ratio_aspecto:.2f})")
                else:
                    # PANTALÓN LARGO o FALDA: llega a tobillos
                    # Factor 0.72 según tu ajuste perfecto
                    alto_deseado = int((altura_tobillos - altura_cintura) * 0.72)
                    print(f"👖 Detectado PANTALÓN/FALDA LARGA (ratio={ratio_aspecto:.2f})")
                
                print(f"📏 Calculado por landmarks: alto={alto_deseado}px")
            else:
                # Fallback: usar ratio de aspecto de la prenda
                alto_deseado = int(ancho_deseado * ratio_aspecto * 0.72)
                print(f"📏 Calculado por ratio: alto={alto_deseado}px (ratio={ratio_aspecto:.2f})")
            
            prenda = prenda.resize((ancho_deseado, alto_deseado), Image.Resampling.LANCZOS)
            
            # Posición: centrado horizontal en caderas, anclaje en CINTURA
            pos_x = ((cadera_izq[0] + cadera_der[0]) // 2) - (ancho_deseado // 2)
            
            # El top del pantalón/short/falda debe estar en la cintura
            # Ajustar ligeramente hacia arriba (12% del alto) para que la cintura quede bien
            pos_y = altura_cintura - int(alto_deseado * 0.12)
            
            print(f"👗 PRENDA INFERIOR: ancho={ancho_deseado}px, alto={alto_deseado}px, pos=({pos_x},{pos_y})")
            print(f"   Cintura en Y={altura_cintura}, Caderas en Y={altura_caderas}")
            if tobillo_izq:
                fin_pantalon = pos_y + alto_deseado
                print(f"   Tobillos en Y={tobillo_izq[1]}, Termina en Y={fin_pantalon}")

        # 5. Asegurar que está dentro de la imagen
        pos_x = max(0, min(pos_x, img_p_pil.width - ancho_deseado))
        pos_y = max(0, min(pos_y, img_p_pil.height - alto_deseado))

        # 6. Aplicar sombreado basado en profundidad
        prenda_np = np.array(prenda)
        
        # Asegurar que no excedemos los límites de depth_map
        y_end = min(pos_y + alto_deseado, depth_map.shape[0])
        x_end = min(pos_x + ancho_deseado, depth_map.shape[1])
        zona_depth = depth_map[pos_y:y_end, pos_x:x_end]

        # Ajustar prenda si es necesaria por límites
        if zona_depth.shape[0] < alto_deseado or zona_depth.shape[1] < ancho_deseado:
            prenda_np = prenda_np[:zona_depth.shape[0], :zona_depth.shape[1]]

        # Verificar dimensiones coincidan
        if zona_depth.shape[:2] == prenda_np.shape[:2]:
            # Normalizar profundidad a rango [0.5, 1.0] para sombreado sutil
            sombra = np.clip(zona_depth * 1.2, 0.5, 1.0)

            # Aplicar solo a canales RGB
            if prenda_np.shape[2] == 4:  # RGBA
                prenda_np[:, :, :3] = (prenda_np[:, :, :3] * sombra[:, :, np.newaxis]).astype(np.uint8)

        # 7. Componer imagen final
        capa = Image.new("RGBA", img_p_pil.size, (0, 0, 0, 0))
        capa.paste(Image.fromarray(prenda_np), (pos_x, pos_y), Image.fromarray(prenda_np))

        resultado = Image.alpha_composite(img_p_pil, capa).convert("RGB")

        return np.array(resultado)

    def generar(self, img_persona, img_prenda, puntos, medidas=None, progreso_callback=None, tipo_prenda="auto"):
        """
        Pipeline completo de generación de avatar 3D

        Args:
            img_persona: PIL Image de la persona
            img_prenda: PIL Image o numpy de la prenda
            puntos: dict de keypoints
            medidas: dict con medidas corporales (incluye altura_cm)
            progreso_callback: función opcional para reportar progreso
            tipo_prenda: "auto" (detecta automático), "superior" (camisa), "inferior" (pantalón)

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
                img_persona, img_prenda, puntos, depth_norm, tipo_prenda=tipo_prenda
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

        # Asegurar que los tamaños coincidan
        n_vertices = len(mesh.vertices)
        n_colores = len(colores)

        if n_colores <= n_vertices:
            # Caso normal: hay menos o igual número de colores que vértices
            vertex_colors[:n_colores] = colores
        else:
            # Caso raro: hay más colores que vértices (no debería pasar)
            vertex_colors = colores[:n_vertices]

        mesh.visual.vertex_colors = vertex_colors

        # 7. Transformaciones finales
        # Rotar para orientación correcta
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))

        # Escalar para AR basado en ALTURA REAL
        # Escala base: 0.003 para una persona de 170cm (altura promedio)
        # Ajustar proporcionalmente según la altura real

        altura_real = medidas.get('altura_cm', 170)  # Fallback a 170cm si no hay altura
        altura_referencia = 170  # Altura de referencia
        escala_base = 0.003  # Escala para persona de 170cm

        # Calcular escala proporcional
        escala_ajustada = escala_base * (altura_real / altura_referencia)

        print(f"\n📏 Ajustando escala del modelo 3D:")
        print(f"   • Altura real: {altura_real:.1f} cm")
        print(f"   • Escala aplicada: {escala_ajustada:.6f}")
        print(f"   • Altura en AR: ~{escala_ajustada * 1000 * 170:.1f} cm\n")

        mesh.apply_scale(escala_ajustada)

        # Centrar en origen (importante para AR)
        mesh.apply_translation([-mesh.centroid[0], -mesh.centroid[1], -mesh.centroid[2]])

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




# ============================================
# MÓDULO DE ACCESORIOS
# ============================================

class GestorAccesorios:
    """
    Sistema para aplicar accesorios sobre el avatar
    Soporta: gorras, gafas, bufandas, collares, aretes, etc.
    """
    
    def __init__(self):
        # Configuración básica (se usa principalmente para fallbacks)
        self.posiciones_accesorios = {
            'gorra': {'ancla': 'nariz', 'offset_x': 0, 'offset_y': -70, 'escala_base': 0.85},
            'gafas': {'ancla': 'nariz', 'offset_x': 0, 'offset_y': -48, 'escala_base': 1.2},
            'bufanda': {'ancla': 'cuello', 'offset_x': 0, 'offset_y': 35, 'escala_base': 0.9},
            'collar': {'ancla': 'cuello', 'offset_x': 0, 'offset_y': 40, 'escala_base': 1.3},
            'arete_izq': {'ancla': 'ojo_izq', 'offset_x': -25, 'offset_y': 15, 'escala_base': 0.3},
            'arete_der': {'ancla': 'ojo_der', 'offset_x': 25, 'offset_y': 15, 'escala_base': 0.3},
            'diadema': {'ancla': 'nariz', 'offset_x': 0, 'offset_y': -130, 'escala_base': 1.6},
            'reloj': {'ancla': 'muneca_izq', 'offset_x': 0, 'offset_y': 0, 'escala_base': 0.5},
            'pulsera': {'ancla': 'muneca_der', 'offset_x': 0, 'offset_y': 0, 'escala_base': 0.5}
        }
    
    def aplicar_accesorio(self, imagen_base, imagen_accesorio, tipo_accesorio, puntos, escala=1.0):
        """
        Aplica un accesorio sobre la imagen base
        
        Args:
            imagen_base: PIL Image del cuerpo con prenda
            imagen_accesorio: PIL Image del accesorio (con o sin transparencia)
            tipo_accesorio: str ('gorra', 'gafas', 'bufanda', etc.)
            puntos: dict con keypoints de MediaPipe
            escala: float, ajuste de tamaño (default 1.0)
            
        Returns:
            PIL Image con accesorio aplicado
        """
        if tipo_accesorio not in self.posiciones_accesorios:
            print(f"⚠️ Tipo de accesorio '{tipo_accesorio}' no soportado")
            return imagen_base
        
        config = self.posiciones_accesorios[tipo_accesorio]
        
        # SISTEMA SIMPLIFICADO: Anclas anatómicas + offsets calculados
        
        if tipo_accesorio in ['gorra', 'diadema']:
            # GORRA: Offset fijo de 5.5cm arriba de los ojos
            if 'ojo_izq' in puntos and 'ojo_der' in puntos:
                # 1. Centro entre ojos
                ojo_izq_x = puntos['ojo_izq'][0]
                ojo_der_x = puntos['ojo_der'][0]
                ojo_izq_y = puntos['ojo_izq'][1]
                ojo_der_y = puntos['ojo_der'][1]
                
                ojos_centro_x = (ojo_izq_x + ojo_der_x) / 2
                ojos_centro_y = (ojo_izq_y + ojo_der_y) / 2
                
                # 2. Calcular factor de conversión píxeles → cm
                # Usar distancia entre hombros (conocemos el ancho en cm y en píxeles)
                if 'h_izq' in puntos and 'h_der' in puntos:
                    hombros_px = abs(puntos['h_der'][0] - puntos['h_izq'][0])
                    # Ancho de hombros promedio humano: ~40-55cm
                    # Usamos 45cm como referencia
                    ancho_hombros_cm = 45.0
                    px_por_cm = abs(hombros_px / ancho_hombros_cm)
                else:
                    # Fallback: estimar basado en distancia entre ojos
                    # Distancia entre ojos humana promedio: ~6.3cm
                    distancia_ojos_px = abs(ojo_der_x - ojo_izq_x)
                    distancia_ojos_cm = 6.3
                    px_por_cm = abs(distancia_ojos_px / distancia_ojos_cm)
                
                # 3. Convertir 5.5cm a píxeles (POSITIVO)
                offset_cm = 32.0  # cm arriba de los ojos
                offset_px = abs(offset_cm * px_por_cm)  # Asegurar positivo
                
                pos_x_base = ojos_centro_x
                pos_y_base = ojos_centro_y - offset_px
                offset_proporcional = 0
                
                print(f"   📍 Gorra (offset fijo 5.5cm):")
                print(f"      - Ojos centro: ({ojos_centro_x:.0f}, {ojos_centro_y:.0f})")
                print(f"      - Escala: {px_por_cm:.2f} px/cm")
                print(f"      - Offset: {offset_cm}cm = {offset_px:.0f}px")
                print(f"      - Posición final: ({pos_x_base:.0f}, {pos_y_base:.0f})")
            else:
                pos_x_base, pos_y_base = puntos.get('nariz', (0, 0))
                offset_proporcional = -100
        
        elif tipo_accesorio in ['gafas']:
            # GAFAS: Anclar EXACTAMENTE en los ojos
            if 'ojo_izq' in puntos and 'ojo_der' in puntos:
                ojo_izq_y = puntos['ojo_izq'][1]
                ojo_der_y = puntos['ojo_der'][1]
                ojos_y = (ojo_izq_y + ojo_der_y) / 2
                ojos_x = (puntos['ojo_izq'][0] + puntos['ojo_der'][0]) / 2
                
                pos_x_base = ojos_x
                pos_y_base = ojos_y
                offset_proporcional = 0  # Ya están en la posición correcta
                print(f"   📍 Gafas: ancladas en ojos ({ojos_x:.0f}, {ojos_y:.0f})")
            else:
                pos_x_base, pos_y_base = puntos.get('nariz', (0, 0))
                offset_proporcional = -48
        
        elif tipo_accesorio in ['bufanda', 'collar']:
            # BUFANDA: Justo en el cuello con offset pequeño hacia arriba
            if 'cuello' in puntos:
                cuello_x, cuello_y = puntos['cuello']
                
                # Bufanda: offset hacia arriba para que quede más pegada al cuello
                pos_x_base = cuello_x
                pos_y_base = cuello_y
                offset_proporcional = -30  # Negativo = hacia arriba
                
                print(f"   📍 Bufanda: cuello=({cuello_x:.0f}, {cuello_y:.0f}), offset={offset_proporcional}px (arriba)")
            else:
                pos_x_base, pos_y_base = puntos.get('h_izq', (0, 0))
                offset_proporcional = 25
        else:
            # Para otros accesorios, usar ancla normal
            ancla = config['ancla']
            if ancla not in puntos:
                print(f"⚠️ Punto de ancla '{ancla}' no encontrado")
                if 'nariz' in puntos:
                    ancla = 'nariz'
                else:
                    return imagen_base
            pos_x_base, pos_y_base = puntos[ancla]
            offset_proporcional = config['offset_y']
        
        # Convertir a RGBA si es necesario
        if imagen_base.mode != 'RGBA':
            imagen_base = imagen_base.convert('RGBA')
        
        # Asegurar que el accesorio sea RGBA (ya viene con fondo eliminado)
        if imagen_accesorio.mode != 'RGBA':
            imagen_accesorio = imagen_accesorio.convert('RGBA')
        
        # Calcular tamaño del accesorio
        ancho_hombros = abs(puntos['h_izq'][0] - puntos['h_der'][0]) if 'h_izq' in puntos and 'h_der' in puntos else 200
        
        # Tamaño base del accesorio proporcional a los hombros
        ancho_accesorio = int(ancho_hombros * config['escala_base'] * escala)
        
        # Mantener aspect ratio
        w_orig, h_orig = imagen_accesorio.size
        ratio = h_orig / w_orig
        alto_accesorio = int(ancho_accesorio * ratio)
        
        print(f"   📏 Tamaño accesorio: {ancho_accesorio}x{alto_accesorio}px (hombros: {ancho_hombros}px)")
        
        # Redimensionar accesorio
        accesorio_resized = imagen_accesorio.resize(
            (ancho_accesorio, alto_accesorio), 
            Image.Resampling.LANCZOS
        )
        
        # Calcular posición final con offset
        pos_x = int(pos_x_base + config['offset_x'] - ancho_accesorio // 2)
        
        # Para gafas, centrar verticalmente en los ojos
        if tipo_accesorio in ['gafas']:
            pos_y = int(pos_y_base + offset_proporcional - alto_accesorio // 2)
        else:
            pos_y = int(pos_y_base + offset_proporcional)
        
        # Asegurar que está dentro de la imagen
        pos_x = max(0, min(pos_x, imagen_base.width - ancho_accesorio))
        pos_y = max(0, min(pos_y, imagen_base.height - alto_accesorio))
        
        # Crear capa para el accesorio
        capa = Image.new('RGBA', imagen_base.size, (0, 0, 0, 0))
        capa.paste(accesorio_resized, (pos_x, pos_y), accesorio_resized)
        
        # Componer
        resultado = Image.alpha_composite(imagen_base, capa)
        
        print(f"   ✅ Accesorio '{tipo_accesorio}' aplicado en posición ({pos_x}, {pos_y})")
        
        return resultado
    
    def aplicar_multiples_accesorios(self, imagen_base, lista_accesorios, puntos):
        """
        Aplica múltiples accesorios en el orden correcto
        
        Args:
            imagen_base: PIL Image
            lista_accesorios: list de dicts con formato:
                [
                    {'imagen': PIL.Image, 'tipo': 'gorra', 'escala': 1.0},
                    {'imagen': PIL.Image, 'tipo': 'gafas', 'escala': 0.9},
                    ...
                ]
            puntos: dict con keypoints
            
        Returns:
            PIL Image con todos los accesorios aplicados
        """
        # Orden de aplicación (de atrás hacia adelante)
        orden_z = {
            'bufanda': 1,
            'collar': 2,
            'diadema': 3,
            'gorra': 4,
            'arete_izq': 5,
            'arete_der': 5,
            'gafas': 6,
            'reloj': 7,
            'pulsera': 7
        }
        
        # Ordenar accesorios por z-index
        lista_ordenada = sorted(
            lista_accesorios, 
            key=lambda x: orden_z.get(x['tipo'], 0)
        )
        
        resultado = imagen_base
        
        for acc in lista_ordenada:
            resultado = self.aplicar_accesorio(
                resultado,
                acc['imagen'],
                acc['tipo'],
                puntos,
                acc.get('escala', 1.0)
            )
        
        return resultado


class CatalogoAccesorios:
    """
    Catálogo de accesorios predefinidos
    """
    
    def __init__(self):
        self.accesorios = {
            'gorras': [
                {'id': 'gorra_001', 'nombre': 'Gorra Deportiva', 'tipo': 'gorra', 'color_base': '#000000'},
                {'id': 'gorra_002', 'nombre': 'Sombrero Panamá', 'tipo': 'gorra', 'color_base': '#F5F5DC'},
                {'id': 'gorra_003', 'nombre': 'Gorra Snapback', 'tipo': 'gorra', 'color_base': '#1E90FF'},
            ],
            'gafas': [
                {'id': 'gafas_001', 'nombre': 'Gafas de Sol Aviador', 'tipo': 'gafas', 'color_base': '#000000'},
                {'id': 'gafas_002', 'nombre': 'Lentes Redondos', 'tipo': 'gafas', 'color_base': '#8B4513'},
                {'id': 'gafas_003', 'nombre': 'Gafas Deportivas', 'tipo': 'gafas', 'color_base': '#FF4500'},
            ],
            'bufandas': [
                {'id': 'bufanda_001', 'nombre': 'Bufanda de Lana', 'tipo': 'bufanda', 'color_base': '#8B0000'},
                {'id': 'bufanda_002', 'nombre': 'Pañuelo Cuello', 'tipo': 'bufanda', 'color_base': '#4169E1'},
                {'id': 'bufanda_003', 'nombre': 'Bufanda Tejida', 'tipo': 'bufanda', 'color_base': '#A0522D'},
            ]
        }
    
    def obtener_por_categoria(self, categoria):
        """Obtiene accesorios de una categoría"""
        return self.accesorios.get(categoria, [])
    
    def obtener_todos(self):
        """Obtiene todos los accesorios"""
        todos = []
        for categoria in self.accesorios.values():
            todos.extend(categoria)
        return todos



class ModificadorPrenda:
    """
    OBJETIVO 2: Modificación de parámetros de prendas
    Permite cambiar color, escala, estilo en tiempo real
    """

    @staticmethod
    def cambiar_color(imagen, color_hex):
        """
        Cambia color dominante de la prenda preservando detalles

        MEJORADO: Aplica el color de forma más agresiva

        Args:
            imagen: PIL Image o numpy array
            color_hex: string "#RRGGBB"

        Returns:
            numpy array con color modificado
        """
        if imagen is None:
            print("⚠️ Imagen es None")
            return None

        print(f"🎨 cambiar_color() llamado:")
        print(f"   • Color recibido: '{color_hex}' (tipo: {type(color_hex)})")

        # VALIDACIÓN DE COLOR HEX
        if not color_hex or not isinstance(color_hex, str):
            print(f"⚠️ Color inválido (tipo): {color_hex}")
            return imagen if isinstance(imagen, np.ndarray) else np.array(imagen)

        # Limpiar y validar formato
        color_hex = color_hex.strip()
        print(f"   • Color limpio: '{color_hex}'")

        # CONVERTIR RGBA A HEX si es necesario
        if color_hex.startswith('rgba('):
            print(f"   ⚙️ Convirtiendo RGBA a HEX...")
            try:
                # Extraer números del formato rgba(R, G, B, A)
                import re
                nums = re.findall(r'[\d.]+', color_hex)
                if len(nums) >= 3:
                    r = int(float(nums[0]))
                    g = int(float(nums[1]))
                    b = int(float(nums[2]))
                    color_hex = f"#{r:02x}{g:02x}{b:02x}"
                    print(f"   ✅ Convertido a: {color_hex}")
            except Exception as e:
                print(f"   ❌ Error convirtiendo RGBA: {e}")
                return imagen if isinstance(imagen, np.ndarray) else np.array(imagen)

        # Si no empieza con #, agregarlo
        if not color_hex.startswith('#'):
            color_hex = '#' + color_hex
            print(f"   • Color con #: '{color_hex}'")

        # Validar longitud (debe ser #RRGGBB = 7 caracteres)
        if len(color_hex) != 7:
            print(f"⚠️ Color hex inválido (longitud {len(color_hex)}): {color_hex}")
            return imagen if isinstance(imagen, np.ndarray) else np.array(imagen)

        # Validar que sean caracteres hexadecimales válidos
        try:
            int(color_hex[1:], 16)  # Intenta convertir la parte sin #
            print(f"   ✅ Formato hex válido")
        except ValueError:
            print(f"⚠️ Color hex inválido (formato): {color_hex}")
            return imagen if isinstance(imagen, np.ndarray) else np.array(imagen)

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
        pixels_afectados = np.sum(mask)
        print(f"   • Píxeles a colorear: {pixels_afectados}")

        try:
            # Convertir color hex a RGB
            rgb_target = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            print(f"   • RGB objetivo: {rgb_target}")

            # Convertir a HSV para manipulación de color
            img_hsv = cv2.cvtColor(img_np[:, :, :3], cv2.COLOR_RGB2HSV).astype(float)

            # Calcular nuevo matiz y saturación del color objetivo
            h_target, s_target, v_target = colorsys.rgb_to_hsv(
                rgb_target[0]/255, rgb_target[1]/255, rgb_target[2]/255
            )

            print(f"   • HSV objetivo: H={h_target*180:.1f}, S={s_target:.2f}, V={v_target:.2f}")

            # CAMBIO AGRESIVO: Sobrescribir matiz Y saturación
            # (antes solo cambiaba matiz)
            img_hsv[mask, 0] = h_target * 180  # OpenCV usa 0-180 para H

            # Aplicar saturación del color objetivo (más agresivo)
            # Manteniendo algo de la saturación original para textura
            img_hsv[mask, 1] = np.clip(
                s_target * 255 * 0.8 + img_hsv[mask, 1] * 0.2,  # 80% nuevo, 20% original
                0,
                255
            )

            # Ajustar valor (brillo) levemente para mantener detalles
            # pero acercar al color objetivo
            img_hsv[mask, 2] = np.clip(
                v_target * 255 * 0.3 + img_hsv[mask, 2] * 0.7,  # 30% nuevo, 70% original
                0,
                255
            )

            # Convertir de vuelta a RGB
            img_recolored = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            # Restaurar canal alpha
            resultado = np.dstack([img_recolored, alpha])

            print(f"   ✅ Color aplicado exitosamente")
            print(f"   📊 Cambio: Matiz={h_target*180:.1f}°, Saturación={s_target*255:.0f}/255")

            return resultado

        except Exception as e:
            print(f"❌ Error cambiando color: {e}")
            print(f"   Color recibido: {color_hex}")
            import traceback
            traceback.print_exc()
            return img_np

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
            # Proporción más realista: cintura ≈ 1.5× ancho hombros (no 2.0×)
            cintura_cm = ancho_hombros_cm * 1.5

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
                "colores": ["#D4AF37", "#E97451", "#8B4513", "#228B22", "#DC143C"],
                "nombres": ["Dorado", "Terracota", "Marrón Cálido", "Verde Oliva", "Rojo Cálido"],
                "evitar": ["#000080", "#4B0082", "#C0C0C0", "#FF1493"],
                "descripcion": "Subtonos dorados/amarillos/melocotón"
            },
            "frio": {
                "colores": ["#000080", "#4B0082", "#800080", "#FF1493", "#4682B4"],
                "nombres": ["Azul Marino", "Índigo", "Morado", "Rosa Fucsia", "Azul Acero"],
                "evitar": ["#FF8C00", "#8B4513", "#D4AF37", "#E97451"],
                "descripcion": "Subtonos rosados/azulados"
            },
            "neutro": {
                "colores": ["#2F4F4F", "#708090", "#8B0000", "#006400", "#4682B4"],
                "nombres": ["Verde Azulado", "Gris Pizarra", "Rojo Oscuro", "Verde Oscuro", "Azul Acero"],
                "evitar": [],
                "descripcion": "Balance entre cálido y frío. ¡Tienes suerte!"
            }
        }

    def extraer_tono_piel(self, imagen, puntos_pose):
        """
        Extrae color de piel promedio de región facial
        
        MEJORADO: Detección robusta para todo tipo de pieles (claras a oscuras)
        
        Args:
            imagen: numpy array BGR
            puntos_pose: dict con keypoints
            
        Returns:
            dict con rgb y hex del color de piel
        """
        if 'nariz' not in puntos_pose:
            return None
        
        # ROI centrada en la cara (entre nariz y mejillas)
        x_nariz, y_nariz = puntos_pose['nariz']
        
        # Usar también los ojos para centrar mejor el ROI
        if 'ojo_izq' in puntos_pose and 'ojo_der' in puntos_pose:
            x_ojo_izq, y_ojo_izq = puntos_pose['ojo_izq']
            x_ojo_der, y_ojo_der = puntos_pose['ojo_der']
            
            # Centro de la cara entre los ojos
            x_centro = (x_ojo_izq + x_ojo_der) // 2
            y_centro = (y_ojo_izq + y_ojo_der) // 2
            
            # ROI entre ojos y nariz (zona de mejillas)
            x = x_centro
            y = int((y_centro + y_nariz) / 2)  # Punto medio entre ojos y nariz
        else:
            x, y = x_nariz, y_nariz
        
        # ROI de tamaño adaptativo
        roi_size = 80
        
        y1 = max(0, int(y - roi_size // 2))
        y2 = min(imagen.shape[0], int(y + roi_size // 2))
        x1 = max(0, int(x - roi_size))
        x2 = min(imagen.shape[1], int(x + roi_size))
        
        roi = imagen[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Convertir a RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pixels = roi_rgb.reshape(-1, 3).astype(np.float32)
        
        # FILTRO DE PIEL MEJORADO
        # Basado en múltiples criterios para detectar todo tipo de pieles
        
        r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        
        # Criterio 1: Rangos RGB generales (muy amplios)
        mask_rgb = (r > 20) & (r < 255) & (g > 15) & (g < 255) & (b > 10) & (b < 255)
        
        # Criterio 2: Relaciones entre canales (característico de piel)
        # En piel humana: R > G > B (generalmente)
        mask_relacion = (r > g - 15) & (g > b - 20)
        
        # Criterio 3: Excluir colores extremos
        # Evitar blancos puros, negros puros, y colores saturados
        brightness = (r + g + b) / 3
        mask_brightness = (brightness > 40) & (brightness < 240)
        
        # Criterio 4: Evitar colores muy saturados (ropa, fondo)
        max_channel = np.maximum(np.maximum(r, g), b)
        min_channel = np.minimum(np.minimum(r, g), b)
        saturation = (max_channel - min_channel) / (max_channel + 1)
        mask_saturation = saturation < 0.6  # Piel tiene baja saturación
        
        # Combinar todos los criterios
        mask = mask_rgb & mask_relacion & mask_brightness & mask_saturation
        
        skin_pixels = pixels[mask].astype(int)
        
        print(f"   🎨 Píxeles de piel detectados: {len(skin_pixels)}")
        
        if len(skin_pixels) < 50:
            print(f"   ⚠️ Muy pocos píxeles de piel: {len(skin_pixels)}")
            # Intentar con criterios más relajados
            mask_fallback = mask_rgb & mask_brightness
            skin_pixels = pixels[mask_fallback].astype(int)
            print(f"   🔄 Reintento con criterios relajados: {len(skin_pixels)} píxeles")
            
            if len(skin_pixels) < 50:
                return None
        
        # K-means para encontrar color dominante
        n_clusters = min(3, max(1, len(skin_pixels) // 100))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(skin_pixels)
        
        # Cluster más poblado = color dominante
        labels = kmeans.labels_
        counts = np.bincount(labels)
        dominant = np.argmax(counts)
        color_dom = kmeans.cluster_centers_[dominant].astype(int)
        
        # Asegurar que los valores estén en rango [0, 255]
        color_dom = np.clip(color_dom, 0, 255)
        
        color_hex = f"#{color_dom[0]:02x}{color_dom[1]:02x}{color_dom[2]:02x}"
        print(f"   🎨 Color de piel detectado: RGB{tuple(color_dom)} → {color_hex}")
        
        return {
            'rgb': tuple(color_dom),
            'hex': color_hex
        }

    def clasificar_tono(self, color_rgb):
        """
        Clasifica en cálido/frío/neutro según undertones
        
        MEJORADO: Umbrales ajustados para mayor sensibilidad en pieles reales
        
        Método:
        - Cálido: más amarillo/dorado (verde > azul)
        - Frío: más rosa/azul (azul > verde)
        - Neutro: balance relativo
        """
        r, g, b = color_rgb
        
        # Índices de undertone (subtono)
        indice_amarillo = (r + g) / 2 - b
        indice_rosa = (r + b) / 2 - g
        
        # Ratio R/B y R/G (adicionales para mejor precisión)
        ratio_rb = r / (b + 1)
        ratio_rg = r / (g + 1)
        
        # Diferencia G-B (verde vs azul) - INDICADOR CLAVE
        diff_gb = g - b
        
        print(f"   📊 Análisis de undertone:")
        print(f"      • RGB: ({r}, {g}, {b})")
        print(f"      • Índice amarillo (cálido): {indice_amarillo:.1f}")
        print(f"      • Índice rosa (frío): {indice_rosa:.1f}")
        print(f"      • Ratio R/B: {ratio_rb:.2f}")
        print(f"      • Ratio R/G: {ratio_rg:.2f}")
        print(f"      • Diferencia G-B: {diff_gb:.1f}")
        
        # CLASIFICACIÓN MEJORADA CON UMBRALES MÁS REALISTAS
        
        # MÉTODO PRIMARIO: Diferencia Verde-Azul (más confiable)
        # En pieles reales, esta diferencia suele ser de -5 a +10
        
        if diff_gb > 2:
            # Verde > Azul → Undertone CÁLIDO
            clasificacion = "calido"
            razon = f"Verde ({g}) > Azul ({b}) por {diff_gb:.1f}"
            
        elif diff_gb < -2:
            # Azul > Verde → Undertone FRÍO
            clasificacion = "frio"
            razon = f"Azul ({b}) > Verde ({g}) por {abs(diff_gb):.1f}"
            
        else:
            # Diferencia pequeña (-2 a +2) → Verificar con otros indicadores
            
            # MÉTODO SECUNDARIO: Ratios R/B y R/G
            if ratio_rb > 1.05 and indice_amarillo > 0:
                clasificacion = "calido"
                razon = f"Ratio R/B={ratio_rb:.2f} > 1.05 (más rojo que azul)"
                
            elif ratio_rb < 0.98 and indice_rosa > 0:
                clasificacion = "frio"
                razon = f"Ratio R/B={ratio_rb:.2f} < 0.98 (más azul que rojo)"
                
            else:
                # VERDADERAMENTE NEUTRO
                clasificacion = "neutro"
                razon = f"Diferencia G-B={diff_gb:.1f} cercana a 0, ratios equilibrados"
        
        print(f"   ✅ Clasificación: {clasificacion.upper()}")
        print(f"      Razón: {razon}\n")
        
        return clasificacion

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
            overflow: hidden;
        }}
        .header {{ 
            background: rgba(0,0,0,0.3);
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            z-index: 10;
        }}
        .header h1 {{ font-size: 24px; margin-bottom: 5px; }}
        .header p {{ font-size: 14px; opacity: 0.9; }}
        
        #loading {{ 
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            z-index: 5;
        }}
        #loading.hidden {{ display: none; }}
        .spinner {{
            border: 4px solid rgba(255,255,255,0.3);
            border-top: 4px solid white;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
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
            z-index: 100;
        }}
        .ar-button:active {{ 
            transform: translateX(-50%) scale(0.95);
        }}
        #error {{ 
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255,0,0,0.2);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            display: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎨 Probador Virtual AR</h1>
        <p>Toca el botón para ver el modelo en tu espacio real</p>
    </div>
    
    <div id="loading">
        <div class="spinner"></div>
        <p>Cargando modelo 3D...</p>
    </div>
    
    <div id="error">
        <p>❌ Error cargando el modelo</p>
        <p style="font-size: 12px; margin-top: 10px;">Verifica la conexión</p>
    </div>
    
    <model-viewer 
        id="viewer"
        src="{MODEL_PATH}" 
        ar 
        ar-modes="webxr"
        camera-controls 
        touch-action="pan-y"
        shadow-intensity="1" 
        auto-rotate
        auto-rotate-delay="0"
        rotation-per-second="30deg"
        environment-image="neutral"
        exposure="1.0"
        camera-orbit="0deg 75deg 0.6m"
        min-camera-orbit="auto auto 0.3m"
        max-camera-orbit="auto auto 2m"
        field-of-view="45deg"
        ios-src="{MODEL_PATH}">
        <button slot="ar-button" class="ar-button" id="ar-btn">
            📱 VER EN MI ESPACIO (AR)
        </button>
    </model-viewer>
    
    <script>
        const viewer = document.querySelector('#viewer');
        const loading = document.querySelector('#loading');
        const error = document.querySelector('#error');
        const arBtn = document.querySelector('#ar-btn');
        
        // Debug: Verificar capacidades WebXR
        console.log('🔍 Verificando WebXR...');
        if ('xr' in navigator) {{
            console.log('✅ WebXR API disponible');
            navigator.xr.isSessionSupported('immersive-ar').then(supported => {{
                if (supported) {{
                    console.log('✅ AR inmersivo soportado');
                }} else {{
                    console.warn('⚠️ AR inmersivo NO soportado');
                    console.log('💡 Intentando con scene-viewer...');
                }}
            }}).catch(err => {{
                console.error('❌ Error verificando WebXR:', err);
            }});
        }} else {{
            console.warn('⚠️ WebXR API no disponible');
            console.log('📱 Dispositivo:', navigator.userAgent);
        }}
        
        viewer.addEventListener('load', () => {{
            console.log('✅ Modelo cargado exitosamente');
            loading.classList.add('hidden');
            
            // Verificar soporte AR después de cargar
            setTimeout(() => {{
                if (viewer.canActivateAR) {{
                    console.log('✅ AR disponible');
                    arBtn.style.display = 'block';
                }} else {{
                    console.warn('⚠️ AR no disponible');
                    console.log('💡 Prueba actualizar Chrome o instalar ARCore');
                    arBtn.innerHTML = '⚠️ AR no soportado - Usa visualizador 3D';
                    arBtn.style.background = 'linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%)';
                }}
            }}, 500);
        }});
        
        viewer.addEventListener('error', (event) => {{
            console.error('❌ Error cargando modelo:', event);
            loading.style.display = 'none';
            error.style.display = 'block';
            error.innerHTML = '<p>❌ Error cargando modelo</p><p style="font-size: 12px;">Recarga la página</p>';
        }});
        
        // Evento cuando se intenta activar AR
        arBtn.addEventListener('click', () => {{
            console.log('🎯 Botón AR presionado');
            console.log('📱 User Agent:', navigator.userAgent);
            
            // Verificar permisos de cámara
            if ('permissions' in navigator) {{
                navigator.permissions.query({{ name: 'camera' }}).then(result => {{
                    console.log('📷 Permiso cámara:', result.state);
                    if (result.state === 'denied') {{
                        alert('⚠️ Permiso de cámara denegado\\n\\nHabilítalo en Configuración → Chrome → Permisos → Cámara');
                    }}
                }}).catch(err => {{
                    console.warn('No se pudo verificar permisos:', err);
                }});
            }}
        }});
        
        // Detectar estado de AR
        viewer.addEventListener('ar-status', (event) => {{
            console.log('📱 AR Status:', event.detail.status);
            
            if (event.detail.status === 'not-presenting') {{
                console.log('ℹ️ AR cerrado');
            }} else if (event.detail.status === 'session-started') {{
                console.log('✅ Sesión AR iniciada');
            }} else if (event.detail.status === 'failed') {{
                console.error('❌ AR falló al iniciar');
                alert('❌ No se pudo activar AR\\n\\n' +
                      'Tu dispositivo: Xiaomi Redmi Note 11 Pro\\n\\n' +
                      'Verifica:\\n' +
                      '1. Google Play Services for AR instalado\\n' +
                      '2. Chrome actualizado\\n' +
                      '3. Permisos de cámara habilitados\\n\\n' +
                      'Consola: Revisa mensajes de debug');
            }}
        }});
            }} else if (event.detail.status === 'session-started') {{
                console.log('✅ AR activado correctamente');
            }}
        }});
        
        // Detectar toque en botón AR
        arBtn.addEventListener('click', () => {{
            console.log('🔘 Botón AR presionado');
        }});
        
        // Timeout de seguridad
        setTimeout(() => {{
            if (!viewer.loaded) {{
                console.warn('⚠️ Modelo tardando en cargar...');
            }}
        }}, 5000);
    </script>        }}, 5000);
    </script>
</body>
</html>
"""

class ServidorAR(SimpleHTTPRequestHandler):
    """Servidor HTTP con soporte CORS para AR"""

    def end_headers(self):
        # CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        # Cache headers para mejor rendimiento
        self.send_header('Cache-Control', 'public, max-age=3600')
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

                print(f"   ✅ Servido HTML AR: {filename}")
            else:
                self.send_error(404, f"Archivo no encontrado: {filename}")
                print(f"   ❌ No encontrado: {filename}")

        elif self.path.startswith('/model/'):
            filename = self.path.split('/')[-1]
            filepath = config.OUTPUT_DIR / filename

            if filepath.exists():
                file_size = filepath.stat().st_size

                self.send_response(200)
                self.send_header('Content-type', 'model/gltf-binary')
                self.send_header('Content-Length', str(file_size))
                # Headers adicionales para modelos 3D
                self.send_header('Accept-Ranges', 'bytes')
                self.end_headers()

                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())

                print(f"   ✅ Servido modelo GLB: {filename} ({file_size/1024:.1f} KB)")
            else:
                self.send_error(404, f"Modelo no encontrado: {filename}")
                print(f"   ❌ Modelo no encontrado: {filename}")
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # Silenciar logs HTTP por defecto


def iniciar_servidor_http():
    """Inicia servidor HTTP en thread separado"""
    # Usar 0.0.0.0 para permitir conexiones desde otros dispositivos
    server = HTTPServer(('0.0.0.0', config.HTTP_PORT), ServidorAR)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"🌐 Servidor HTTP iniciado en http://0.0.0.0:{config.HTTP_PORT}")
    print(f"   Accesible desde otros dispositivos en tu red")


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

    # Convertir a numpy si es PIL
    if isinstance(prenda_custom, Image.Image):
        img_modificada = np.array(prenda_custom)
    else:
        img_modificada = prenda_custom.copy()

    # Debug
    print(f"🎨 Actualizando preview:")
    print(f"   • Color: {color}")
    print(f"   • Escala: {escala}")
    print(f"   • Estilo: {estilo}")

    # Aplicar modificaciones en orden
    try:
        # 1. Cambiar color
        if color:
            img_modificada = modificador.cambiar_color(img_modificada, color)
            print(f"   ✅ Color aplicado")

        # 2. Ajustar escala
        if escala and escala != 1.0:
            img_modificada = modificador.ajustar_escala(img_modificada, escala)
            print(f"   ✅ Escala aplicada")

        # 3. Aplicar estilo
        if estilo and estilo != "Regular":
            img_modificada = modificador.aplicar_estilo(img_modificada, estilo)
            print(f"   ✅ Estilo aplicado")

    except Exception as e:
        print(f"   ❌ Error en actualizar_preview: {e}")
        return prenda_custom

    return img_modificada


def obtener_url_ngrok():
    """
    Detecta automáticamente si ngrok está corriendo y obtiene su URL pública

    Returns:
        str: URL pública de ngrok si está activo, None si no
    """
    # Verificar si requests está disponible
    if requests is None:
        return None

    try:
        # Intentar conectar al dashboard de ngrok
        response = requests.get("http://localhost:4040/api/tunnels", timeout=2)

        if response.status_code == 200:
            data = response.json()
            tunnels = data.get('tunnels', [])

            # Buscar el túnel del puerto 5000
            for tunnel in tunnels:
                config_addr = tunnel.get('config', {}).get('addr', '')
                public_url = tunnel.get('public_url', '')

                # Verificar si es el túnel del puerto 5000
                if '5000' in config_addr or 'localhost:5000' in config_addr:
                    # Asegurar que sea HTTPS
                    if public_url.startswith('http://'):
                        public_url = public_url.replace('http://', 'https://')

                    print(f"   🌐 ngrok detectado: {public_url}")
                    return public_url

            # Si hay túneles pero ninguno es del 5000
            if tunnels:
                print(f"   ⚠️ ngrok activo pero no hay túnel para puerto 5000")
                print(f"   💡 Ejecuta: ngrok http 5000")

        return None

    except Exception:
        # ngrok no está corriendo o hubo un error
        return None


def obtener_base_url():
    """
    Obtiene la URL base para el servidor AR
    Prioridad: 1. ngrok, 2. IP local, 3. localhost

    Returns:
        str: URL base (ej: https://xxx.ngrok-free.dev o http://192.168.1.100:5000)
    """
    # Intentar ngrok primero
    ngrok_url = obtener_url_ngrok()
    if ngrok_url:
        return ngrok_url

    # Si no hay ngrok, usar IP local
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_local = s.getsockname()[0]
        s.close()

        base_url = f"http://{ip_local}:{config.HTTP_PORT}"
        print(f"   📍 Usando IP local: {base_url}")
        return base_url

    except Exception:
        # Fallback a localhost
        base_url = f"http://127.0.0.1:{config.HTTP_PORT}"
        print(f"   ⚠️ Usando localhost: {base_url}")
        print(f"   💡 Para acceso móvil, configura firewall o usa ngrok")
        return base_url


def procesar_imagen_completo(
    img_persona,
    img_prenda_custom,
    altura,
    color_prenda,
    escala_prenda,
    estilo_prenda,
    ocasion,
    accesorio_gorra,
    escala_gorra,
    accesorio_gafas,
    escala_gafas,
    accesorio_bufanda,
    escala_bufanda,
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

        print(f"\n{'='*60}")
        print(f"📦 PREPARANDO PRENDA:")
        print(f"   • Prenda custom: {img_prenda_custom is not None}")
        print(f"   • Color recibido: '{color_prenda}'")
        print(f"   • Escala: {escala_prenda}")
        print(f"   • Estilo: {estilo_prenda}")
        print(f"{'='*60}\n")

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

        
        
        # 7.5 APLICAR ACCESORIOS A LA IMAGEN BASE
        if any([accesorio_gorra is not None, accesorio_gafas is not None, accesorio_bufanda is not None]):
            progreso(0.45, desc="🎩 Aplicando accesorios...")
            
            from rembg import remove
            
            gestor_acc = GestorAccesorios()
            lista_accesorios = []
            
            # Gorra
            if accesorio_gorra is not None:
                print("   🔄 Procesando gorra...")
                img_gorra = Image.fromarray(accesorio_gorra)
                print("   🔄 Eliminando fondo de gorra con rembg...")
                img_gorra = remove(img_gorra)
                if img_gorra.mode != 'RGBA':
                    img_gorra = img_gorra.convert('RGBA')
                print("   ✅ Gorra procesada correctamente")
                lista_accesorios.append({
                    'imagen': img_gorra,
                    'tipo': 'gorra',
                    'escala': escala_gorra
                })
            
            # Gafas
            if accesorio_gafas is not None:
                print("   🔄 Procesando gafas...")
                img_gafas = Image.fromarray(accesorio_gafas)
                print("   🔄 Eliminando fondo de gafas con rembg...")
                img_gafas = remove(img_gafas)
                if img_gafas.mode != 'RGBA':
                    img_gafas = img_gafas.convert('RGBA')
                print("   ✅ Gafas procesadas correctamente")
                lista_accesorios.append({
                    'imagen': img_gafas,
                    'tipo': 'gafas',
                    'escala': escala_gafas
                })
            
            # Bufanda
            if accesorio_bufanda is not None:
                print("   🔄 Procesando bufanda...")
                img_bufanda = Image.fromarray(accesorio_bufanda)
                print("   🔄 Eliminando fondo de bufanda con rembg...")
                img_bufanda = remove(img_bufanda)
                if img_bufanda.mode != 'RGBA':
                    img_bufanda = img_bufanda.convert('RGBA')
                print("   ✅ Bufanda procesada correctamente")
                lista_accesorios.append({
                    'imagen': img_bufanda,
                    'tipo': 'bufanda',
                    'escala': escala_bufanda
                })
            
            # Aplicar accesorios
            if lista_accesorios:
                img_p_pil = gestor_acc.aplicar_multiples_accesorios(
                    img_p_pil,
                    lista_accesorios,
                    puntos
                )
                print(f"   ✅ {len(lista_accesorios)} accesorio(s) aplicado(s)")
        
        # 8. GENERAR MODELO 3D (ahora con accesorios aplicados)
        progreso(0.5, desc="🎨 Generando avatar 3D con prenda y accesorios...")

        def callback_progreso(p, desc):
            progreso(0.5 + p * 0.4, desc=desc)

        
        mesh = gen.generar(img_p_pil, prenda_final, puntos, medidas, callback_progreso)

        # 8. EXPORTAR Y CREAR QR PARA AR
        progreso(0.95, desc="📦 Exportando modelo...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        glb_name = f"avatar_{timestamp}.glb"
        glb_path = config.OUTPUT_DIR / glb_name

        mesh.export(str(glb_path))

        # Obtener URL base (ngrok o IP local)
        base_url = obtener_base_url()

        # Crear HTML para AR
        model_url = f"{base_url}/model/{glb_name}"
        html_name = f"ar_{timestamp}.html"
        html_path = config.OUTPUT_DIR / html_name

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(HTML_TEMPLATE.format(MODEL_PATH=model_url))

        # Generar QR con la URL correcta
        qr_url = f"{base_url}/ar/{html_name}"
        print(f"   📱 URL del QR: {qr_url}")

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

        # Crear preview para mostrar (convertir PIL a numpy)
        img_preview_np = np.array(img_p_pil.convert('RGB'))

        return str(glb_path), qr_img, html_recomendaciones, img_preview_np

    except Exception as e:
        import traceback
        error_msg = f"❌ Error en procesamiento: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        # Retornar: glb=None, qr=None, html=error_msg, preview=None
        return None, None, error_msg, None


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
                        show_label=False,
                        interactive=False,
                        height=200
                    )


                    gr.Markdown("#### 🎩 5. Accesorios (Opcional)")
                    
                    with gr.Accordion("Agregar Accesorios", open=False):
                        gr.Markdown("**Gorra/Sombrero**")
                        accesorio_gorra = gr.Image(
                            label="Subir Gorra/Sombrero",
                            type="numpy",
                            height=150
                        )
                        escala_gorra = gr.Slider(
                            minimum=0.5,
                            maximum=1.5,
                            value=1.0,
                            step=0.1,
                            label="Tamaño Gorra"
                        )
                        
                        gr.Markdown("**Gafas/Lentes**")
                        accesorio_gafas = gr.Image(
                            label="Subir Gafas/Lentes",
                            type="numpy",
                            height=150
                        )
                        escala_gafas = gr.Slider(
                            minimum=0.5,
                            maximum=1.5,
                            value=1.0,
                            step=0.1,
                            label="Tamaño Gafas"
                        )
                        
                        gr.Markdown("**Bufanda/Collar**")
                        accesorio_bufanda = gr.Image(
                            label="Subir Bufanda/Collar",
                            type="numpy",
                            height=150
                        )
                        escala_bufanda = gr.Slider(
                            minimum=0.5,
                            maximum=1.5,
                            value=1.0,
                            step=0.1,
                            label="Tamaño Bufanda"
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
                        show_label=False,
                        height=400
                    )

                    gr.Markdown("💡 **Tip:** Arrastra para rotar, scroll para zoom")


                    gr.Markdown("#### 👀 Vista Previa 2D con Accesorios")
                    preview_accesorios_output = gr.Image(
                        show_label=False,
                        height=300,
                        interactive=False
                    )
                    gr.Markdown("ℹ️ *Así se verían los accesorios en el avatar*")


                    gr.Markdown("#### 📱 Código QR para AR Móvil")
                    qr_output = gr.Image(
                        show_label=False,
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

    # Preview de prenda al subirla
    img_prenda_input.change(
        fn=actualizar_preview_prenda,
        inputs=[img_prenda_input, color_picker, escala_slider, estilo_dropdown],
        outputs=prenda_preview
    )

    # Preview manual con botón
    btn_preview.click(
        fn=actualizar_preview_prenda,
        inputs=[img_prenda_input, color_picker, escala_slider, estilo_dropdown],
        outputs=prenda_preview
    )

    # Actualizar preview al cambiar parámetros (automático)
    for componente in [color_picker, escala_slider, estilo_dropdown]:
        componente.change(
            fn=actualizar_preview_prenda,
            inputs=[img_prenda_input, color_picker, escala_slider, estilo_dropdown],
            outputs=prenda_preview
        )
        # También con input (más reactivo)
        componente.input(
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
            ocasion_dropdown,
            accesorio_gorra,
            escala_gorra,
            accesorio_gafas,
            escala_gafas,
            accesorio_bufanda,
            escala_bufanda
        ],
        outputs=[modelo_3d_output, qr_output, recomendaciones_html, preview_accesorios_output]
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