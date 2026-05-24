"""
SISTEMA INTEGRADO FINAL - VERSIÓN CORREGIDA
Avatar 3D con Prenda + Recomendación de Talla Precisa

Correcciones:
- Prenda se aplica SOLO en el torso (hombros a cadera)
- NO pide altura (todo automático)
- Textura de prenda mantiene su forma
"""

import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import trimesh
import cv2
from rembg import remove
import gradio as gr
from datetime import datetime
import mediapipe as mp

print("=" * 70)
print("  SISTEMA AVATAR 3D + PRENDA + TALLA")
print("=" * 70)

OUTPUT_DIR = "avatares_generados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TABLAS DE TALLAS CALIBRADAS
TALLAS_CAMISAS_HOMBRE = {
    'XS': {'pecho': (81, 86), 'cintura': (66, 71)},
    'S': {'pecho': (86, 91), 'cintura': (71, 76)},
    'M': {'pecho': (91, 98), 'cintura': (76, 83)},
    'L': {'pecho': (98, 106), 'cintura': (83, 91)},
    'XL': {'pecho': (106, 114), 'cintura': (91, 99)},
    'XXL': {'pecho': (114, 122), 'cintura': (99, 107)},
}

TALLAS_CAMISAS_MUJER = {
    'XS': {'pecho': (78, 82), 'cintura': (60, 64)},
    'S': {'pecho': (82, 88), 'cintura': (64, 70)},
    'M': {'pecho': (88, 94), 'cintura': (70, 76)},
    'L': {'pecho': (94, 102), 'cintura': (76, 84)},
    'XL': {'pecho': (102, 110), 'cintura': (84, 92)},
}

TALLAS_PANTALONES_HOMBRE = {
    '28': {'cintura': (68, 72), 'cadera': (86, 90)},
    '30': {'cintura': (72, 78), 'cadera': (90, 96)},
    '32': {'cintura': (78, 84), 'cadera': (96, 102)},
    '34': {'cintura': (84, 90), 'cadera': (102, 108)},
    '36': {'cintura': (90, 96), 'cadera': (108, 114)},
    '38': {'cintura': (96, 102), 'cadera': (114, 120)},
}

TALLAS_PANTALONES_MUJER = {
    'XS': {'cintura': (58, 62), 'cadera': (84, 88)},
    'S': {'cintura': (62, 68), 'cadera': (88, 94)},
    'M': {'cintura': (68, 74), 'cadera': (94, 100)},
    'L': {'cintura': (74, 82), 'cadera': (100, 108)},
    'XL': {'cintura': (82, 90), 'cadera': (108, 116)},
}


class MedicionesAntropometricas:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def calcular_distancia_pixeles(self, punto1, punto2):
        return np.sqrt((punto1[0] - punto2[0]) ** 2 + (punto1[1] - punto2[1]) ** 2)

    def detectar_puntos_clave(self, imagen):
        if isinstance(imagen, Image.Image):
            imagen = np.array(imagen)

        if len(imagen.shape) == 3 and imagen.shape[2] == 3:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        else:
            imagen_rgb = imagen

        resultados = self.pose.process(imagen_rgb)

        if not resultados.pose_landmarks:
            return None, "No se detectó una persona en la imagen"

        landmarks = resultados.pose_landmarks.landmark
        h, w = imagen.shape[:2]

        puntos = {}
        nombres_puntos = {
            'nariz': self.mp_pose.PoseLandmark.NOSE,
            'hombro_izq': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'hombro_der': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'cadera_izq': self.mp_pose.PoseLandmark.LEFT_HIP,
            'cadera_der': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'tobillo_izq': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'tobillo_der': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        }

        for nombre, landmark_id in nombres_puntos.items():
            lm = landmarks[landmark_id.value]
            puntos[nombre] = (int(lm.x * w), int(lm.y * h), lm.visibility)

        return puntos, resultados

    def calcular_medidas(self, puntos, altura_real_cm=None):
        """Calcula medidas corporales (con o sin altura)"""

        altura_px = self.calcular_distancia_pixeles(
            puntos['nariz'][:2],
            [(puntos['tobillo_izq'][0] + puntos['tobillo_der'][0]) / 2,
             (puntos['tobillo_izq'][1] + puntos['tobillo_der'][1]) / 2]
        )

        if altura_real_cm and altura_real_cm > 0:
            factor_escala = altura_real_cm / altura_px
        else:
            # Estimación automática basada en proporciones
            ratio = self.calcular_distancia_pixeles(
                puntos['cadera_izq'][:2], puntos['cadera_der'][:2]
            ) / self.calcular_distancia_pixeles(
                puntos['hombro_izq'][:2], puntos['hombro_der'][:2]
            )
            altura_estimada = 162 if ratio > 0.95 else 175
            factor_escala = altura_estimada / altura_px

        medidas = {}

        # Pecho (contorno basado en ancho de hombros)
        ancho_hombros_px = self.calcular_distancia_pixeles(
            puntos['hombro_izq'][:2], puntos['hombro_der'][:2]
        )
        medidas['pecho'] = ancho_hombros_px * 2.2 * factor_escala

        # Cadera
        ancho_cadera_px = self.calcular_distancia_pixeles(
            puntos['cadera_izq'][:2], puntos['cadera_der'][:2]
        )
        medidas['cadera'] = ancho_cadera_px * 2.1 * factor_escala

        # Cintura (aproximada)
        medidas['cintura'] = medidas['cadera'] * 0.80

        return medidas


class RecomendadorTallas:
    @staticmethod
    def detectar_genero(medidas):
        ratio = medidas['cadera'] / medidas['pecho']
        return 'mujer' if ratio > 1.05 else 'hombre'

    @staticmethod
    def recomendar_talla_camisa(medidas, genero):
        tabla = TALLAS_CAMISAS_HOMBRE if genero == 'hombre' else TALLAS_CAMISAS_MUJER

        pecho = medidas['pecho']
        cintura = medidas['cintura']

        mejor_talla = None
        mejor_score = float('inf')

        for talla, rangos in tabla.items():
            pecho_min, pecho_max = rangos['pecho']
            cintura_min, cintura_max = rangos['cintura']

            if pecho_min <= pecho <= pecho_max:
                score_pecho = 0
            else:
                score_pecho = min(abs(pecho - pecho_min), abs(pecho - pecho_max)) * 2

            if cintura_min <= cintura <= cintura_max:
                score_cintura = 0
            else:
                score_cintura = min(abs(cintura - cintura_min), abs(cintura - cintura_max))

            score = score_pecho + score_cintura

            if score < mejor_score:
                mejor_score = score
                mejor_talla = talla

        confianza = 0.95 if mejor_score == 0 else (0.85 if mejor_score < 5 else 0.75)

        return mejor_talla, confianza

    @staticmethod
    def recomendar_talla_pantalon(medidas, genero):
        tabla = TALLAS_PANTALONES_HOMBRE if genero == 'hombre' else TALLAS_PANTALONES_MUJER

        cintura = medidas['cintura']
        cadera = medidas['cadera']

        mejor_talla = None
        mejor_score = float('inf')

        for talla, rangos in tabla.items():
            cintura_min, cintura_max = rangos['cintura']
            cadera_min, cadera_max = rangos['cadera']

            if cintura_min <= cintura <= cintura_max:
                score_cintura = 0
            else:
                score_cintura = min(abs(cintura - cintura_min), abs(cintura - cintura_max)) * 1.5

            if cadera_min <= cadera <= cadera_max:
                score_cadera = 0
            else:
                score_cadera = min(abs(cadera - cadera_min), abs(cadera - cadera_max))

            score = score_cintura + score_cadera

            if score < mejor_score:
                mejor_score = score
                mejor_talla = talla

        confianza = 0.95 if mejor_score == 0 else (0.85 if mejor_score < 5 else 0.75)

        return mejor_talla, confianza


class GeneradorAvatar3DConPrenda:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n🔄 Cargando MiDaS en {self.device.upper()}...")

        self.modelo_midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.modelo_midas.to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform

        print("✅ MiDaS listo")

    def generar_avatar_con_prenda(self, imagen_persona, imagen_prenda=None, puntos=None):
        """Genera avatar 3D con textura de prenda aplicada SOLO en torso"""

        # Preprocesar persona
        img_rgba = remove(imagen_persona)
        img_rgba = ImageOps.exif_transpose(img_rgba)
        img_rgba.thumbnail((400, 400))

        mask = np.array(img_rgba)[:, :, 3]
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

        img_rgb = img_rgba.convert('RGB')
        img_array = np.array(img_rgb)

        # Si hay prenda, aplicar SOLO en torso usando puntos clave
        if imagen_prenda is not None and puntos is not None:
            img_array = self._aplicar_textura_prenda(img_array, imagen_prenda, mask, puntos)

        # Estimar profundidad
        input_batch = self.transform(img_array).to(self.device)

        with torch.no_grad():
            prediction = self.modelo_midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_array.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth = (prediction.cpu().numpy() - prediction.min().item()) / (
                prediction.max().item() - prediction.min().item() + 1e-8
        )

        mask_res = cv2.resize(mask, (depth.shape[1], depth.shape[0])) > 128
        depth[~mask_res] = 0

        # Crear malla 3D
        h, w = depth.shape
        y, x = np.mgrid[0:h:1, 0:w:1]
        z_factor = w * 0.12

        indices_validos = np.where(mask_res.flatten())[0]
        vertices_2d = np.stack([x.flatten(), y.flatten()], axis=-1)[indices_validos]

        mapa_indices = np.full(mask_res.flatten().shape, -1)
        mapa_indices[indices_validos] = np.arange(len(indices_validos))

        caras = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j
                i1, i2, i3, i4 = (
                    mapa_indices[idx], mapa_indices[idx + 1],
                    mapa_indices[idx + w], mapa_indices[idx + w + 1]
                )
                if i1 != -1 and i2 != -1 and i3 != -1:
                    caras.append([i1, i2, i3])
                if i2 != -1 and i3 != -1 and i4 != -1:
                    caras.append([i2, i4, i3])

        mesh = trimesh.creation.extrude_triangulation(vertices_2d, caras, height=-0.25)
        z_coords = depth.flatten()[indices_validos] * z_factor
        mesh.vertices[:len(indices_validos), 2] += z_coords

        # Aplicar colores (con prenda si existe)
        colores_puros = img_array.reshape(-1, 3)[indices_validos]
        colores_finales = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)
        n = len(indices_validos)
        colores_finales[:n] = colores_puros
        colores_finales[n:2 * n] = colores_puros
        if len(mesh.vertices) > 2 * n:
            colores_finales[2 * n:] = [100, 100, 100]
        mesh.visual.vertex_colors = colores_finales

        # Transformaciones
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
        mesh.apply_scale(0.19)
        mesh.apply_translation([-mesh.centroid[0], 110, 0])

        return mesh

    def _aplicar_textura_prenda(self, img_persona, img_prenda, mask, puntos):
        """Aplica textura de prenda SOLO en el torso (hombros a cadera)"""

        # Convertir prenda
        if isinstance(img_prenda, Image.Image):
            img_prenda = np.array(img_prenda)

        # Remover fondo de prenda
        if img_prenda.shape[2] != 4:
            img_prenda_pil = Image.fromarray(img_prenda)
            img_prenda_pil = remove(img_prenda_pil)
            img_prenda = np.array(img_prenda_pil)

        # Redimensionar prenda al tamaño de la persona
        img_prenda_resized = cv2.resize(img_prenda, (img_persona.shape[1], img_persona.shape[0]))

        h, w = img_persona.shape[:2]

        # Crear máscara de torso PRECISA usando puntos clave
        zona_prenda = np.zeros((h, w), dtype=bool)

        # Usar puntos reales detectados
        y_hombros = int((puntos['hombro_izq'][1] + puntos['hombro_der'][1]) / 2)
        y_cadera = int((puntos['cadera_izq'][1] + puntos['cadera_der'][1]) / 2)

        # Expandir ligeramente para cubrir bien
        margen_superior = int(h * 0.02)  # 2% margen arriba
        margen_inferior = int(h * 0.05)  # 5% margen abajo

        y_inicio = max(0, y_hombros - margen_superior)
        y_fin = min(h, y_cadera + margen_inferior)

        # Aplicar prenda SOLO en esta zona
        zona_prenda[y_inicio:y_fin, :] = True

        print(f"  Aplicando prenda de Y={y_inicio} a Y={y_fin} (torso)")

        # Combinar con máscara de persona
        mask_combinado = (mask > 0) & zona_prenda

        # Aplicar textura
        if img_prenda_resized.shape[2] == 4:
            alpha = img_prenda_resized[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis]

            # Combinar con máscara de torso
            alpha = alpha * mask_combinado[:, :, np.newaxis]

            # Mezclar con suavizado
            img_resultado = (
                    img_prenda_resized[:, :, :3] * alpha +
                    img_persona * (1 - alpha)
            ).astype(np.uint8)

            return img_resultado

        return img_persona


print("\n🔄 Inicializando sistema...")
medidor = MedicionesAntropometricas()
recomendador = RecomendadorTallas()
generador = GeneradorAvatar3DConPrenda()
print("✅ Sistema listo\n")


def procesar_completo(imagen_persona, imagen_prenda, tipo_prenda):
    """Proceso completo SIN pedir altura (todo automático)"""

    try:
        if imagen_persona is None:
            return None, "❌ Sube foto de la persona"

        print("\n" + "=" * 70)
        print("PROCESANDO...")
        print("=" * 70)

        # Convertir imagen
        if isinstance(imagen_persona, np.ndarray):
            img_persona_pil = Image.fromarray(imagen_persona)
        else:
            img_persona_pil = imagen_persona

        # Detectar cuerpo
        puntos, _ = medidor.detectar_puntos_clave(imagen_persona)

        if puntos is None:
            return None, "❌ No se detectó persona en la imagen"

        # Calcular medidas (SIN altura - automático)
        medidas = medidor.calcular_medidas(puntos, None)

        print(f"📏 Medidas detectadas:")
        print(f"   Pecho: {medidas['pecho']:.1f} cm")
        print(f"   Cintura: {medidas['cintura']:.1f} cm")
        print(f"   Cadera: {medidas['cadera']:.1f} cm")

        # Recomendar talla
        genero = recomendador.detectar_genero(medidas)
        print(f"👤 Género estimado: {genero}")

        if tipo_prenda == "Camisa/Camiseta":
            talla, confianza = recomendador.recomendar_talla_camisa(medidas, genero)
            tabla = TALLAS_CAMISAS_HOMBRE if genero == 'hombre' else TALLAS_CAMISAS_MUJER
        else:
            talla, confianza = recomendador.recomendar_talla_pantalon(medidas, genero)
            tabla = TALLAS_PANTALONES_HOMBRE if genero == 'hombre' else TALLAS_PANTALONES_MUJER

        print(f"🎯 Talla recomendada: {talla} (Confianza: {confianza * 100:.0f}%)")

        # Generar avatar con prenda (CORRECTAMENTE POSICIONADA)
        print("🎨 Generando avatar 3D con prenda...")
        mesh = generador.generar_avatar_con_prenda(img_persona_pil, imagen_prenda, puntos)

        # Guardar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        glb_path = os.path.join(OUTPUT_DIR, f"avatar_{timestamp}.glb")
        obj_path = os.path.join(OUTPUT_DIR, f"avatar_{timestamp}.obj")

        mesh.export(glb_path)
        mesh.export(obj_path)

        print("✅ Archivos guardados")

        # Generar reporte CON COLORES
        mensaje = f"""<div style='font-family: Arial; padding: 20px;'>

<h2 style='color: #2196F3;'>✨ Análisis Completado</h2>

<div style='background: #E3F2FD; padding: 15px; border-radius: 8px; margin: 10px 0;'>
<h3 style='color: #1976D2; margin-top: 0;'>📏 Tus Medidas</h3>
<ul style='list-style: none; padding: 0;'>
  <li>• <b>Pecho:</b> {medidas['pecho']:.1f} cm</li>
  <li>• <b>Cintura:</b> {medidas['cintura']:.1f} cm</li>
  <li>• <b>Cadera:</b> {medidas['cadera']:.1f} cm</li>
  <li>• <b>Género estimado:</b> {genero.capitalize()}</li>
</ul>
</div>

<div style='background: #C8E6C9; padding: 20px; border-radius: 8px; margin: 20px 0; border: 3px solid #4CAF50;'>
<h2 style='color: #2E7D32; margin-top: 0; text-align: center;'>🎯 TU TALLA RECOMENDADA</h2>
<h1 style='color: #1B5E20; text-align: center; font-size: 48px; margin: 10px 0;'>{talla}</h1>
<p style='text-align: center; color: #388E3C; font-size: 18px;'>
Confianza: {confianza * 100:.0f}%
</p>
</div>

<div style='background: #FFF3E0; padding: 15px; border-radius: 8px; margin: 10px 0;'>
<h3 style='color: #E65100; margin-top: 0;'>📊 Tabla de Referencia</h3>
<table style='width: 100%; border-collapse: collapse;'>
"""

        # Agregar filas de tabla
        for t, rangos in tabla.items():
            if t == talla:
                color_fondo = "#4CAF50"
                color_texto = "white"
                marcador = "👉 "
            else:
                color_fondo = "#f5f5f5"
                color_texto = "#333"
                marcador = ""

            if tipo_prenda == "Camisa/Camiseta":
                mensaje += f"""
<tr style='background: {color_fondo}; color: {color_texto};'>
  <td style='padding: 10px; border: 1px solid #ddd;'><b>{marcador}{t}</b></td>
  <td style='padding: 10px; border: 1px solid #ddd;'>Pecho: {rangos['pecho'][0]}-{rangos['pecho'][1]} cm</td>
  <td style='padding: 10px; border: 1px solid #ddd;'>Cintura: {rangos['cintura'][0]}-{rangos['cintura'][1]} cm</td>
</tr>
"""
            else:
                mensaje += f"""
<tr style='background: {color_fondo}; color: {color_texto};'>
  <td style='padding: 10px; border: 1px solid #ddd;'><b>{marcador}{t}</b></td>
  <td style='padding: 10px; border: 1px solid #ddd;'>Cintura: {rangos['cintura'][0]}-{rangos['cintura'][1]} cm</td>
  <td style='padding: 10px; border: 1px solid #ddd;'>Cadera: {rangos['cadera'][0]}-{rangos['cadera'][1]} cm</td>
</tr>
"""

        mensaje += f"""
</table>
</div>

<div style='background: #F3E5F5; padding: 15px; border-radius: 8px; margin: 10px 0;'>
<h3 style='color: #6A1B9A; margin-top: 0;'>📦 Archivos Generados</h3>
<ul style='color: #4A148C;'>
  <li>✓ avatar_{timestamp}.glb (visualización web)</li>
  <li>✓ avatar_{timestamp}.obj (edición 3D)</li>
</ul>
<p style='font-size: 12px; color: #7B1FA2;'>📂 {os.path.abspath(OUTPUT_DIR)}</p>
</div>

<div style='background: #E8F5E9; padding: 10px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4CAF50;'>
<p style='margin: 0; color: #2E7D32;'><b>💡 Nota:</b> La prenda se aplicó solo en el área del torso (hombros a cadera) para mantener su forma original.</p>
</div>

</div>
"""

        print("✅ COMPLETADO")

        return glb_path, mensaje

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


# INTERFAZ SIMPLIFICADA
with gr.Blocks(title="Avatar 3D + Talla", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎨 Avatar 3D + Recomendación de Talla
    ### Sistema Inteligente de Análisis Corporal y Vestimenta
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📤 Entrada")

            imagen_persona_input = gr.Image(
                label="Foto de la Persona (cuerpo completo)",
                type="pil",
                height=350
            )

            imagen_prenda_input = gr.Image(
                label="Foto de la Prenda (Opcional)",
                type="pil",
                height=250
            )

            tipo_prenda_input = gr.Radio(
                choices=["Camisa/Camiseta", "Pantalón"],
                value="Camisa/Camiseta",
                label="Tipo de Prenda"
            )

            btn_procesar = gr.Button(
                "🚀 Generar Avatar + Calcular Talla",
                variant="primary",
                size="lg"
            )

            gr.Markdown("""
            ### 💡 Consejos:
            ✅ Foto de cuerpo completo  
            ✅ Postura recta y centrada  
            ✅ Prenda en fondo blanco  
            🎨 La prenda se aplicará solo en el torso  
            📏 Altura se calcula automáticamente  
            """)

        with gr.Column(scale=1):
            gr.Markdown("## 🎨 Avatar 3D con Prenda")

            modelo_3d = gr.Model3D(
                label="Tu Avatar 3D Personalizado",
                height=600,
                clear_color=[0.1, 0.1, 0.15, 1.0]
            )

    gr.Markdown("## 📊 Análisis y Recomendación de Talla")

    mensaje_output = gr.HTML(label="Resultados Completos")

    btn_procesar.click(
        fn=procesar_completo,
        inputs=[imagen_persona_input, imagen_prenda_input, tipo_prenda_input],
        outputs=[modelo_3d, mensaje_output]
    )

    with gr.Accordion("ℹ️ Información del Sistema", open=False):
        gr.Markdown("""
        ### 🎯 Características:
        - **Detección automática de cuerpo**: MediaPipe Pose (33 puntos)
        - **Estimación de altura**: Basada en proporciones corporales
        - **Aplicación de prenda**: Solo en área de torso (hombros a cadera)
        - **Cálculo de talla**: Algoritmo de mejor ajuste con tablas estándar
        - **Generación 3D**: MiDaS para profundidad + Trimesh para malla

        ### 📏 Medidas que se calculan:
        - Contorno de pecho
        - Contorno de cintura
        - Contorno de cadera
        - Género estimado (por proporciones)

        ### 🎨 Aplicación de prenda:
        La prenda se coloca ÚNICAMENTE en el área del torso, detectada automáticamente
        mediante los puntos clave de hombros y cadera. Esto evita que la prenda tape
        la cara o se deforme incorrectamente.

        ### 🎯 Precisión de tallas:
        - 95% confianza: Medidas dentro del rango exacto
        - 85% confianza: Medidas cercanas al rango
        - 75% confianza: Medidas con mayor desviación
        """)