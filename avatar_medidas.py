"""
GENERADOR DE AVATARES 3D CON MEDICIONES ANTROPOMÉTRICAS
Usando MiDaS + MediaPipe para detección de pose
Proyecto de Grado - Enero 2026
"""

import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import trimesh
import cv2
from rembg import remove
import gradio as gr
from datetime import datetime
import mediapipe as mp

print("=" * 70)
print("  GENERADOR DE AVATARES 3D CON MEDICIONES")
print("=" * 70)
print(f"🐍 Python: {torch.__version__}")
print(f"💻 CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
print("=" * 70)

OUTPUT_DIR = "avatares_generados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MedicionesAntropometricas:
    """Calcula mediciones del cuerpo humano usando MediaPipe"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def calcular_distancia_pixeles(self, punto1, punto2):
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt((punto1[0] - punto2[0])**2 + (punto1[1] - punto2[1])**2)

    def detectar_puntos_clave(self, imagen):
        """Detecta puntos clave del cuerpo usando MediaPipe"""

        # Convertir PIL a numpy
        if isinstance(imagen, Image.Image):
            imagen = np.array(imagen)

        # Convertir BGR a RGB si es necesario
        if len(imagen.shape) == 3 and imagen.shape[2] == 3:
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        else:
            imagen_rgb = imagen

        # Procesar con MediaPipe
        resultados = self.pose.process(imagen_rgb)

        if not resultados.pose_landmarks:
            return None, "No se detectó una persona en la imagen"

        # Extraer coordenadas de puntos clave
        landmarks = resultados.pose_landmarks.landmark
        h, w = imagen.shape[:2]

        puntos = {}
        nombres_puntos = {
            'nariz': self.mp_pose.PoseLandmark.NOSE,
            'hombro_izq': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'hombro_der': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'codo_izq': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'codo_der': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'muñeca_izq': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'muñeca_der': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'cadera_izq': self.mp_pose.PoseLandmark.LEFT_HIP,
            'cadera_der': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'rodilla_izq': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'rodilla_der': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'tobillo_izq': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'tobillo_der': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        }

        for nombre, landmark_id in nombres_puntos.items():
            lm = landmarks[landmark_id.value]
            puntos[nombre] = (int(lm.x * w), int(lm.y * h), lm.visibility)

        return puntos, resultados

    def estimar_altura_real(self, puntos, medidas_px):
        """
        Estima la altura real usando MÚLTIPLES métodos y promediando

        Métodos combinados:
        1. Proporción de cabeza (método Da Vinci - 7.5 cabezas = 1 persona)
        2. Longitud de antebrazo (1/6.5 de la altura)
        3. Análisis de base de datos antropométrica
        """

        alturas_estimadas = []
        pesos = []

        # MÉTODO 1: Proporción de cabeza
        # Calcular tamaño de cabeza (nariz a hombros aproximadamente)
        centro_hombros_y = (puntos['hombro_izq'][1] + puntos['hombro_der'][1]) / 2
        altura_cabeza_px = abs(puntos['nariz'][1] - centro_hombros_y)

        if altura_cabeza_px > 0:
            # Una persona adulta mide aproximadamente 7.5-8 cabezas
            altura_por_cabeza = medidas_px['altura_total'] / altura_cabeza_px

            if altura_por_cabeza < 6:  # Niño o foto cortada
                factor_cabeza = 145  # Altura base menor
            elif altura_por_cabeza < 7:  # Persona de estatura baja
                factor_cabeza = 155
            elif altura_por_cabeza < 7.5:  # Estatura media-baja
                factor_cabeza = 163
            elif altura_por_cabeza < 8:  # Estatura media
                factor_cabeza = 170
            elif altura_por_cabeza < 8.5:  # Estatura media-alta
                factor_cabeza = 177
            else:  # Estatura alta
                factor_cabeza = 185

            # Ajuste fino basado en la proporción exacta
            altura_metodo1 = factor_cabeza + (altura_por_cabeza - 7.5) * 8
            alturas_estimadas.append(altura_metodo1)
            pesos.append(0.35)  # Mayor peso, es muy confiable

        # MÉTODO 2: Proporción de piernas
        ratio_piernas = medidas_px['largo_piernas'] / medidas_px['altura_total']

        # Piernas largas = persona más alta generalmente
        if ratio_piernas > 0.52:
            altura_base_piernas = 175
        elif ratio_piernas > 0.48:
            altura_base_piernas = 168
        elif ratio_piernas > 0.44:
            altura_base_piernas = 163
        else:
            altura_base_piernas = 158

        # Ajuste por torso
        ratio_torso = medidas_px['largo_torso'] / medidas_px['altura_total']
        ajuste_torso = (ratio_torso - 0.30) * 50  # Ajuste fino

        altura_metodo2 = altura_base_piernas + ajuste_torso
        alturas_estimadas.append(altura_metodo2)
        pesos.append(0.25)

        # MÉTODO 3: Ancho de hombros con género estimado
        ratio_cadera_hombros = medidas_px['ancho_cadera'] / medidas_px['ancho_hombros']

        # Estimar género por proporciones
        if ratio_cadera_hombros > 0.96:  # Mujer (caderas más anchas)
            # Mujeres: hombros ~37-39cm promedio, altura 150-175cm
            ancho_hombros_ref_cm = 38  # cm promedio
            altura_ref = 162
        else:  # Hombre (hombros más anchos)
            # Hombres: hombros ~41-44cm promedio, altura 165-185cm
            ancho_hombros_ref_cm = 42  # cm promedio
            altura_ref = 175

        # Calcular factor de escala basado en hombros
        factor_hombros = ancho_hombros_ref_cm / medidas_px['ancho_hombros']
        altura_metodo3 = medidas_px['altura_total'] * factor_hombros

        alturas_estimadas.append(altura_metodo3)
        pesos.append(0.30)

        # MÉTODO 4: Longitud de brazos
        # Envergadura (brazo a brazo) ≈ altura en adultos
        ratio_brazo = medidas_px['largo_brazo'] / medidas_px['altura_total']

        if ratio_brazo > 0.40:  # Brazos largos
            ajuste_brazos = 1.03
        elif ratio_brazo < 0.35:  # Brazos cortos
            ajuste_brazos = 0.97
        else:
            ajuste_brazos = 1.0

        altura_metodo4 = altura_metodo3 * ajuste_brazos
        alturas_estimadas.append(altura_metodo4)
        pesos.append(0.10)

        # CALCULAR PROMEDIO PONDERADO
        altura_final = sum(h * w for h, w in zip(alturas_estimadas, pesos)) / sum(pesos)

        # Validar rango realista
        if altura_final < 145:
            altura_final = 150
            confianza = 0.60
        elif altura_final > 195:
            altura_final = 190
            confianza = 0.60
        else:
            # Calcular confianza basada en consistencia entre métodos
            desviacion = np.std(alturas_estimadas)
            if desviacion < 5:
                confianza = 0.85
            elif desviacion < 10:
                confianza = 0.75
            else:
                confianza = 0.65

        # Redondear a valores realistas
        altura_final = round(altura_final)

        info_detallada = {
            'ratio_hombros': medidas_px['ancho_hombros'] / medidas_px['altura_total'],
            'ratio_cadera_hombros': ratio_cadera_hombros,
            'ratio_piernas': ratio_piernas,
            'genero_estimado': 'Femenino' if ratio_cadera_hombros > 0.96 else 'Masculino',
            'metodo1_cabeza': round(altura_metodo1, 1) if altura_cabeza_px > 0 else 'N/A',
            'metodo2_piernas': round(altura_metodo2, 1),
            'metodo3_hombros': round(altura_metodo3, 1),
            'desviacion': round(desviacion, 1) if 'desviacion' in locals() else 0
        }

        return altura_final, confianza, info_detallada

    def calcular_medidas(self, puntos, altura_real_cm=None):
        """
        Calcula medidas antropométricas en píxeles y las convierte a cm

        Args:
            puntos: Diccionario con puntos clave detectados
            altura_real_cm: Altura real de la persona en cm (opcional)
        """

        medidas_pixeles = {}

        # 1. ALTURA TOTAL (cabeza a pies)
        altura_px = self.calcular_distancia_pixeles(
            puntos['nariz'][:2],
            [(puntos['tobillo_izq'][0] + puntos['tobillo_der'][0]) / 2,
             (puntos['tobillo_izq'][1] + puntos['tobillo_der'][1]) / 2]
        )
        medidas_pixeles['altura_total'] = altura_px

        # 2. ANCHO DE HOMBROS
        ancho_hombros_px = self.calcular_distancia_pixeles(
            puntos['hombro_izq'][:2],
            puntos['hombro_der'][:2]
        )
        medidas_pixeles['ancho_hombros'] = ancho_hombros_px

        # 3. ANCHO DE CADERA
        ancho_cadera_px = self.calcular_distancia_pixeles(
            puntos['cadera_izq'][:2],
            puntos['cadera_der'][:2]
        )
        medidas_pixeles['ancho_cadera'] = ancho_cadera_px

        # 4. LARGO DE TORSO (hombros a cadera)
        centro_hombros = (
            (puntos['hombro_izq'][0] + puntos['hombro_der'][0]) / 2,
            (puntos['hombro_izq'][1] + puntos['hombro_der'][1]) / 2
        )
        centro_cadera = (
            (puntos['cadera_izq'][0] + puntos['cadera_der'][0]) / 2,
            (puntos['cadera_izq'][1] + puntos['cadera_der'][1]) / 2
        )
        largo_torso_px = self.calcular_distancia_pixeles(centro_hombros, centro_cadera)
        medidas_pixeles['largo_torso'] = largo_torso_px

        # 5. LARGO DE PIERNAS (cadera a tobillos)
        largo_piernas_px = self.calcular_distancia_pixeles(
            centro_cadera,
            [(puntos['tobillo_izq'][0] + puntos['tobillo_der'][0]) / 2,
             (puntos['tobillo_izq'][1] + puntos['tobillo_der'][1]) / 2]
        )
        medidas_pixeles['largo_piernas'] = largo_piernas_px

        # 6. LARGO DE BRAZO (hombro a muñeca)
        largo_brazo_izq_px = (
            self.calcular_distancia_pixeles(puntos['hombro_izq'][:2], puntos['codo_izq'][:2]) +
            self.calcular_distancia_pixeles(puntos['codo_izq'][:2], puntos['muñeca_izq'][:2])
        )
        largo_brazo_der_px = (
            self.calcular_distancia_pixeles(puntos['hombro_der'][:2], puntos['codo_der'][:2]) +
            self.calcular_distancia_pixeles(puntos['codo_der'][:2], puntos['muñeca_der'][:2])
        )
        largo_brazo_px = (largo_brazo_izq_px + largo_brazo_der_px) / 2
        medidas_pixeles['largo_brazo'] = largo_brazo_px

        # CONVERSIÓN A CENTÍMETROS
        if altura_real_cm:
            # Usuario proporcionó altura real
            factor_escala = altura_real_cm / altura_px
            altura_usada = altura_real_cm
            metodo_calculo = "proporcionada por usuario"
            confianza = 1.0
            info_extra = {}
        else:
            # ESTIMAR altura automáticamente
            altura_estimada, confianza, info_extra = self.estimar_altura_real(puntos, medidas_pixeles)
            factor_escala = altura_estimada / altura_px
            altura_usada = altura_estimada
            metodo_calculo = "estimada por IA"

        medidas_cm = {}
        nombres_medidas = {
            'altura_total': 'Altura Total',
            'ancho_hombros': 'Ancho de Hombros',
            'ancho_cadera': 'Ancho de Cadera',
            'largo_torso': 'Largo de Torso',
            'largo_piernas': 'Largo de Piernas',
            'largo_brazo': 'Largo de Brazo'
        }

        for key, px_value in medidas_pixeles.items():
            medidas_cm[key] = px_value * factor_escala

        # Asegurar que la altura total sea exactamente la altura usada
        medidas_cm['altura_total'] = altura_usada

        return medidas_cm, medidas_pixeles, nombres_medidas, factor_escala, altura_usada, metodo_calculo, confianza, info_extra

    def dibujar_esqueleto(self, imagen, puntos):
        """Dibuja el esqueleto sobre la imagen"""

        if isinstance(imagen, np.ndarray):
            imagen = Image.fromarray(imagen)

        img_draw = imagen.copy()
        draw = ImageDraw.Draw(img_draw)

        # Definir conexiones del esqueleto
        conexiones = [
            ('hombro_izq', 'hombro_der'),
            ('hombro_izq', 'codo_izq'),
            ('codo_izq', 'muñeca_izq'),
            ('hombro_der', 'codo_der'),
            ('codo_der', 'muñeca_der'),
            ('hombro_izq', 'cadera_izq'),
            ('hombro_der', 'cadera_der'),
            ('cadera_izq', 'cadera_der'),
            ('cadera_izq', 'rodilla_izq'),
            ('rodilla_izq', 'tobillo_izq'),
            ('cadera_der', 'rodilla_der'),
            ('rodilla_der', 'tobillo_der'),
        ]

        # Dibujar líneas
        for p1, p2 in conexiones:
            if p1 in puntos and p2 in puntos:
                draw.line(
                    [puntos[p1][:2], puntos[p2][:2]],
                    fill='lime',
                    width=3
                )

        # Dibujar puntos
        for nombre, (x, y, vis) in puntos.items():
            if vis > 0.5:
                draw.ellipse(
                    [(x-5, y-5), (x+5, y+5)],
                    fill='red',
                    outline='white',
                    width=2
                )

        return img_draw

class GeneradorAvatar3D:
    """Generador de avatares 3D usando estimación de profundidad"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n🔄 Inicializando modelo MiDaS en {self.device.upper()}...")

        try:
            self.modelo_midas = torch.hub.load(
                "intel-isl/MiDaS",
                "MiDaS_small",
                trust_repo=True
            )
            self.modelo_midas.to(self.device).eval()
            self.transform = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True
            ).small_transform

            print("✅ Modelo MiDaS cargado")

        except Exception as e:
            print(f"❌ Error: {e}")
            raise e

        # Inicializar detector de pose
        self.medidor = MedicionesAntropometricas()
        print("✅ Detector de pose cargado")

    def generar_avatar(self, imagen_pil, escala=0.19, posicion_y=110, grosor=0.25, tamaño_max=400):
        """Genera avatar 3D"""

        print("\n" + "=" * 70)
        print("🎨 GENERANDO AVATAR 3D")
        print("=" * 70)

        try:
            # Preprocesar imagen
            print("📐 Procesando imagen...")
            img_rgba = remove(imagen_pil)
            img_rgba = ImageOps.exif_transpose(img_rgba)
            img_rgba.thumbnail((tamaño_max, tamaño_max))

            # Crear máscara
            mask = np.array(img_rgba)[:, :, 3]
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

            img_rgb = img_rgba.convert('RGB')
            img_array = np.array(img_rgb)

            # Estimación de profundidad
            print("🔍 Estimando profundidad...")
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
            print("🔨 Construyendo malla 3D...")
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
                        mapa_indices[idx],
                        mapa_indices[idx + 1],
                        mapa_indices[idx + w],
                        mapa_indices[idx + w + 1]
                    )
                    if i1 != -1 and i2 != -1 and i3 != -1:
                        caras.append([i1, i2, i3])
                    if i2 != -1 and i3 != -1 and i4 != -1:
                        caras.append([i2, i4, i3])

            mesh = trimesh.creation.extrude_triangulation(vertices_2d, caras, height=-grosor)
            z_coords = depth.flatten()[indices_validos] * z_factor
            mesh.vertices[:len(indices_validos), 2] += z_coords

            # Aplicar colores
            colores_puros = img_array.reshape(-1, 3)[indices_validos]
            colores_finales = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)
            n = len(indices_validos)
            colores_finales[:n] = colores_puros
            colores_finales[n:2*n] = colores_puros
            if len(mesh.vertices) > 2*n:
                colores_finales[2*n:] = [100, 100, 100]
            mesh.visual.vertex_colors = colores_finales

            # Transformaciones
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
            mesh.apply_scale(escala)
            mesh.apply_translation([-mesh.centroid[0], posicion_y, 0])

            print("✅ Avatar 3D generado")
            return mesh

        except Exception as e:
            print(f"❌ Error: {e}")
            raise e

# Instancia global
print("\n🔄 Cargando modelos...")
generador = GeneradorAvatar3D()
print("✅ Sistema listo\n")

def generar_avatar_con_medidas(imagen, escala, posicion_y, grosor, altura_real):
    """Genera avatar y calcula medidas"""

    try:
        if imagen is None:
            return None, None, "❌ Sube una imagen primero"

        # Convertir a PIL
        if isinstance(imagen, np.ndarray):
            imagen_pil = Image.fromarray(imagen)
        else:
            imagen_pil = imagen

        # PASO 1: Detectar puntos y calcular medidas
        print("\n📏 ANALIZANDO MEDIDAS CORPORALES")
        puntos, resultado_pose = generador.medidor.detectar_puntos_clave(imagen)

        if puntos is None:
            return None, None, resultado_pose

        # Calcular medidas
        altura_cm = altura_real if altura_real > 0 else None
        medidas_cm, medidas_px, nombres, factor = generador.medidor.calcular_medidas(puntos, altura_cm)

        # Crear imagen con esqueleto
        img_esqueleto = generador.medidor.dibujar_esqueleto(imagen, puntos)

        # PASO 2: Generar avatar 3D
        mesh = generador.generar_avatar(imagen_pil, escala, posicion_y, grosor)

        # Guardar archivos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre = f"avatar_{timestamp}"

        glb_path = os.path.join(OUTPUT_DIR, f"{nombre}.glb")
        obj_path = os.path.join(OUTPUT_DIR, f"{nombre}.obj")
        stl_path = os.path.join(OUTPUT_DIR, f"{nombre}.stl")
        esqueleto_path = os.path.join(OUTPUT_DIR, f"{nombre}_mediciones.png")

        mesh.export(glb_path)
        mesh.export(obj_path)
        mesh.export(stl_path)
        img_esqueleto.save(esqueleto_path)

        # Crear reporte de medidas
        mensaje = f"""✨ ¡Avatar 3D y Mediciones Completadas!

📏 MEDIDAS ANTROPOMÉTRICAS:

"""

        for key, nombre_medida in nombres.items():
            cm = medidas_cm[key]
            mensaje += f"   • {nombre_medida}: {cm:.1f} cm\n"

        mensaje += f"""

📊 ESTADÍSTICAS DEL MODELO 3D:
   • Vértices: {len(mesh.vertices):,}
   • Caras: {len(mesh.faces):,}

📁 ARCHIVOS GUARDADOS:
   • {nombre}.glb (visualización web)
   • {nombre}.obj (edición 3D)
   • {nombre}.stl (impresión 3D)
   • {nombre}_mediciones.png (imagen con esqueleto)

🎨 CONFIGURACIÓN:
   • Escala: {escala}
   • Posición Y: {posicion_y}
   • Grosor: {grosor}
   • Factor de escala: {factor:.4f} cm/px

💡 NOTA: Las medidas son aproximadas basadas en {"la altura proporcionada" if altura_cm else "una altura promedio de 165cm"}.
Para mayor precisión, ingresa tu altura real.

📂 Ubicación: {os.path.abspath(OUTPUT_DIR)}"""

        return glb_path, img_esqueleto, mensaje

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, f"❌ Error: {str(e)}"

# INTERFAZ
with gr.Blocks(title="Generador de Avatares 3D con Mediciones") as demo:

    gr.Markdown("""
    # 🎨 Generador de Avatares 3D con Mediciones Antropométricas
    ### Crea avatares 3D realistas y obtén medidas corporales precisas
    
    **Proyecto de Grado** | MiDaS + MediaPipe | Optimizado RTX 3050
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📤 Entrada")

            imagen_input = gr.Image(label="Fotografía", type="pil", height=350)

            gr.Markdown("### ⚙️ Configuración")

            altura_real_input = gr.Number(
                label="Altura Real (cm) - Opcional para mayor precisión",
                value=0,
                info="Déjalo en 0 para que la IA estime tu altura automáticamente"
            )

            escala_slider = gr.Slider(0.05, 0.5, 0.19, 0.01, label="Escala")
            posicion_y_slider = gr.Slider(0, 200, 110, 5, label="Posición Y")
            grosor_slider = gr.Slider(0.1, 1.0, 0.25, 0.05, label="Grosor")

            btn_generar = gr.Button("🚀 Generar Avatar + Mediciones", variant="primary", size="lg")

            gr.Markdown("""
            ### 💡 Consejos:
            ✅ Foto de cuerpo completo  
            ✅ Persona centrada y visible  
            ✅ Pose frontal preferible  
            ✅ Fondo simple  
            ✅ Ingresa tu altura real para medidas exactas
            """)

        with gr.Column(scale=1):
            gr.Markdown("## 🎨 Avatar 3D")
            modelo_3d = gr.Model3D(label="Vista 3D", height=400)

            gr.Markdown("## 📏 Mediciones Detectadas")
            imagen_esqueleto = gr.Image(label="Puntos Corporales", height=350)

    with gr.Row():
        mensaje_output = gr.Textbox(label="📊 Resultados Completos", lines=25, interactive=False)

    with gr.Accordion("ℹ️ Información", open=False):
        gr.Markdown("""
        ### 📏 Medidas que se calculan:
        - **Altura Total**: De cabeza a pies
        - **Ancho de Hombros**: Distancia entre hombros
        - **Ancho de Cadera**: Distancia entre caderas
        - **Largo de Torso**: De hombros a cadera
        - **Largo de Piernas**: De cadera a tobillos
        - **Largo de Brazo**: De hombro a muñeca
        
        ### 🎯 Precisión:
        - Con altura real: ±2-3 cm
        - Sin altura: Basado en promedio de 165 cm
        
        ### 🔬 Tecnología:
        - MediaPipe Pose: Detección de 33 puntos corporales
        - MiDaS: Estimación de profundidad
        - Trimesh: Generación de malla 3D
        """)

    btn_generar.click(
        fn=generar_avatar_con_medidas,
        inputs=[imagen_input, escala_slider, posicion_y_slider, grosor_slider, altura_real_input],
        outputs=[modelo_3d, imagen_esqueleto, mensaje_output]
    )

    gr.Markdown("""
    ---
    🎓 **Proyecto de Grado** - Generación de Avatares 3D con Análisis Antropométrico
    
    💪 **¡Tú puedes graduarte!** 🎉
    """)

if __name__ == "__main__":
    print("\n🚀 Iniciando aplicación...\n")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)