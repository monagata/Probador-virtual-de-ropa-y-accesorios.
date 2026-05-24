"""
DIAGNÓSTICO: Ver por qué la prenda no se aplica al avatar
Incluye prints de debug y guardado de imágenes intermedias
"""

import os
import torch
import numpy as np
from PIL import Image
import trimesh
import cv2
from rembg import remove
import gradio as gr
from datetime import datetime
import mediapipe as mp
import qrcode
import io

OUTPUT_DIR = "avatares_generados"
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


class MedicionesAntropometricas:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

    def detectar_puntos_clave(self, imagen):
        img_np = np.array(imagen)
        res = self.pose.process(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None

        h, w = img_np.shape[:2]
        return {
            'hombro_izq': (int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                           int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)),
            'hombro_der': (int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                           int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)),
            'codo_izq': (int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w),
                         int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h)),
            'codo_der': (int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w),
                         int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h)),
            'cadera_izq': (int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                           int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h)),
            'cadera_der': (int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                           int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)),
            'nariz': (int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE.value].x * w),
                      int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE.value].y * h)),
            'tobillo_izq': (int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w),
                            int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h))
        }


class GeneradorAvatar3D:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🔧 Usando dispositivo: {self.device}")
        self.modelo_midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform

    def _aplicar_prenda_SIMPLE(self, img_p, img_prenda, puntos, timestamp):
        """Versión simple y directa para debugging"""

        print("\n" + "=" * 60)
        print("🎨 APLICANDO PRENDA - MODO DEBUG")
        print("=" * 60)

        # Convertir persona a RGB
        if isinstance(img_p, Image.Image):
            img_persona = np.array(img_p.convert("RGB"))
        else:
            img_persona = np.array(Image.fromarray(img_p).convert("RGB"))

        h_p, w_p = img_persona.shape[:2]
        print(f"  📐 Dimensiones persona: {w_p}x{h_p}")

        # Guardar imagen original
        Image.fromarray(img_persona).save(os.path.join(DEBUG_DIR, f"{timestamp}_1_persona_original.png"))
        print(f"  ✓ Guardado: {timestamp}_1_persona_original.png")

        # Procesar prenda
        if img_prenda is None:
            print("  ⚠️ No hay prenda - retornando persona original")
            return img_persona

        print(f"  📦 Procesando prenda...")

        # Convertir prenda
        if isinstance(img_prenda, np.ndarray):
            prenda_pil = Image.fromarray(img_prenda)
        else:
            prenda_pil = img_prenda

        # Guardar prenda original
        prenda_pil.save(os.path.join(DEBUG_DIR, f"{timestamp}_2_prenda_original.png"))
        print(f"  ✓ Guardado: {timestamp}_2_prenda_original.png")

        # Remover fondo de prenda
        print("  🔄 Removiendo fondo de prenda...")
        prenda_sin_fondo = remove(prenda_pil)
        prenda_sin_fondo.save(os.path.join(DEBUG_DIR, f"{timestamp}_3_prenda_sin_fondo.png"))
        print(f"  ✓ Guardado: {timestamp}_3_prenda_sin_fondo.png")

        prenda_np = np.array(prenda_sin_fondo)
        print(f"  📐 Dimensiones prenda: {prenda_np.shape}")

        if prenda_np.shape[2] != 4:
            print("  ⚠️ Prenda no tiene canal alpha - agregando...")
            prenda_rgb = prenda_np[:, :, :3]
            alpha = np.ones((prenda_np.shape[0], prenda_np.shape[1]), dtype=np.uint8) * 255
            prenda_np = np.dstack([prenda_rgb, alpha])

        # Obtener puntos clave
        h_izq = puntos['hombro_izq']
        h_der = puntos['hombro_der']
        c_izq = puntos['cadera_izq']
        c_der = puntos['cadera_der']

        print(f"\n  📍 Puntos clave:")
        print(f"     Hombro izq: {h_izq}")
        print(f"     Hombro der: {h_der}")
        print(f"     Cadera izq: {c_izq}")
        print(f"     Cadera der: {c_der}")

        # Calcular dimensiones del torso
        ancho_hombros = int(abs(h_der[0] - h_izq[0]))
        ancho_cadera = int(abs(c_der[0] - c_izq[0]))
        alto_torso = int(abs(c_izq[1] - h_izq[1]))

        print(f"\n  📏 Dimensiones torso:")
        print(f"     Ancho hombros: {ancho_hombros}px")
        print(f"     Ancho cadera: {ancho_cadera}px")
        print(f"     Alto torso: {alto_torso}px")

        # Redimensionar prenda generosamente
        nuevo_ancho = int(ancho_hombros * 2.5)  # MÁS ANCHO
        nuevo_alto = int(alto_torso * 1.5)  # MÁS ALTO

        print(f"\n  🔧 Redimensionando prenda a: {nuevo_ancho}x{nuevo_alto}")

        prenda_resized = cv2.resize(prenda_np, (nuevo_ancho, nuevo_alto),
                                    interpolation=cv2.INTER_LANCZOS4)

        # Guardar prenda redimensionada
        Image.fromarray(prenda_resized).save(os.path.join(DEBUG_DIR, f"{timestamp}_4_prenda_resized.png"))
        print(f"  ✓ Guardado: {timestamp}_4_prenda_resized.png")

        # Crear canvas
        resultado = img_persona.copy()

        # Calcular posición
        centro_x = (h_izq[0] + h_der[0]) // 2
        pos_x = centro_x - (nuevo_ancho // 2)
        pos_y = h_izq[1] - int(nuevo_alto * 0.25)  # Un poco arriba de los hombros

        print(f"\n  📍 Posición de pegado:")
        print(f"     X: {pos_x}")
        print(f"     Y: {pos_y}")

        # Asegurar que no se sale de los límites
        pos_x = max(0, min(pos_x, w_p - nuevo_ancho))
        pos_y = max(0, min(pos_y, h_p - nuevo_alto))

        print(f"     X ajustado: {pos_x}")
        print(f"     Y ajustado: {pos_y}")

        # Extraer región donde se va a pegar
        x1, y1 = pos_x, pos_y
        x2, y2 = min(pos_x + nuevo_ancho, w_p), min(pos_y + nuevo_alto, h_p)

        region_ancho = x2 - x1
        region_alto = y2 - y1

        print(f"\n  📦 Región de pegado: {region_ancho}x{region_alto}")

        # Recortar prenda si es necesario
        prenda_a_pegar = prenda_resized[:region_alto, :region_ancho]

        # Aplicar alpha blending
        if prenda_a_pegar.shape[2] == 4:
            alpha = prenda_a_pegar[:, :, 3].astype(float) / 255.0
            alpha = cv2.GaussianBlur(alpha, (7, 7), 0)  # Suavizar bordes
            alpha = alpha[:, :, np.newaxis]

            prenda_rgb = prenda_a_pegar[:, :, :3].astype(float)
            fondo_rgb = resultado[y1:y2, x1:x2].astype(float)

            # Mezclar
            mezcla = (prenda_rgb * alpha + fondo_rgb * (1 - alpha)).astype(np.uint8)
            resultado[y1:y2, x1:x2] = mezcla

            print(f"  ✓ Alpha blending aplicado")
        else:
            resultado[y1:y2, x1:x2] = prenda_a_pegar
            print(f"  ✓ Pegado directo (sin alpha)")

        # Guardar resultado
        Image.fromarray(resultado).save(os.path.join(DEBUG_DIR, f"{timestamp}_5_resultado_final.png"))
        print(f"  ✓ Guardado: {timestamp}_5_resultado_final.png")

        print("\n" + "=" * 60)
        print("✅ PRENDA APLICADA - Revisa la carpeta debug/")
        print("=" * 60 + "\n")

        return resultado

    def generar(self, img_p, img_prenda, puntos, timestamp):
        print("\n🔄 Generando modelo 3D...")

        # Aplicar prenda PRIMERO
        img_vestida = self._aplicar_prenda_SIMPLE(img_p, img_prenda, puntos, timestamp)

        # Guardar para verificar
        Image.fromarray(img_vestida).save(os.path.join(DEBUG_DIR, f"{timestamp}_6_antes_de_3d.png"))

        # Eliminar fondo
        print("🔄 Eliminando fondo de imagen vestida...")
        img_vestida_pil = Image.fromarray(img_vestida)
        img_limpia = remove(img_vestida_pil)

        # Guardar imagen sin fondo
        img_limpia.save(os.path.join(DEBUG_DIR, f"{timestamp}_7_sin_fondo.png"))

        mask = cv2.threshold(
            cv2.GaussianBlur(np.array(img_limpia)[:, :, 3], (7, 7), 0),
            150, 255, cv2.THRESH_BINARY
        )[1]

        # Convertir de vuelta a RGB para MiDaS
        img_vestida_rgb = np.array(img_limpia.convert("RGB"))

        print("🔄 Calculando profundidad con MiDaS...")
        input_b = self.transform(img_vestida_rgb).to(self.device)

        with torch.no_grad():
            depth = self.modelo_midas(input_b)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=img_vestida_rgb.shape[:2],
                mode="bicubic"
            ).squeeze().cpu().numpy()

        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        mask_res = cv2.resize(mask, (depth.shape[1], depth.shape[0])) > 128
        depth[~mask_res] = 0

        print("🔄 Creando malla 3D...")
        h, w = depth.shape
        indices = np.where(mask_res.flatten())[0]
        vert_2d = np.stack([
            np.mgrid[0:h, 0:w][1].flatten(),
            np.mgrid[0:h, 0:w][0].flatten()
        ], axis=-1)[indices]

        mapa = np.full(mask_res.flatten().shape, -1)
        mapa[indices] = np.arange(len(indices))

        caras = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j
                i1, i2, i3, i4 = mapa[idx], mapa[idx + 1], mapa[idx + w], mapa[idx + w + 1]
                if i1 != -1 and i2 != -1 and i3 != -1:
                    caras.append([i1, i2, i3])
                if i2 != -1 and i4 != -1 and i3 != -1:
                    caras.append([i2, i4, i3])

        mesh = trimesh.creation.extrude_triangulation(vert_2d, caras, height=-0.25)
        mesh.vertices[:len(indices), 2] += depth.flatten()[indices] * (w * 0.12)

        # IMPORTANTE: Usar los colores de la imagen VESTIDA
        cols = img_vestida_rgb.reshape(-1, 3)[indices]
        v_cols = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)
        v_cols[:len(indices)] = cols
        v_cols[len(indices):2 * len(indices)] = cols
        mesh.visual.vertex_colors = v_cols

        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
        mesh.apply_scale(0.19)
        mesh.apply_translation([-mesh.centroid[0], 110, 0])

        print("✅ Modelo 3D generado\n")
        return mesh


# Inicialización
med = MedicionesAntropometricas()
gen = GeneradorAvatar3D()


def procesar(img_p, img_prenda, altura):
    if img_p is None:
        return None, "❌ Sube foto de la persona"
    if img_prenda is None:
        return None, "❌ Sube foto de la prenda"

    try:
        img_p_pil = Image.fromarray(img_p)
        puntos = med.detectar_puntos_clave(img_p_pil)

        if not puntos:
            return None, "❌ No se detectó el cuerpo"

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        print("\n" + "🎯" * 30)
        print(f"TIMESTAMP: {timestamp}")
        print("🎯" * 30 + "\n")

        # Generar modelo 3D con debugging
        mesh = gen.generar(img_p_pil, img_prenda, puntos, timestamp)

        glb_filename = f"avatar_{timestamp}.glb"
        glb_path = os.path.join(OUTPUT_DIR, glb_filename)
        mesh.export(glb_path)

        resultado_html = f"""
        <div style='background: #4CAF50; padding: 20px; border-radius: 10px; color: white;'>
            <h2>✅ Avatar Generado con Prenda</h2>
            <p style='font-size: 16px; margin: 15px 0;'>
                <b>📁 Revisa la carpeta debug/</b><br>
                Encontrarás todas las imágenes intermedias del proceso.
            </p>
            <p style='font-size: 14px; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;'>
                <b>Archivos generados:</b><br>
                • {timestamp}_1_persona_original.png<br>
                • {timestamp}_2_prenda_original.png<br>
                • {timestamp}_3_prenda_sin_fondo.png<br>
                • {timestamp}_4_prenda_resized.png<br>
                • {timestamp}_5_resultado_final.png ← <b>IMAGEN CON PRENDA</b><br>
                • {timestamp}_6_antes_de_3d.png<br>
                • {timestamp}_7_sin_fondo.png
            </p>
            <p style='font-size: 13px; margin-top: 15px;'>
                🔍 Si la prenda no aparece en el 3D, revisa la imagen #5 y #6<br>
                Esas muestran si la prenda se aplicó correctamente antes del 3D
            </p>
        </div>
        """

        return glb_path, resultado_html

    except Exception as e:
        import traceback
        error_completo = traceback.format_exc()
        return None, f"❌ Error:\n\n{error_completo}"


# Interfaz
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 MODO DEBUG - Ver Por Qué la Prenda No Aparece")
    gr.Markdown("### Este modo guarda todas las imágenes intermedias")

    with gr.Row():
        with gr.Column():
            in_persona = gr.Image(label="Foto Persona", type="numpy")
            in_prenda = gr.Image(label="Foto Prenda", type="numpy")
            in_altura = gr.Number(label="Altura (cm)", value=170)
            btn = gr.Button("🔍 Generar (Debug)", variant="primary")

        with gr.Column():
            out_modelo = gr.Model3D(label="Modelo 3D")

    out_info = gr.HTML()

    btn.click(fn=procesar, inputs=[in_persona, in_prenda, in_altura], outputs=[out_modelo, out_info])

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  MODO DEBUG ACTIVADO")
    print("=" * 70)
    print(f"\n📁 Las imágenes se guardarán en: {DEBUG_DIR}")
    print("🔍 Podrás ver exactamente qué pasa en cada paso\n")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)