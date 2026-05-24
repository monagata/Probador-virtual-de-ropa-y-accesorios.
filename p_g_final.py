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

# --- CONFIGURACIÓN ---
OUTPUT_DIR = "avatares_generados"
# CORRECCIÓN AQUÍ: Agregado exist_ok=True para evitar el FileExistsError
os.makedirs(OUTPUT_DIR, exist_ok=True)


class AnalizadorModa:
    @staticmethod
    def recomendar_talla(medidas, genero="hombre"):
        p = medidas['pecho']
        # Lógica de tallaje internacional basada en cm de pecho
        if genero == "hombre":
            if p < 91: return "S"
            if p < 98: return "M"
            if p < 106: return "L"
            return "XL"
        else:
            if p < 88: return "S"
            if p < 94: return "M"
            if p < 102: return "L"
            return "XL"

    @staticmethod
    def analizar_color(img_prenda):
        # Extraer color predominante de forma rápida
        img = img_prenda.resize((50, 50))
        img_np = np.array(img.convert("RGB"))
        promedio = img_np.mean(axis=(0, 1))

        r, g, b = promedio
        if r > 200 and g > 200 and b > 200:
            return "Blanco", "Transmite pureza y frescura. Combina con todo, ideal con azul denim o tonos tierra."
        if r < 60 and g < 60 and b < 60:
            return "Negro", "Transmite elegancia y autoridad. Combina excelente con grises o colores vibrantes."
        if r > g and r > b:
            return "Rojo / Cálido", "Transmite energía y pasión. Combina bien con negro, blanco o azul marino."
        if g > r and g > b:
            return "Verde", "Transmite armonía y naturaleza. Combina con beige, blanco o chocolate."
        if b > r and b > g:
            return "Azul", "Transmite confianza y serenidad. Perfecto con blanco, gris claro o camel."
        return "Tono Neutro", "Es una opción versátil. Combina con accesorios llamativos o tonos pasteles."


class MedicionesAntropometricas:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

    def detectar_puntos_clave(self, imagen):
        img_np = np.array(imagen)
        res = self.pose.process(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks: return None
        h, w = img_np.shape[:2]
        return {n: (int(res.pose_landmarks.landmark[id.value].x * w), int(res.pose_landmarks.landmark[id.value].y * h))
                for n, id in {'hombro_izq': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                              'hombro_der': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                              'cadera_izq': self.mp_pose.PoseLandmark.LEFT_HIP,
                              'nariz': self.mp_pose.PoseLandmark.NOSE,
                              'tobillo_izq': self.mp_pose.PoseLandmark.LEFT_ANKLE}.items()}

    def obtener_medidas(self, puntos, altura_cm):
        # Cálculo de escala píxel a centímetro
        px_h = np.linalg.norm(np.array(puntos['nariz']) - np.array(puntos['tobillo_izq']))
        escala = (altura_cm if altura_cm > 0 else 170) / px_h
        pecho = np.linalg.norm(np.array(puntos['hombro_izq']) - np.array(puntos['hombro_der'])) * 2.2 * escala
        return {'pecho': pecho}


class GeneradorAvatar3D:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.modelo_midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(self.device).eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform

    def _aplicar_textura(self, img_p, img_prenda, puntos):
        if img_prenda is None: return np.array(img_p.convert("RGB"))
        img_p_pil = img_p.convert("RGBA")
        prenda = remove(Image.fromarray(img_prenda) if isinstance(img_prenda, np.ndarray) else img_prenda)

        h_izq, h_der = puntos['hombro_izq'], puntos['hombro_der']
        c_izq = puntos['cadera_izq']

        ancho = int(abs(h_izq[0] - h_der[0]) * 2.2)
        alto = int(abs(h_izq[1] - c_izq[1]) * 1.35)
        prenda = prenda.resize((ancho, alto), Image.Resampling.LANCZOS)

        capa = Image.new("RGBA", img_p_pil.size, (0, 0, 0, 0))
        pos_x = ((h_izq[0] + h_der[0]) // 2) - (ancho // 2)
        # AJUSTE DE HOMBROS (Subido un 22% del alto de la prenda)
        pos_y = h_izq[1] - int(alto * 0.22)

        capa.paste(prenda, (pos_x, pos_y), prenda)
        return np.array(Image.alpha_composite(img_p_pil, capa).convert("RGB"))

    def generar(self, img_p, img_prenda, puntos):
        img_limpia = remove(img_p)
        mask = cv2.threshold(cv2.GaussianBlur(np.array(img_limpia)[:, :, 3], (7, 7), 0), 150, 255, cv2.THRESH_BINARY)[1]
        img_vestida = self._aplicar_textura(img_p, img_prenda, puntos)

        input_b = self.transform(img_vestida).to(self.device)
        with torch.no_grad():
            depth = self.modelo_midas(input_b)
            depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=img_vestida.shape[:2],
                                                    mode="bicubic").squeeze().cpu().numpy()

        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        mask_res = cv2.resize(mask, (depth.shape[1], depth.shape[0])) > 128
        depth[~mask_res] = 0

        h, w = depth.shape
        indices = np.where(mask_res.flatten())[0]
        vert_2d = np.stack([np.mgrid[0:h, 0:w][1].flatten(), np.mgrid[0:h, 0:w][0].flatten()], axis=-1)[indices]
        mapa = np.full(mask_res.flatten().shape, -1);
        mapa[indices] = np.arange(len(indices))

        caras = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j;
                i1, i2, i3, i4 = mapa[idx], mapa[idx + 1], mapa[idx + w], mapa[idx + w + 1]
                if i1 != -1 and i2 != -1 and i3 != -1: caras.append([i1, i2, i3])
                if i2 != -1 and i4 != -1 and i3 != -1: caras.append([i2, i4, i3])

        # Grosor 0.25 y Escala 0.19 aplicados
        mesh = trimesh.creation.extrude_triangulation(vert_2d, caras, height=-0.25)
        mesh.vertices[:len(indices), 2] += depth.flatten()[indices] * (w * 0.12)
        cols = img_vestida.reshape(-1, 3)[indices]
        v_cols = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)
        v_cols[:len(indices)] = cols;
        v_cols[len(indices):2 * len(indices)] = cols
        mesh.visual.vertex_colors = v_cols

        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
        mesh.apply_scale(0.19)
        mesh.apply_translation([-mesh.centroid[0], 110, 0])
        return mesh


# --- INICIALIZACIÓN ---
med = MedicionesAntropometricas()
gen = GeneradorAvatar3D()
ana = AnalizadorModa()


def procesar(img_p, img_prenda, altura):
    if img_p is None: return None, "Por favor sube una imagen de la persona."
    if img_prenda is None: return None, "Por favor sube una imagen de la prenda."

    img_p_pil = Image.fromarray(img_p)
    puntos = med.detectar_puntos_clave(img_p_pil)
    if not puntos: return None, "No se detectó el cuerpo. Asegúrate de que la persona sea visible de pies a cabeza."

    # Generar Modelo 3D
    mesh = gen.generar(img_p_pil, img_prenda, puntos)
    path = os.path.join(OUTPUT_DIR, f"avatar_{datetime.now().strftime('%H%M%S')}.glb")
    mesh.export(path)

    # Análisis de Talla y Psicología de Color
    medidas = med.obtener_medidas(puntos, altura)
    talla = ana.recomendar_talla(medidas)
    color_nom, color_desc = ana.analizar_color(Image.fromarray(img_prenda))

    res_html = f"""
    <div style='background: #f0f4f8; padding: 20px; border-radius: 12px; border-left: 6px solid #2196f3; font-family: sans-serif;'>
        <h3 style='margin-top:0; color:#1565c0;'>📋 Informe de Estilo Personalizado</h3>
        <p style='font-size:1.1em;'><b>Talla Recomendada:</b> <span style='background:#e91e63; color:white; padding:2px 8px; border-radius:4px;'>{talla}</span></p>
        <p><b>Medida estimada de pecho:</b> {medidas['pecho']:.1f} cm</p>
        <hr style='border: 0; border-top: 1px solid #ccc;'>
        <p><b>Análisis de Color:</b> <span style='color:#1565c0; font-weight:bold;'>{color_nom}</span></p>
        <p style='font-style: italic; color: #444; line-height:1.4;'>"{color_desc}"</p>
    </div>
    """
    return path, res_html


# --- INTERFAZ GRADIO ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👔 Probador Virtual y Asesor de Imagen 3D")
    gr.Markdown("Sube tu foto y la de una prenda para ver cómo te queda y recibir consejos de talla y color.")

    with gr.Row():
        with gr.Column():
            in_p = gr.Image(label="Tu Foto (Cuerpo completo)", type="numpy")
            in_prenda = gr.Image(label="Foto de la Prenda", type="numpy")
            in_h = gr.Number(label="Ingresa tu Altura (cm)", value=170)
            btn = gr.Button("🚀 Generar Mi Avatar", variant="primary")
        with gr.Column():
            out_3d = gr.Model3D(label="Visualización 3D")
            out_html = gr.HTML()

    btn.click(procesar, [in_p, in_prenda, in_h], [out_3d, out_html])

if __name__ == "__main__":
    demo.launch()