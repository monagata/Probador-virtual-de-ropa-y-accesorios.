"""
PROBADOR VIRTUAL AR - VERSIÓN SIMPLE CON MODEL-VIEWER
Solución directa y simple - escanea QR y funciona inmediatamente

COMO model-viewer.dev - SIN COMPLICACIONES
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
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

OUTPUT_DIR = "avatares_generados"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# HTML SIMPLE con model-viewer (como el que viste)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Avatar AR</title>
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}

        .header {{
            background: rgba(0,0,0,0.3);
            color: white;
            padding: 15px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 20px;
            margin-bottom: 5px;
        }}

        .header p {{
            font-size: 13px;
            opacity: 0.9;
        }}

        model-viewer {{
            width: 100%;
            height: 100%;
            background-color: transparent;
        }}

        .ar-button {{
            background: #4CAF50;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>👔 Tu Avatar 3D</h1>
        <p>Toca el botón AR para verlo en tu espacio</p>
    </div>

    <model-viewer
        src="{MODEL_PATH}"
        alt="Avatar 3D"
        ar
        ar-modes="webxr scene-viewer quick-look"
        camera-controls
        touch-action="pan-y"
        shadow-intensity="1"
        auto-rotate
        camera-orbit="0deg 75deg 2.5m">

        <button slot="ar-button" class="ar-button">
            📱 Ver en AR
        </button>
    </model-viewer>
</body>
</html>
"""


class SimpleHTTPRequestHandlerWithCORS(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        super().end_headers()

    def do_GET(self):
        if self.path.startswith('/ar/'):
            filename = self.path.replace('/ar/', '')
            filepath = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        elif self.path.startswith('/model/'):
            filename = self.path.replace('/model/', '')
            filepath = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(filepath):
                self.send_response(200)
                self.send_header('Content-type', 'model/gltf-binary')
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        else:
            self.send_error(404)


def run_http_server(port=5000):
    os.chdir(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd())
    server = HTTPServer(('0.0.0.0', port), SimpleHTTPRequestHandlerWithCORS)
    print(f"✅ Servidor corriendo en puerto {port}")
    server.serve_forever()


class AnalizadorModa:
    @staticmethod
    def recomendar_talla(medidas, genero="hombre"):
        p = medidas['pecho']
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
        img = img_prenda.resize((50, 50))
        img_np = np.array(img.convert("RGB"))
        promedio = img_np.mean(axis=(0, 1))
        r, g, b = promedio

        if r > 200 and g > 200 and b > 200:
            return "Blanco", "Transmite pureza."
        if r < 60 and g < 60 and b < 60:
            return "Negro", "Transmite elegancia."
        if r > g and r > b:
            return "Rojo", "Transmite energía."
        if g > r and g > b:
            return "Verde", "Transmite armonía."
        if b > r and b > g:
            return "Azul", "Transmite confianza."
        return "Neutro", "Versátil."


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
            'cadera_izq': (int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                           int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h)),
            'nariz': (int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE.value].x * w),
                      int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE.value].y * h)),
            'tobillo_izq': (int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w),
                            int(res.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h))
        }

    def obtener_medidas(self, puntos, altura_cm):
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
        if img_prenda is None:
            return np.array(img_p.convert("RGB"))

        img_p_pil = img_p.convert("RGBA")
        prenda = remove(Image.fromarray(img_prenda) if isinstance(img_prenda, np.ndarray) else img_prenda)

        h_izq, h_der = puntos['hombro_izq'], puntos['hombro_der']
        c_izq = puntos['cadera_izq']

        ancho = int(abs(h_izq[0] - h_der[0]) * 2.2)
        alto = int(abs(h_izq[1] - c_izq[1]) * 1.35)
        prenda = prenda.resize((ancho, alto), Image.Resampling.LANCZOS)

        capa = Image.new("RGBA", img_p_pil.size, (0, 0, 0, 0))
        pos_x = ((h_izq[0] + h_der[0]) // 2) - (ancho // 2)
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

        cols = img_vestida.reshape(-1, 3)[indices]
        v_cols = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)
        v_cols[:len(indices)] = cols
        v_cols[len(indices):2 * len(indices)] = cols
        mesh.visual.vertex_colors = v_cols

        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
        mesh.apply_scale(0.19)
        mesh.apply_translation([-mesh.centroid[0], 110, 0])

        return mesh


# Inicialización
med = MedicionesAntropometricas()
gen = GeneradorAvatar3D()
ana = AnalizadorModa()

print("🔄 Iniciando servidor HTTP...")
http_thread = threading.Thread(target=run_http_server, args=(5000,), daemon=True)
http_thread.start()


def obtener_ngrok_url():
    """Intenta obtener la URL de ngrok si está corriendo"""
    try:
        import requests
        response = requests.get("http://localhost:4040/api/tunnels", timeout=2)
        tunnels = response.json()['tunnels']
        for tunnel in tunnels:
            if tunnel['proto'] == 'https':
                return tunnel['public_url']
    except:
        pass
    return None


def generar_qr_code(url):
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img_qr.save(buffer, format='PNG')
    buffer.seek(0)
    return Image.open(buffer)


def procesar(img_p, img_prenda, altura):
    if img_p is None:
        return None, None, "❌ Sube foto de la persona"
    if img_prenda is None:
        return None, None, "❌ Sube foto de la prenda"

    try:
        img_p_pil = Image.fromarray(img_p)
        puntos = med.detectar_puntos_clave(img_p_pil)

        if not puntos:
            return None, None, "❌ No se detectó el cuerpo"

        # Generar modelo 3D
        mesh = gen.generar(img_p_pil, img_prenda, puntos)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        glb_filename = f"avatar_{timestamp}.glb"
        glb_path = os.path.join(OUTPUT_DIR, glb_filename)
        mesh.export(glb_path)

        # Detectar URL (ngrok o local)
        ngrok_url = obtener_ngrok_url()

        if ngrok_url:
            base_url = ngrok_url
            model_url = f"{ngrok_url}/model/{glb_filename}"
        else:
            base_url = "http://192.168.40.42:5000"  # Cambia si tu IP es diferente
            model_url = f"http://192.168.40.42:5000/model/{glb_filename}"

        # Crear HTML
        ar_html_filename = f"ar_{timestamp}.html"
        ar_html_path = os.path.join(OUTPUT_DIR, ar_html_filename)

        html_content = HTML_TEMPLATE.replace('{MODEL_PATH}', model_url)

        with open(ar_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # URL del visor
        ar_url = f"{base_url}/ar/{ar_html_filename}"

        # Generar QR
        qr_image = generar_qr_code(ar_url)

        # Análisis
        medidas = med.obtener_medidas(puntos, altura)
        talla = ana.recomendar_talla(medidas)
        color_nom, color_desc = ana.analizar_color(Image.fromarray(img_prenda))

        # Mensaje
        url_type = "🔒 HTTPS (ngrok)" if ngrok_url else "⚠️ HTTP (requiere ngrok para AR)"

        resultado_html = f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; color: white;'>
            <h2 style='margin-top:0; text-align: center;'>✨ ¡Avatar Generado!</h2>

            <div style='background: rgba(255,255,255,0.15); padding: 20px; border-radius: 12px; margin: 15px 0;'>
                <h3 style='margin-top:0; text-align: center;'>📱 ESCANEA EL QR</h3>
                <p style='text-align: center; font-size: 16px; margin: 12px 0;'>
                    El código QR aparece a la derecha →
                </p>
                <div style='background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; margin: 12px 0;'>
                    <p style='font-size: 14px; margin: 8px 0;'>
                        <b>Pasos:</b><br>
                        1️⃣ Escanea el QR<br>
                        2️⃣ Se abre la página<br>
                        3️⃣ Toca "Ver en AR"<br>
                        4️⃣ ¡Coloca el avatar!
                    </p>
                </div>
                <p style='font-size: 13px; text-align: center; margin: 10px 0;'>
                    {url_type}
                </p>
            </div>

            <div style='background: rgba(255,255,255,0.15); padding: 15px; border-radius: 12px; margin: 15px 0;'>
                <h3 style='margin-top:0;'>📏 Talla: <span style='background: #FF6B6B; padding: 6px 18px; border-radius: 15px; font-size: 22px;'>{talla}</span></h3>
                <p>Pecho: {medidas['pecho']:.1f} cm</p>
            </div>

            <div style='background: rgba(255,255,255,0.15); padding: 15px; border-radius: 12px;'>
                <h3 style='margin-top:0;'>🎨 Color: {color_nom}</h3>
                <p style='font-size: 14px;'>"{color_desc}"</p>
            </div>

            <p style='font-size: 11px; text-align: center; margin-top: 15px; opacity: 0.8;'>
                URL: {ar_url}
            </p>
        </div>
        """

        return glb_path, qr_image, resultado_html

    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"


# Interfaz
with gr.Blocks(theme=gr.themes.Soft(), title="Probador AR") as demo:
    gr.Markdown("# 👔 Probador Virtual AR - Simple")
    gr.Markdown("### Escanea el QR y toca 'Ver en AR' - ¡Así de fácil!")

    gr.Markdown("""
    <div style='background: #FFE082; padding: 15px; border-radius: 8px; margin: 15px 0;'>
        <b>⚠️ IMPORTANTE:</b> Para que funcione la cámara, necesitas ngrok corriendo:<br>
        <code>ngrok http 5000</code>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            in_persona = gr.Image(label="📸 Foto Persona", type="numpy", height=250)
            in_prenda = gr.Image(label="👕 Foto Prenda", type="numpy", height=250)
            in_altura = gr.Number(label="📏 Altura (cm)", value=170)
            btn = gr.Button("🚀 Generar", variant="primary", size="lg")

        with gr.Column():
            out_modelo = gr.Model3D(label="Vista Previa", height=280)
            out_qr = gr.Image(label="📱 QR CODE", height=280)

    out_info = gr.HTML()

    btn.click(fn=procesar, inputs=[in_persona, in_prenda, in_altura], outputs=[out_modelo, out_qr, out_info])

if __name__ == "__main__":
    import time

    time.sleep(1)

    ngrok_url = obtener_ngrok_url()

    print("\n" + "=" * 70)
    print("  SISTEMA LISTO")
    print("=" * 70)
    if ngrok_url:
        print(f"\n✅ ngrok detectado: {ngrok_url}")
        print("📱 Los QR usarán HTTPS - La cámara funcionará")
    else:
        print("\n⚠️  ngrok NO detectado")
        print("📱 Inicia ngrok para que funcione la cámara AR:")
        print("   ngrok http 5000")
    print(f"\n💻 Interfaz: http://localhost:7860\n")

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)