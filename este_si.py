"""
GENERADOR DE AVATARES 3D - PROYECTO DE GRADO
Usando MiDaS para estimación de profundidad
Optimizado para RTX 3050 (4GB)
Autor: [Tu Nombre]
Enero 2026
"""

import os
import torch
import numpy as np
from PIL import Image, ImageOps
import trimesh
import cv2
from rembg import remove
import gradio as gr
from datetime import datetime

print("=" * 70)
print("  GENERADOR DE AVATARES 3D - PROYECTO DE GRADO")
print("=" * 70)
print(f"🐍 Python: {torch.__version__}")
print(f"💻 CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
print("=" * 70)

OUTPUT_DIR = "avatares_generados"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

            print("✅ Modelo MiDaS cargado correctamente")

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise e

    def generar_avatar(
            self,
            imagen_pil,
            escala=0.19,
            posicion_y=110,
            grosor=0.25,
            tamaño_max=400
    ):
        """
        Genera un avatar 3D desde una imagen

        Args:
            imagen_pil: Imagen PIL
            escala: Escala del modelo (default: 0.19)
            posicion_y: Posición vertical (default: 110)
            grosor: Grosor del modelo (default: 0.25)
            tamaño_max: Tamaño máximo de la imagen (default: 400)

        Returns:
            mesh: Objeto trimesh con el avatar 3D
        """

        print("\n" + "=" * 70)
        print("🎨 GENERANDO AVATAR 3D")
        print("=" * 70)

        try:
            # 1. PREPROCESAR IMAGEN
            print("📐 Procesando imagen...")

            # Remover fondo
            print("🔄 Removiendo fondo...")
            img_rgba = remove(imagen_pil)
            img_rgba = ImageOps.exif_transpose(img_rgba)

            # Redimensionar manteniendo aspecto
            img_rgba.thumbnail((tamaño_max, tamaño_max))
            print(f"   Tamaño final: {img_rgba.size}")

            # Crear máscara
            print("🔄 Creando máscara...")
            mask = np.array(img_rgba)[:, :, 3]
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            _, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)

            # Convertir a RGB
            img_rgb = img_rgba.convert('RGB')
            img_array = np.array(img_rgb)

            # 2. ESTIMACIÓN DE PROFUNDIDAD
            print("🔍 Estimando profundidad con MiDaS...")

            input_batch = self.transform(img_array).to(self.device)

            with torch.no_grad():
                prediction = self.modelo_midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_array.shape[:2],
                    mode="bicubic",
                    align_corners=False
                ).squeeze()

            # Normalizar profundidad
            depth = (prediction.cpu().numpy() - prediction.min().item()) / (
                    prediction.max().item() - prediction.min().item() + 1e-8
            )

            # Aplicar máscara
            mask_res = cv2.resize(mask, (depth.shape[1], depth.shape[0])) > 128
            depth[~mask_res] = 0

            print("✅ Profundidad estimada")

            # 3. CREAR MALLA 3D
            print("🔨 Construyendo malla 3D...")

            h, w = depth.shape
            y, x = np.mgrid[0:h:1, 0:w:1]
            z_factor = w * 0.12

            # Obtener vértices válidos
            indices_validos = np.where(mask_res.flatten())[0]
            vertices_2d = np.stack([x.flatten(), y.flatten()], axis=-1)[indices_validos]

            # Mapear índices
            mapa_indices = np.full(mask_res.flatten().shape, -1)
            mapa_indices[indices_validos] = np.arange(len(indices_validos))

            # Crear caras
            caras = []
            for i in range(h - 1):
                for j in range(w - 1):
                    idx = i * w + j
                    i1 = mapa_indices[idx]
                    i2 = mapa_indices[idx + 1]
                    i3 = mapa_indices[idx + w]
                    i4 = mapa_indices[idx + w + 1]

                    if i1 != -1 and i2 != -1 and i3 != -1:
                        caras.append([i1, i2, i3])
                    if i2 != -1 and i3 != -1 and i4 != -1:
                        caras.append([i2, i4, i3])

            print(f"   Vértices: {len(indices_validos):,}")
            print(f"   Caras: {len(caras):,}")

            # Extruir malla
            mesh = trimesh.creation.extrude_triangulation(
                vertices_2d,
                caras,
                height=-grosor
            )

            # Aplicar profundidad
            z_coords = depth.flatten()[indices_validos] * z_factor
            mesh.vertices[:len(indices_validos), 2] += z_coords

            # 4. APLICAR COLORES
            print("🎨 Aplicando texturas...")

            colores_puros = img_array.reshape(-1, 3)[indices_validos]
            colores_finales = np.zeros((len(mesh.vertices), 3), dtype=np.uint8)
            n = len(indices_validos)

            colores_finales[:n] = colores_puros
            colores_finales[n:2 * n] = colores_puros

            if len(mesh.vertices) > 2 * n:
                colores_finales[2 * n:] = [100, 100, 100]

            mesh.visual.vertex_colors = colores_finales

            # 5. TRANSFORMACIONES
            print("🔄 Aplicando transformaciones...")

            # Rotaciones
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            )
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
            )

            # Escala y posición
            mesh.apply_scale(escala)
            mesh.apply_translation([-mesh.centroid[0], posicion_y, 0])

            print("\n" + "=" * 70)
            print("✅ ¡AVATAR 3D GENERADO EXITOSAMENTE!")
            print("=" * 70)

            return mesh

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise e


# Instancia global del generador
print("\n🔄 Cargando modelo...")
generador = GeneradorAvatar3D()
print("✅ Sistema listo\n")


def generar_avatar_interfaz(imagen, escala, posicion_y, grosor):
    """Función para la interfaz de Gradio"""

    try:
        if imagen is None:
            return None, "❌ Por favor sube una imagen primero"

        # Convertir a PIL si es necesario
        if isinstance(imagen, np.ndarray):
            imagen = Image.fromarray(imagen)

        # Generar avatar
        mesh = generador.generar_avatar(
            imagen,
            escala=escala,
            posicion_y=posicion_y,
            grosor=grosor
        )

        # Guardar archivos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre = f"avatar_{timestamp}"

        glb_path = os.path.join(OUTPUT_DIR, f"{nombre}.glb")
        obj_path = os.path.join(OUTPUT_DIR, f"{nombre}.obj")
        stl_path = os.path.join(OUTPUT_DIR, f"{nombre}.stl")

        mesh.export(glb_path)
        mesh.export(obj_path)
        mesh.export(stl_path)

        mensaje = f"""✨ ¡Avatar 3D generado exitosamente!

📊 Estadísticas:
   • Vértices: {len(mesh.vertices):,}
   • Caras: {len(mesh.faces):,}

📁 Archivos guardados:
   • {nombre}.glb (web/visualización)
   • {nombre}.obj (edición en Blender)
   • {nombre}.stl (impresión 3D)

🎨 Configuración aplicada:
   • Escala: {escala}
   • Posición Y: {posicion_y}
   • Grosor: {grosor}

📂 Ubicación: {os.path.abspath(OUTPUT_DIR)}

🎮 Usa el mouse para rotar, zoom y explorar tu avatar"""

        return glb_path, mensaje

    except Exception as e:
        error_msg = f"""❌ Error generando avatar: {str(e)}

💡 Sugerencias:
• Verifica que la imagen sea clara
• Intenta con una imagen más pequeña
• Asegúrate de que el sujeto esté centrado"""

        return None, error_msg


# INTERFAZ GRADIO
with gr.Blocks(
        title="Generador de Avatares 3D - Proyecto de Grado",
) as demo:
    gr.Markdown("""
    # 🎨 Generador de Avatares 3D
    ### Convierte fotografías en avatares 3D realistas con estimación de profundidad

    **Proyecto de Grado** | Tecnología: MiDaS + Estimación de Profundidad | Optimizado para RTX 3050
    """)

    with gr.Row():
        # COLUMNA IZQUIERDA - INPUT
        with gr.Column(scale=1):
            gr.Markdown("## 📤 Imagen de Entrada")

            imagen_input = gr.Image(
                label="Sube una fotografía",
                type="pil",
                height=400
            )

            gr.Markdown("### ⚙️ Parámetros de Generación")

            escala_slider = gr.Slider(
                minimum=0.05,
                maximum=0.5,
                value=0.19,
                step=0.01,
                label="Escala del Avatar",
                info="Tamaño del modelo 3D"
            )

            posicion_y_slider = gr.Slider(
                minimum=0,
                maximum=200,
                value=110,
                step=5,
                label="Posición Vertical (Y)",
                info="Altura del avatar"
            )

            grosor_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.25,
                step=0.05,
                label="Grosor del Modelo",
                info="Profundidad de la malla"
            )

            btn_generar = gr.Button(
                "🚀 Generar Avatar 3D",
                variant="primary",
                size="lg"
            )

            gr.Markdown("""
            ### 💡 Consejos para mejores resultados:

            ✅ **Fotografía de buena calidad**  
            ✅ **Sujeto centrado y visible**  
            ✅ **Fondo simple o uniforme**  
            ✅ **Buena iluminación**  
            ✅ **Pose frontal o semi-frontal**

            ⏱️ **Tiempo de generación:** 30-60 segundos
            """)

        # COLUMNA DERECHA - OUTPUT
        with gr.Column(scale=1):
            gr.Markdown("## 🎨 Avatar 3D Generado")

            modelo_3d_viewer = gr.Model3D(
                label="Vista Previa 3D Interactiva",
                height=550,
                clear_color=[0.15, 0.15, 0.2, 1.0]
            )

            mensaje_output = gr.Textbox(
                label="📊 Información y Estado",
                lines=20,
                interactive=False
            )

    # SECCIÓN INFORMATIVA
    with gr.Accordion("ℹ️ Información Técnica", open=False):
        gpu_info = f"✅ {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "❌ CPU"

        gr.Markdown(f"""
        ### 🖥️ Sistema:
        - **GPU**: {gpu_info}
        - **PyTorch**: {torch.__version__}
        - **Modelo**: MiDaS (Intel ISL)
        - **Optimización**: RTX 3050 (4GB)

        ### 🔬 Tecnología:
        Este generador utiliza:
        1. **Remoción de fondo** (rembg)
        2. **Estimación de profundidad** (MiDaS)
        3. **Construcción de malla 3D** (Trimesh)
        4. **Texturizado automático** (color de imagen)

        ### 🎮 Controles del Visor:
        - **Rotar**: Click izquierdo + arrastrar
        - **Zoom**: Scroll del mouse
        - **Pan**: Click derecho + arrastrar

        ### 📥 Formatos de Salida:
        - **GLB**: Visualización web (Three.js, Babylon.js)
        - **OBJ**: Edición 3D (Blender, Maya, 3DS Max)
        - **STL**: Impresión 3D

        ### ⚙️ Parámetros Ajustables:
        - **Escala**: Controla el tamaño del avatar
        - **Posición Y**: Ajusta la altura vertical
        - **Grosor**: Profundidad de la extrusión 3D

        ### 🎯 Ventajas del Método:
        - ✅ Rápido (30-60 segundos)
        - ✅ Ligero (funciona en 4GB VRAM)
        - ✅ Realista (usa profundidad estimada)
        - ✅ Colorido (texturas de la imagen original)
        """)

    # Conectar botón con función
    btn_generar.click(
        fn=generar_avatar_interfaz,
        inputs=[imagen_input, escala_slider, posicion_y_slider, grosor_slider],
        outputs=[modelo_3d_viewer, mensaje_output]
    )

    gr.Markdown("""
    ---
    🎓 **Proyecto de Grado** - Generación de Avatares 3D mediante Estimación de Profundidad

    🚀 **Tecnologías**: Python • PyTorch • MiDaS • Trimesh • Gradio • CUDA

    💪 **¡Tú puedes graduarte!** 🎉
    """)

# LANZAR APLICACIÓN
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 INICIANDO APLICACIÓN WEB")
    print("=" * 70)
    print("🌐 Abriendo navegador...")
    print("🔗 URL: http://127.0.0.1:7860")
    print("\n💡 Para detener: presiona Ctrl+C")
    print("=" * 70 + "\n")

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )