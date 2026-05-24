"""
Script de Diagnóstico y Prueba
Verifica que todo esté correctamente instalado
"""

import sys


def verificar_python():
    """Verifica versión de Python"""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor == 11:
        print("  ✓ Versión correcta de Python")
        return True
    else:
        print("  ⚠ Se recomienda Python 3.11.x")
        return True


def verificar_torch():
    """Verifica PyTorch y CUDA"""
    try:
        import torch
        print(f"\n✓ PyTorch {torch.__version__}")

        if torch.cuda.is_available():
            print(f"  ✓ CUDA disponible")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")

            # Información detallada
            props = torch.cuda.get_device_properties(0)
            vram_total = props.total_memory / (1024 ** 3)
            print(f"  ✓ VRAM Total: {vram_total:.2f} GB")

            # Prueba de memoria disponible
            torch.cuda.empty_cache()
            vram_libre = torch.cuda.memory_reserved(0) / (1024 ** 3)
            print(f"  ✓ VRAM Libre: {vram_total - vram_libre:.2f} GB")

            return True
        else:
            print("  ❌ CUDA NO disponible")
            print("  ⚠ El código funcionará en CPU (más lento)")
            return False

    except ImportError:
        print("\n❌ PyTorch no está instalado")
        print(
            "   Ejecuta: pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118")
        return False
    except Exception as e:
        print(f"\n⚠ Error al verificar PyTorch: {e}")
        return False


def verificar_librerias():
    """Verifica librerías necesarias"""
    librerias = {
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'trimesh': 'trimesh',
        'networkx': 'networkx'
    }

    print("\n📦 Verificando librerías...")
    todas_ok = True

    for lib, pip_name in librerias.items():
        try:
            __import__(lib)
            print(f"  ✓ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name} - Ejecuta: pip install {pip_name}")
            todas_ok = False

    return todas_ok


def verificar_archivos():
    """Verifica que los scripts principales existan"""
    import os

    archivos = [
        'foto_2d_a_3d.py',
        'foto_2d_3d_lite.py',
        'requirements.txt'
    ]

    print("\n📁 Verificando archivos del proyecto...")
    todos_ok = True

    for archivo in archivos:
        if os.path.exists(archivo):
            print(f"  ✓ {archivo}")
        else:
            print(f"  ⚠ {archivo} no encontrado")
            todos_ok = False

    return todos_ok


def prueba_conversion_simple():
    """Prueba rápida de conversión"""
    print("\n🧪 Ejecutando prueba rápida...")

    try:
        import numpy as np
        from PIL import Image
        import trimesh

        # Crear imagen de prueba
        print("  Creando imagen de prueba...")
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Simular mapa de profundidad
        print("  Generando mapa de profundidad...")
        depth = np.random.rand(100, 100)

        # Crear malla simple
        print("  Creando malla 3D...")
        x, y = np.meshgrid(range(100), range(100))
        vertices = np.stack([x.flatten(), y.flatten(), (depth * 10).flatten()], axis=-1)

        # Crear algunas caras
        faces = []
        for i in range(99):
            for j in range(99):
                idx = i * 100 + j
                faces.append([idx, idx + 1, idx + 100])

        mesh = trimesh.Trimesh(vertices=vertices[:1000], faces=faces[:100])

        print("  ✓ Prueba exitosa - El sistema puede crear mallas 3D")
        return True

    except Exception as e:
        print(f"  ❌ Error en prueba: {e}")
        return False


def mostrar_recomendaciones(cuda_ok, libs_ok):
    """Muestra recomendaciones según el estado del sistema"""
    print("\n" + "=" * 60)
    print("📋 RECOMENDACIONES")
    print("=" * 60)

    if cuda_ok and libs_ok:
        print("\n✨ ¡Sistema listo para usar!")
        print("\nPuedes ejecutar:")
        print("  python foto_2d_3d_lite.py  (Recomendado para RTX 3050)")
        print("\nO usar el script completo:")
        print("  python foto_2d_a_3d.py")

    elif not cuda_ok and libs_ok:
        print("\n⚠ CUDA no disponible")
        print("\nPasos para solucionar:")
        print("1. Verifica que tu driver NVIDIA esté actualizado")
        print("   Descarga: https://www.nvidia.com/Download/index.aspx")
        print("2. Reinstala PyTorch con CUDA:")
        print("   pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118")
        print("3. Reinicia PyCharm")
        print("\nMientras tanto, el código funcionará en CPU (más lento)")

    elif not libs_ok:
        print("\n⚠ Faltan librerías")
        print("\nEjecuta:")
        print("  pip install -r requirements.txt")
        print("\nO instala manualmente las librerías faltantes")

    print("\n📚 Consulta el README.md para más información")
    print("=" * 60)


def main():
    """Función principal de diagnóstico"""
    print("=" * 60)
    print("DIAGNÓSTICO DEL SISTEMA - Conversión 2D a 3D")
    print("=" * 60)

    # Verificaciones
    python_ok = verificar_python()
    cuda_ok = verificar_torch()
    libs_ok = verificar_librerias()
    archivos_ok = verificar_archivos()

    # Prueba de conversión
    if libs_ok:
        prueba_ok = prueba_conversion_simple()
    else:
        prueba_ok = False

    # Resumen
    print("\n" + "=" * 60)
    print("📊 RESUMEN")
    print("=" * 60)
    print(f"Python:              {'✓' if python_ok else '❌'}")
    print(f"PyTorch + CUDA:      {'✓' if cuda_ok else '❌'}")
    print(f"Librerías:           {'✓' if libs_ok else '❌'}")
    print(f"Archivos proyecto:   {'✓' if archivos_ok else '⚠'}")
    print(f"Prueba funcional:    {'✓' if prueba_ok else '❌'}")

    # Recomendaciones
    mostrar_recomendaciones(cuda_ok, libs_ok)

    return cuda_ok and libs_ok and prueba_ok


if __name__ == "__main__":
    exito = main()
    sys.exit(0 if exito else 1)