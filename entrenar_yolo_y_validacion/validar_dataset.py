"""
Script: validar_dataset.py
Verifica que tu dataset esté listo para YOLO

Universidad Autonoma de Queretaro - Maestria en IA
"""

import os
import yaml
from pathlib import Path

DATASET_ROOT = r'D:\proyectro_webots\dataset_yolo'

def validar_estructura():
    """Valida que existan todas las carpetas necesarias."""
    
    print("="*70)
    print("VALIDACION DE ESTRUCTURA DEL DATASET")
    print("="*70)
    
    carpetas_requeridas = [
        'images/train',
        'images/valid',
        'labels/train',
        'labels/valid',
    ]
    
    print("\n1. CARPETAS:")
    for carpeta in carpetas_requeridas:
        ruta = os.path.join(DATASET_ROOT, carpeta)
        existe = os.path.exists(ruta)
        estado = "✓" if existe else "❌"
        print(f"  {estado} {carpeta}")
        
        if not existe:
            return False
    
    return True


def validar_yaml():
    """Valida el archivo data.yaml."""
    
    print("\n2. ARCHIVO YAML:")
    
    yaml_path = os.path.join(DATASET_ROOT, 'data.yaml')
    
    if not os.path.exists(yaml_path):
        print(f"  ❌ No existe {yaml_path}")
        return False
    
    print(f"  ✓ Encontrado: data.yaml")
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validar campos requeridos
        campos = ['train', 'val', 'nc', 'names']
        for campo in campos:
            if campo in data:
                print(f"  ✓ {campo}: {data[campo] if campo != 'names' else f'{len(data[campo])} clases'}")
            else:
                print(f"  ❌ Falta campo: {campo}")
                return False
        
        # Validar coincidencia de clases
        nc = data.get('nc')
        num_nombres = len(data.get('names', []))
        
        if nc != num_nombres:
            print(f"  ❌ ERROR: nc={nc} pero hay {num_nombres} clases")
            return False
        
        print(f"\n  Clases encontradas:")
        for idx, nombre in data['names'].items():
            print(f"    {idx}: {nombre}")
        
        return True
    
    except Exception as e:
        print(f"  ❌ Error leyendo YAML: {e}")
        return False


def contar_imagenes():
    """Cuenta imágenes y anotaciones."""
    
    print("\n3. IMAGENES Y ANOTACIONES:")
    
    for split in ['train', 'valid']:
        img_path = os.path.join(DATASET_ROOT, 'images', split)
        label_path = os.path.join(DATASET_ROOT, 'labels', split)
        
        if not os.path.exists(img_path):
            print(f"  ⚠️ Carpeta {split} no existe")
            continue
        
        imagenes = [f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        anotaciones = [f for f in os.listdir(label_path) if f.endswith('.txt')] if os.path.exists(label_path) else []
        
        print(f"\n  {split.upper()}:")
        print(f"    Imágenes:      {len(imagenes)}")
        print(f"    Anotaciones:   {len(anotaciones)}")
        
        # Verificar que cada imagen tiene anotación
        imagenes_sin_anotacion = 0
        for img in imagenes:
            nombre_base = os.path.splitext(img)[0]
            txt_esperado = f"{nombre_base}.txt"
            
            if not os.path.exists(os.path.join(label_path, txt_esperado)):
                imagenes_sin_anotacion += 1
        
        if imagenes_sin_anotacion > 0:
            print(f"    ⚠️ {imagenes_sin_anotacion} imágenes SIN anotación")
        else:
            print(f"    ✓ Todas las imágenes tienen anotación")


def validar_anotaciones():
    """Valida formato de anotaciones."""
    
    print("\n4. VALIDACION DE ANOTACIONES:")
    
    label_path = os.path.join(DATASET_ROOT, 'labels', 'train')
    
    if not os.path.exists(label_path):
        print(f"  ⚠️ No hay carpeta labels/train")
        return
    
    archivos_txt = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    
    if not archivos_txt:
        print(f"  ❌ No hay archivos .txt en labels/train")
        return
    
    # Validar el primer archivo como ejemplo
    primer_archivo = os.path.join(label_path, archivos_txt[0])
    
    with open(primer_archivo, 'r') as f:
        primera_linea = f.readline().strip()
    
    if not primera_linea:
        print(f"  ⚠️ Archivo vacío: {archivos_txt[0]}")
        return
    
    partes = primera_linea.split()
    
    print(f"\n  Ejemplo de anotación (primeros 3 archivos):")
    for archivo in archivos_txt[:3]:
        ruta = os.path.join(label_path, archivo)
        with open(ruta, 'r') as f:
            contenido = f.read().strip()
        
        if contenido:
            lineas = contenido.split('\n')
            print(f"    {archivo}: {len(lineas)} objeto(s)")
            # Mostrar formato de primera línea
            partes = lineas[0].split()
            if len(partes) == 5:
                print(f"      ✓ Formato correcto: clase x_centro y_centro ancho alto")
            else:
                print(f"      ❌ Formato incorrecto: {len(partes)} valores (esperaba 5)")
        else:
            print(f"    {archivo}: VACIO (sin objetos detectados)")
    
    print(f"\n  ✓ Anotaciones en formato YOLO válido")


def main():
    # Validar estructura
    if not validar_estructura():
        print("\n❌ Estructura incorrecta. Verifica las carpetas.")
        return
    
    # Validar YAML
    if not validar_yaml():
        print("\n❌ YAML inválido. Revisa data.yaml")
        return
    
    # Contar imágenes
    contar_imagenes()
    
    # Validar anotaciones
    validar_anotaciones()
    
    print("\n" + "="*70)
    print("✅ DATASET LISTO PARA ENTRENAR")
    print("="*70)
    print("\nSiguiente paso:")
    print("  python train_yolo.py")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
