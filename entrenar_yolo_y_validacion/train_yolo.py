"""
Script: train_yolo.py
Entrena YOLOv8s con tu dataset de objetos en calle

Universidad Autonoma de Queretaro - Maestria en IA
Alumno: Ing. Michael Louie Guerra Huante
"""

from ultralytics import YOLO
import os

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURACION
# ════════════════════════════════════════════════════════════════════════════════

DATASET_YAML = r'D:\proyectro_webots\dataset_yolo\data.yaml'

# Hiperparámetros recomendados para datos sintéticos de Webots
EPOCHS = 100
IMGSZ = 640
BATCH = 16
DEVICE = "cpu"          # 0 = GPU, "cpu" = CPU
PATIENCE = 20       # Early stopping
AUGMENT = True      # Datos sintéticos necesitan augmentation
SEED = 42           # Reproducibilidad

# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("ENTRENAMIENTO YOLOV8S - DETECCION DE OBJETOS EN CALLE")
    print("="*70)
    
    # Validar que exista el YAML
    if not os.path.exists(DATASET_YAML):
        print(f"\n❌ ERROR: No existe {DATASET_YAML}")
        print("Ejecuta primero: python validar_dataset.py")
        return
    
    print(f"\nDataset: {DATASET_YAML}")
    print(f"Modelo: YOLOv8s")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH}")
    print(f"Image Size: {IMGSZ}x{IMGSZ}")
    print(f"Augmentation: {'Sí' if AUGMENT else 'No'}")
    
    print("\n" + "-"*70)
    print("Descargando modelo YOLOv8s...")
    print("-"*70 + "\n")
    
    # Cargar modelo base (descarga automáticamente si no existe)
    model = YOLO('yolov8s.pt')
    
    print("\n" + "-"*70)
    print("Iniciando entrenamiento...")
    print("-"*70 + "\n")
    
    # Entrenar
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        patience=PATIENCE,
        augment=AUGMENT,
        
        # Augmentation agresiva para datos sintéticos
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.5,
        fliplr=0.5,
        
        # Otras configuraciones
        seed=SEED,
        save=True,
        verbose=True,
        val=True,
        plots=True,  # Descomenta para generar gráficos
        
        # Proyecto
        project='runs/detect',
        name='train_dataset',
        exist_ok=False,  # Cambia a True para reanudar entrenamiento
    )
    
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    
    # Resumen de resultados
    print("\nRESULTADOS:")
    print("-"*70)
    
    # El mejor modelo se guarda automáticamente en:
    best_model_path = os.path.join('runs/detect/train_dataset/weights/best.pt')
    
    if os.path.exists(best_model_path):
        print(f"✓ Mejor modelo guardado en: {best_model_path}")
    
    last_model_path = os.path.join('runs/detect/train_dataset/weights/last.pt')
    if os.path.exists(last_model_path):
        print(f"✓ Último modelo guardado en: {last_model_path}")
    
    # Mostrar métricas
    print("\nMÉTRICAS FINALES:")
    print("-"*70)
    
    if hasattr(results, 'results_dict'):
        for metrica, valor in results.results_dict.items():
            if isinstance(valor, (int, float)):
                print(f"  {metrica}: {valor:.4f}")
    
    print("\n" + "="*70)
    print("PROXIMOS PASOS:")
    print("="*70)
    print(f"\n1. Ver resultados en: runs/detect/train_dataset/")
    print(f"\n2. Evaluar modelo:")
    print(f"   python evaluar_modelo.py")
    print(f"\n3. Detectar en imágenes:")
    print(f"   python inferencia.py --imagen imagen.jpg")
    print(f"\n4. Detectar en video:")
    print(f"   python inferencia.py --video video.mp4")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
