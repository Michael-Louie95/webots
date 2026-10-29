"""
Script: evaluar_modelo.py
Evalúa el modelo entrenado en el dataset de validación

Universidad Autonoma de Queretaro - Maestria en IA
"""

from ultralytics import YOLO
import os

MODELO = r'runs/detect/train_dataset/weights/best.pt'
DATASET_YAML = r'D:\proyectro_webots\dataset_yolo\data.yaml'

def main():
    print("="*70)
    print("EVALUACION DEL MODELO ENTRENADO")
    print("="*70)
    
    # Validar que exista el modelo
    if not os.path.exists(MODELO):
        print(f"\n❌ ERROR: No existe {MODELO}")
        print("Ejecuta primero: python train_yolo.py")
        return
    
    print(f"\nModelo: {MODELO}")
    print(f"Dataset: {DATASET_YAML}")
    
    print("\n" + "-"*70)
    print("Cargando modelo...")
    print("-"*70 + "\n")
    
    # Cargar modelo entrenado
    model = YOLO(MODELO)
    
    print("\n" + "-"*70)
    print("Validando en dataset de validación...")
    print("-"*70 + "\n")
    
    # Validar
    metrics = model.val(data=DATASET_YAML)
    
    print("\n" + "="*70)
    print("METRICAS DE VALIDACION")
    print("="*70)
    
    print("\nRESULTADOS:")
    print("-"*70)
    
    # Extraer métricas principales
    if hasattr(metrics, 'results_dict'):
        resultados = metrics.results_dict
        
        # mAP (Mean Average Precision) - métrica principal
        if 'metrics/mAP50' in resultados:
            mAP50 = resultados['metrics/mAP50']
            print(f"mAP@0.5 (detectar objetos): {mAP50:.4f}")
            
            if mAP50 > 0.85:
                print("  ✅ EXCELENTE - Supera meta de tesis (>85%)")
            elif mAP50 > 0.75:
                print("  ✓ BUENO - Acercándose a la meta")
            else:
                print("  ⚠️ Necesita mejora")
        
        if 'metrics/mAP50-95' in resultados:
            mAP50_95 = resultados['metrics/mAP50-95']
            print(f"mAP@0.5:0.95 (más estricto): {mAP50_95:.4f}")
        
        # Precisión y Recall
        if 'metrics/precision' in resultados:
            precision = resultados['metrics/precision']
            print(f"\nPrecisión (falsos positivos): {precision:.4f}")
        
        if 'metrics/recall' in resultados:
            recall = resultados['metrics/recall']
            print(f"Recall (detecciones perdidas): {recall:.4f}")
        
        # Mostrar todas las métricas
        print("\n" + "-"*70)
        print("TODAS LAS METRICAS:")
        print("-"*70)
        for metrica, valor in sorted(resultados.items()):
            if isinstance(valor, (int, float)):
                print(f"  {metrica}: {valor:.4f}")
    
    print("\n" + "="*70)
    print("ANALISIS:")
    print("="*70)
    
    # Interpretar resultados
    if hasattr(metrics, 'results_dict'):
        mAP50 = metrics.results_dict.get('metrics/mAP50', 0)
        precision = metrics.results_dict.get('metrics/precision', 0)
        recall = metrics.results_dict.get('metrics/recall', 0)
        
        print(f"\n📊 INTERPRETACION:")
        
        if precision > 0.9:
            print(f"  ✅ Precisión excelente - muy pocos falsos positivos")
        elif precision > 0.75:
            print(f"  ✓ Precisión buena - algunos falsos positivos")
        else:
            print(f"  ⚠️ Baja precisión - demasiados falsos positivos")
        
        if recall > 0.9:
            print(f"  ✅ Recall excelente - detecta casi todos los objetos")
        elif recall > 0.75:
            print(f"  ✓ Recall bueno - detecta la mayoría")
        else:
            print(f"  ⚠️ Bajo recall - no detecta muchos objetos")
        
        if mAP50 > 0.85:
            print(f"\n  ✅ META DE TESIS CUMPLIDA (mAP@0.5 > 85%)")
        elif mAP50 > 0.70:
            print(f"\n  ✓ Desempeño aceptable - considera ajustes")
        else:
            print(f"\n  ⚠️ Necesita mejora - revisa dataset o hiperparámetros")
    
    print("\n" + "="*70)
    print("PROXIMOS PASOS:")
    print("="*70)
    print(f"\n1. Para detección en imágenes:")
    print(f"   python inferencia.py --imagen foto.jpg")
    print(f"\n2. Para detección en video:")
    print(f"   python inferencia.py --video video.mp4")
    print(f"\n3. Si necesitas mejor precisión:")
    print(f"   - Agrega más datos de validación")
    print(f"   - Aumenta epochs en train_yolo.py")
    print(f"   - Usa YOLOv8m en lugar de v8s")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
