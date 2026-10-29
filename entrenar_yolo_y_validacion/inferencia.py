"""
Script: inferencia.py
Detecta objetos en imágenes y videos

Uso:
  python inferencia.py --imagen foto.jpg
  python inferencia.py --video video.mp4
  python inferencia.py --camara 0

Universidad Autonoma de Queretaro - Maestria en IA
"""

from ultralytics import YOLO
import cv2
import argparse
import os

MODELO = r'runs/detect/train_dataset/weights/best.pt'
CONF_THRESHOLD = 0.5  # Confianza mínima (0-1)

def detectar_imagen(ruta_imagen):
    """Detecta objetos en una imagen."""
    
    print(f"\n{'='*70}")
    print(f"DETECCION EN IMAGEN")
    print(f"{'='*70}")
    
    if not os.path.exists(ruta_imagen):
        print(f"❌ ERROR: No existe {ruta_imagen}")
        return
    
    print(f"Imagen: {ruta_imagen}")
    print(f"Modelo: {MODELO}")
    
    # Cargar modelo
    model = YOLO(MODELO)
    
    # Detectar
    print(f"\nProcessando...")
    resultados = model.predict(ruta_imagen, conf=CONF_THRESHOLD, verbose=False)
    
    # Procesar resultados
    resultado = resultados[0]
    
    print(f"\n✅ DETECCIONES:")
    print(f"-"*70)
    
    if len(resultado.boxes) == 0:
        print(f"  No se detectaron objetos")
    else:
        for i, box in enumerate(resultado.boxes):
            clase_id = int(box.cls[0])
            clase_nombre = resultado.names[clase_id]
            confianza = float(box.conf[0])
            
            # Coordenadas del bbox
            x1, y1, x2, y2 = box.xyxy[0]
            ancho = float(x2 - x1)
            alto = float(y2 - y1)
            
            print(f"\n  Objeto {i+1}:")
            print(f"    Clase: {clase_nombre}")
            print(f"    Confianza: {confianza*100:.1f}%")
            print(f"    Bbox: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})")
            print(f"    Tamaño: {int(ancho)} x {int(alto)} px")
    
    # Guardar imagen con detecciones
    imagen_resultado = resultado.plot()
    nombre_salida = os.path.splitext(ruta_imagen)[0] + '_detecciones.jpg'
    cv2.imwrite(nombre_salida, imagen_resultado)
    
    print(f"\n✓ Imagen con detecciones guardada: {nombre_salida}")
    print(f"{'='*70}\n")


def detectar_video(ruta_video):
    """Detecta objetos en un video."""
    
    print(f"\n{'='*70}")
    print(f"DETECCION EN VIDEO")
    print(f"{'='*70}")
    
    if not os.path.exists(ruta_video):
        print(f"❌ ERROR: No existe {ruta_video}")
        return
    
    print(f"Video: {ruta_video}")
    print(f"Modelo: {MODELO}")
    
    # Cargar modelo
    model = YOLO(MODELO)
    
    # Detectar en video
    print(f"\nProcessando (esto puede tardar)...")
    resultados = model.predict(ruta_video, conf=CONF_THRESHOLD, verbose=False)
    
    # Contar detecciones totales
    total_detecciones = sum(len(r.boxes) for r in resultados)
    
    print(f"\n✅ RESULTADOS:")
    print(f"-"*70)
    print(f"  Frames procesados: {len(resultados)}")
    print(f"  Total detecciones: {total_detecciones}")
    print(f"  Promedio por frame: {total_detecciones/len(resultados):.1f}")
    
    # Guardar video con detecciones
    nombre_salida = os.path.splitext(ruta_video)[0] + '_detecciones.mp4'
    print(f"\n  Guardando video con detecciones...")
    print(f"  Esto puede tardar un momento...")
    
    # Usar YOLO predict con guardar
    model.predict(ruta_video, conf=CONF_THRESHOLD, save=True, verbose=False, 
                  project='runs/detect', name='video_detecciones')
    
    print(f"\n✓ Video procesado: runs/detect/video_detecciones/")
    print(f"{'='*70}\n")


def detectar_camara(id_camara=0):
    """Detecta objetos en tiempo real desde la cámara."""
    
    print(f"\n{'='*70}")
    print(f"DETECCION EN TIEMPO REAL")
    print(f"{'='*70}")
    print(f"Cámara: {id_camara}")
    print(f"Presiona 'Q' para salir")
    
    # Cargar modelo
    model = YOLO(MODELO)
    
    # Abrir cámara
    cap = cv2.VideoCapture(id_camara)
    
    if not cap.isOpened():
        print(f"❌ ERROR: No se puede abrir la cámara {id_camara}")
        return
    
    print(f"\n{'='*70}\n")
    
    frame_count = 0
    detecciones_total = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detectar cada 2 frames para acelerar (opcional)
        if frame_count % 2 == 0:
            resultados = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
            resultado = resultados[0]
            detecciones_total += len(resultado.boxes)
            
            # Dibujar detecciones
            frame_anotado = resultado.plot()
        else:
            frame_anotado = frame
        
        # Mostrar info
        info = f"FPS: ~30 | Detecciones: {detecciones_total} | Frames: {frame_count}"
        cv2.putText(frame_anotado, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow('Deteccion en Tiempo Real', frame_anotado)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ SESION COMPLETADA")
    print(f"  Frames capturados: {frame_count}")
    print(f"  Total detecciones: {detecciones_total}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Detecta objetos usando modelo YOLOv8 entrenado'
    )
    
    parser.add_argument('--imagen', type=str, help='Ruta a imagen')
    parser.add_argument('--video', type=str, help='Ruta a video')
    parser.add_argument('--camara', type=int, default=0, help='ID de cámara (0=default)')
    
    args = parser.parse_args()
    
    # Validar que exista el modelo
    if not os.path.exists(MODELO):
        print(f"\n❌ ERROR: No existe {MODELO}")
        print(f"Ejecuta primero: python train_yolo.py")
        return
    
    # Procesar según argumento
    if args.imagen:
        detectar_imagen(args.imagen)
    elif args.video:
        detectar_video(args.video)
    else:
        # Por defecto, usar cámara
        detectar_camara(args.camara)


if __name__ == "__main__":
    main()
