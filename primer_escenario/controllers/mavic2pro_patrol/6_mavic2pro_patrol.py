"""
Controlador: mavic2pro_patrol.py
Navegación autónoma multi-corrida con métricas para artículo científico.

Patrón de vuelo (rectángulo):
    C ←──────── B
    |            |
    D ────────→ A   (regresa al inicio)

Cada corrida: A → B pasando por los obstáculos
Retorno:      B → C → D → A  (rodeando los obstáculos por fuera)

Métricas registradas por corrida:
  - Tiempo total de la corrida
  - Objetos detectados (YOLO) y clases
  - Evasiones exitosas y fallidas
  - Distancia mínima al obstáculo durante evasión
  - Tasa de éxito de evasión
  - Distancia total recorrida

Salidas:
  - log_sesion_YYYYMMDD_HHMMSS.csv   → datos por evento
  - metricas_YYYYMMDD_HHMMSS.json    → resumen por corrida
  - reporte_YYYYMMDD_HHMMSS.txt      → reporte legible para artículo

Universidad Autónoma de Querétaro - Maestría en IA
Autor: Ing. Michael Louie Guerra Huante
"""

from controller import Robot
import math
import csv
import json
import os
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURACION
# ════════════════════════════════════════════════════════════════════════════════

# ── Waypoints (metros) ────────────────────────────────────────────────────────
#
#   C(-5,1) ←────────── B(39,1)
#      |                     |
#   D(-5,-1) ───────────→ A(0,0) ───→ B(39,0) (ruta de obstáculos)
#
# La ruta de obstáculos es A→B (por el centro, con obstáculos)
# El retorno es B→C→D→A (por fuera, sin obstáculos)

PUNTO_A  = (0.0,   0.0)   # Inicio de la corrida
PUNTO_B  = (39.0,  0.0)   # Final de la corrida (con obstáculos en medio)
PUNTO_C  = (39.0,  1.0)  # Esquina retorno (lateral derecho)
PUNTO_D  = (0.0,   1.0)  # Esquina retorno (lateral izquierdo)

# ── Parámetros de vuelo ───────────────────────────────────────────────────────
ALTURA_CRUCERO    = 1.0    # metros
VELOCIDAD_CRUCERO = 0.5    # pitch disturbance
TOLERANCIA_WP     = 1.0    # metros para considerar waypoint alcanzado

# ── Evasión ───────────────────────────────────────────────────────────────────
DISTANCIA_CRITICA  = 0.80
YOLO_CONFIANZA_MIN = 0.50
VELOCIDAD_EVASION  = 1.2
TIEMPO_EVASION     = 5.0
DELAY_CRUCE        = 3.0   # segundos avanzando después de evadir para asegurar cruce
YOLO_CADA_N_FRAMES = 5

# ── Experimento ───────────────────────────────────────────────────────────────
NUM_CORRIDAS       = 2     # cuántas veces hace el recorrido A→B
PAUSA_ENTRE        = 2.0   # segundos de hovering entre corridas

# ── Modelo ────────────────────────────────────────────────────────────────────
MODELO_YOLO_PATH = r"D:\proyectro_webots\runs\detect\train_dataset\weights\best.pt"

# ── Directorio de logs ────────────────────────────────────────────────────────
LOG_DIR = r"D:\proyectro_webots\logs"

# ════════════════════════════════════════════════════════════════════════════════
# CARGA YOLO
# ════════════════════════════════════════════════════════════════════════════════

try:
    from ultralytics import YOLO
    import numpy as np
    modelo_yolo = YOLO(MODELO_YOLO_PATH)
    YOLO_ACTIVO = True
    print(f"✓ YOLO cargado | clases: {list(modelo_yolo.names.values())}")
except Exception as e:
    YOLO_ACTIVO = False
    print(f"⚠  YOLO no disponible: {e}")

def clamp(v, lo, hi): return max(lo, min(v, hi))

# ════════════════════════════════════════════════════════════════════════════════
# SISTEMA DE LOGGING Y MÉTRICAS
# ════════════════════════════════════════════════════════════════════════════════

os.makedirs(LOG_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

RUTA_CSV     = os.path.join(LOG_DIR, f"log_sesion_{ts}.csv")
RUTA_JSON    = os.path.join(LOG_DIR, f"metricas_{ts}.json")
RUTA_REPORTE = os.path.join(LOG_DIR, f"reporte_{ts}.txt")

# Cabeceras del CSV de eventos
CSV_CABECERAS = [
    "timestamp_real",   # hora real del sistema
    "t_sim",            # tiempo de simulación (s)
    "corrida",          # número de corrida (1..N)
    "evento",           # tipo de evento
    "fase",             # fase del controlador
    "pos_x", "pos_y", "altitud",
    "distancia_lidar",
    "yolo_clase", "yolo_confianza",
    "evasion_direccion",
    "dist_al_destino",
    "detalle",          # texto libre
]

csv_file   = open(RUTA_CSV, "w", newline="", encoding="utf-8")
csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_CABECERAS)
csv_writer.writeheader()

# Estructura de métricas por corrida
def nueva_corrida_metricas(num):
    return {
        "corrida":                   num,
        "t_inicio":                  0.0,
        "t_fin":                     0.0,
        "duracion_s":                0.0,
        "distancia_recorrida_m":     0.0,
        "completada":                False,

        # Detección
        "detecciones_yolo":          0,
        "clases_detectadas":         {},   # {clase: count}

        # Evasión
        "evasiones_iniciadas":       0,
        "evasiones_exitosas":        0,
        "evasiones_timeout":         0,
        "dist_minima_obstaculo_m":   99.0,
        "distancias_evasion":        [],   # lista de dist mínimas por evasión
        "direcciones_evasion":       [],   # ["derecha","arriba",...]

        # Tiempos
        "tiempo_navegando_s":        0.0,
        "tiempo_evadiendo_s":        0.0,
        "tiempo_detenido_s":         0.0,
    }

# Estado global de métricas
sesion = {
    "fecha":              ts,
    "num_corridas_prog":  NUM_CORRIDAS,
    "punto_a":            PUNTO_A,
    "punto_b":            PUNTO_B,
    "altura_crucero_m":   ALTURA_CRUCERO,
    "distancia_critica_m":DISTANCIA_CRITICA,
    "modelo_yolo":        MODELO_YOLO_PATH,
    "corridas":           [],
}

corrida_actual_num  = 1
corrida_metricas    = nueva_corrida_metricas(1)
pos_anterior        = None   # para calcular distancia recorrida


def registrar_evento(t, corrida, evento, fase, px, py, alt,
                     dist_lid, yolo_cls, yolo_conf,
                     ev_dir, dist_dest, detalle=""):
    """Escribe una fila en el CSV de eventos."""
    csv_writer.writerow({
        "timestamp_real":   datetime.now().isoformat(),
        "t_sim":            round(t, 3),
        "corrida":          corrida,
        "evento":           evento,
        "fase":             fase,
        "pos_x":            round(px, 3),
        "pos_y":            round(py, 3),
        "altitud":          round(alt, 3),
        "distancia_lidar":  round(dist_lid, 3),
        "yolo_clase":       yolo_cls or "",
        "yolo_confianza":   round(yolo_conf, 3),
        "evasion_direccion":ev_dir or "",
        "dist_al_destino":  round(dist_dest, 3),
        "detalle":          detalle,
    })
    csv_file.flush()


def guardar_metricas_json():
    """Guarda el JSON de métricas acumuladas."""
    # Calcular totales de sesión
    corridas_ok = [c for c in sesion["corridas"] if c["completada"]]
    sesion["resumen"] = {
        "corridas_completadas":       len(corridas_ok),
        "tasa_completitud_%":         round(len(corridas_ok)/max(1,len(sesion["corridas"]))*100,1),
        "total_detecciones":          sum(c["detecciones_yolo"] for c in sesion["corridas"]),
        "total_evasiones_iniciadas":  sum(c["evasiones_iniciadas"] for c in sesion["corridas"]),
        "total_evasiones_exitosas":   sum(c["evasiones_exitosas"] for c in sesion["corridas"]),
        "tasa_evasion_%":             round(
            sum(c["evasiones_exitosas"] for c in sesion["corridas"]) /
            max(1, sum(c["evasiones_iniciadas"] for c in sesion["corridas"])) * 100, 1),
        "dist_minima_global_m":       round(min(
            (c["dist_minima_obstaculo_m"] for c in sesion["corridas"]), default=99.0), 3),
        "duracion_promedio_s":        round(
            sum(c["duracion_s"] for c in corridas_ok) / max(1, len(corridas_ok)), 2),
        "distancia_promedio_m":       round(
            sum(c["distancia_recorrida_m"] for c in corridas_ok) / max(1, len(corridas_ok)), 2),
    }
    with open(RUTA_JSON, "w", encoding="utf-8") as f:
        json.dump(sesion, f, indent=2, ensure_ascii=False)


def generar_reporte_txt():
    """Genera reporte legible para artículo científico."""
    r = sesion.get("resumen", {})
    corridas = sesion["corridas"]

    lineas = []
    lineas.append("=" * 70)
    lineas.append("  REPORTE DE EXPERIMENTO — DETECCIÓN Y EVASIÓN DE OBSTÁCULOS")
    lineas.append("  Universidad Autónoma de Querétaro — Maestría en IA")
    lineas.append("  Alumno: Ing. Michael Louie Guerra Huante")
    lineas.append("=" * 70)
    lineas.append(f"\nFecha del experimento: {ts}")
    lineas.append(f"Modelo YOLO:          {MODELO_YOLO_PATH}")
    lineas.append(f"Distancia crítica:    {DISTANCIA_CRITICA} m")
    lineas.append(f"Altura de crucero:    {ALTURA_CRUCERO} m")
    lineas.append(f"Ruta:                 A{PUNTO_A} → B{PUNTO_B}")
    lineas.append(f"Corridas programadas: {NUM_CORRIDAS}")

    lineas.append("\n" + "─" * 70)
    lineas.append("  RESUMEN GLOBAL")
    lineas.append("─" * 70)
    lineas.append(f"  Corridas completadas:       {r.get('corridas_completadas','?')} / {NUM_CORRIDAS}")
    lineas.append(f"  Tasa de completitud:        {r.get('tasa_completitud_%','?')} %")
    lineas.append(f"  Total detecciones YOLO:     {r.get('total_detecciones','?')}")
    lineas.append(f"  Evasiones iniciadas:        {r.get('total_evasiones_iniciadas','?')}")
    lineas.append(f"  Evasiones exitosas:         {r.get('total_evasiones_exitosas','?')}")
    lineas.append(f"  Tasa de evasión:            {r.get('tasa_evasion_%','?')} %")
    lineas.append(f"  Distancia mínima global:    {r.get('dist_minima_global_m','?')} m")
    lineas.append(f"  Duración promedio/corrida:  {r.get('duracion_promedio_s','?')} s")
    lineas.append(f"  Distancia promedio/corrida: {r.get('distancia_promedio_m','?')} m")

    lineas.append("\n" + "─" * 70)
    lineas.append("  DETALLE POR CORRIDA")
    lineas.append("─" * 70)

    for c in corridas:
        estado = "✓ COMPLETADA" if c["completada"] else "✗ INCOMPLETA"
        tasa_ev = (round(c["evasiones_exitosas"] / max(1, c["evasiones_iniciadas"]) * 100, 1)
                   if c["evasiones_iniciadas"] > 0 else "N/A")
        dist_min = (f"{c['dist_minima_obstaculo_m']:.3f} m"
                    if c["dist_minima_obstaculo_m"] < 99.0 else "N/A")

        lineas.append(f"\n  Corrida #{c['corrida']}  {estado}")
        lineas.append(f"    Duración:              {c['duracion_s']:.2f} s")
        lineas.append(f"    Distancia recorrida:   {c['distancia_recorrida_m']:.2f} m")
        lineas.append(f"    Detecciones YOLO:      {c['detecciones_yolo']}")
        lineas.append(f"    Clases detectadas:     {c['clases_detectadas']}")
        lineas.append(f"    Evasiones iniciadas:   {c['evasiones_iniciadas']}")
        lineas.append(f"    Evasiones exitosas:    {c['evasiones_exitosas']}")
        lineas.append(f"    Tasa de evasión:       {tasa_ev} %")
        lineas.append(f"    Dist. mínima obst.:    {dist_min}")
        lineas.append(f"    Direcciones usadas:    {c['direcciones_evasion']}")
        lineas.append(f"    Tiempo navegando:      {c['tiempo_navegando_s']:.2f} s")
        lineas.append(f"    Tiempo evadiendo:      {c['tiempo_evadiendo_s']:.2f} s")
        lineas.append(f"    Tiempo detenido:       {c['tiempo_detenido_s']:.2f} s")

    lineas.append("\n" + "─" * 70)
    lineas.append("  VALIDACIÓN DE HIPÓTESIS DE TESIS")
    lineas.append("─" * 70)
    tasa_ev  = r.get("tasa_evasion_%", 0)
    dist_min = r.get("dist_minima_global_m", 99)
    lineas.append(f"\n  H1: Detección > 85% mAP@0.5")
    lineas.append(f"      (Ver métricas de entrenamiento YOLO)")
    lineas.append(f"\n  H2: Tasa de evasión ≥ 80%")
    lineas.append(f"      Resultado: {tasa_ev} %  →  {'✓ CUMPLE' if tasa_ev >= 80 else '✗ NO CUMPLE'}")
    lineas.append(f"\n  H3: Distancia mínima al obstáculo > 0.60 m")
    lineas.append(f"      Resultado: {dist_min} m  →  {'✓ CUMPLE' if dist_min > 0.60 else '✗ NO CUMPLE'}")

    lineas.append("\n" + "=" * 70)
    lineas.append("  ARCHIVOS GENERADOS")
    lineas.append("─" * 70)
    lineas.append(f"  CSV de eventos: {RUTA_CSV}")
    lineas.append(f"  JSON métricas:  {RUTA_JSON}")
    lineas.append(f"  Este reporte:   {RUTA_REPORTE}")
    lineas.append("=" * 70 + "\n")

    with open(RUTA_REPORTE, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

    # También imprimir en consola
    print("\n".join(lineas))


# ════════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN WEBOTS
# ════════════════════════════════════════════════════════════════════════════════

robot    = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

lidar = robot.getDevice("lidar_frontal")
lidar.enable(timestep)
lidar.enablePointCloud()

imu = robot.getDevice("inertial unit")
imu.enable(timestep)

gps = robot.getDevice("gps")
gps.enable(timestep)

gyro = robot.getDevice("gyro")
gyro.enable(timestep)

compass = robot.getDevice("compass")
compass.enable(timestep)

camera_roll_motor  = robot.getDevice("camera roll")
camera_pitch_motor = robot.getDevice("camera pitch")

front_left_led  = robot.getDevice("front left led")
front_right_led = robot.getDevice("front right led")

motor_names = [
    "front left propeller",
    "front right propeller",
    "rear left propeller",
    "rear right propeller",
]
motors = [robot.getDevice(n) for n in motor_names]
for m in motors:
    m.setPosition(float("inf"))
    m.setVelocity(1.0)

# ── Constantes PID ─────────────────────────────────────────────────────────────
K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P      = 3.0
K_ROLL_P          = 50.0
K_PITCH_P         = 30.0
K_YAW_P           = 2.0

# ════════════════════════════════════════════════════════════════════════════════
# MÁQUINA DE ESTADOS
# ════════════════════════════════════════════════════════════════════════════════

class Fase:
    DESPEGAR      = "despegar"
    NAVEGAR_AB    = "navegar_AB"    # A → B (con obstáculos)
    NAVEGAR_RET   = "navegar_RET"   # B → C → D → A (retorno libre)
    DETENIDO      = "detenido"
    EVADIENDO     = "evadiendo"
    DESPEJANDO    = "despejando"
    PAUSA         = "pausa"         # hovering entre corridas
    FIN           = "fin"

fase               = Fase.DESPEGAR
target_altitude    = ALTURA_CRUCERO
evasion_inicio     = 0.0
evasion_direccion  = None
altura_pre_evasion = ALTURA_CRUCERO   # guarda altura antes de evadir para regresar
pausa_inicio       = 0.0

# Waypoints del retorno: B → C → D → A
WP_RETORNO        = [PUNTO_C, PUNTO_D, PUNTO_A]
wp_retorno_idx    = 0         # cuál waypoint del retorno vamos

# YOLO / LiDAR
frame_contador    = 0
yolo_detecta      = False
clase_detectada   = None
confianza_yolo    = 0.0
distancia_lidar   = 99.0
d_izq = d_cen = d_der = 99.0

# Evasión actual
ev_dist_min_actual = 99.0    # distancia mínima en la evasión en curso

# Tiempo en cada fase (para métricas)
t_fase_inicio     = 0.0

# ════════════════════════════════════════════════════════════════════════════════
# FUNCIONES
# ════════════════════════════════════════════════════════════════════════════════

_lidar_ultima   = None
_lidar_n_igual  = 0
_LIDAR_MAX_EQ   = 60   # ~0.5s a 8ms/frame — zona abierta puede dar mismo valor legítimamente

def leer_lidar():
    """
    Lee LiDAR y devuelve (global, izq, centro, der).
    Anti-congelamiento mejorado:
      - Checksum sobre TODOS los rayos válidos (no solo primeros 10)
      - Zona abierta (todos inf) NO se trata como congelamiento
      - Solo silencia si hay valores válidos que no cambian
    """
    global _lidar_ultima, _lidar_n_igual
    try:
        ranges = lidar.getRangeImage()
        if not ranges:
            return 99.0, 99.0, 99.0, 99.0

        def ok(r):
            return (r is not None and
                    not math.isnan(r) and
                    not math.isinf(r) and
                    0.05 < r < 10.0)

        def mn(seg):
            v = [r for r in seg if ok(r)]
            return round(min(v), 3) if v else 99.0

        n   = len(ranges)
        t   = n // 3
        izq = mn(ranges[:t])
        cen = mn(ranges[t:2*t])
        der = mn(ranges[2*t:])
        gl  = mn(ranges)

        # Checksum sobre TODOS los rayos válidos (más representativo)
        vals_validos = [r for r in ranges if ok(r)]
        num_validos  = len(vals_validos)

        if num_validos == 0:
            # Zona completamente abierta — no hay nada cerca, es válido
            # No cuenta como congelamiento
            _lidar_n_igual = 0
            _lidar_ultima  = None
            return 99.0, 99.0, 99.0, 99.0

        # Checksum: suma de los 5 valores más pequeños (los obstáculos más cercanos)
        vals_validos.sort()
        cs = round(sum(vals_validos[:5]), 4)

        if cs == _lidar_ultima:
            _lidar_n_igual += 1
            if _lidar_n_igual == _LIDAR_MAX_EQ:
                print(f"  [LIDAR] ⚠ Posible congelamiento "
                      f"({_lidar_n_igual} frames sin cambio, min={gl:.2f}m)")
            if _lidar_n_igual > _LIDAR_MAX_EQ:
                # Solo silenciar si hay valores CERCANOS que no cambian
                # Si el mínimo es lejano (>3m) probablemente es zona abierta real
                if gl > 3.0:
                    _lidar_n_igual = 0   # resetear — es zona abierta, no congelamiento
                else:
                    return 99.0, 99.0, 99.0, 99.0
        else:
            _lidar_n_igual = 0

        _lidar_ultima = cs
        return gl, izq, cen, der

    except Exception as e:
        return 99.0, 99.0, 99.0, 99.0


def ejecutar_yolo():
    if not YOLO_ACTIVO: return False, None, 0.0
    try:
        import numpy as np
        img = camera.getImage()
        if img is None: return False, None, 0.0
        w,h = camera.getWidth(), camera.getHeight()
        arr = np.frombuffer(img, dtype=np.uint8).reshape((h,w,4))
        res = modelo_yolo.predict(arr[:,:,:3], conf=YOLO_CONFIANZA_MIN,
                                  verbose=False, imgsz=320)
        boxes = res[0].boxes
        if len(boxes)==0: return False, None, 0.0
        idx  = boxes.conf.argmax().item()
        cid  = int(boxes.cls[idx])
        conf = float(boxes.conf[idx])
        return True, res[0].names[cid], conf
    except Exception as e:
        print(f"  [YOLO] {e}"); return False, None, 0.0


def decidir_evasion(iz, ce, de, alt):
    """
    Prioridad:
      1. Derecha  — si hay ≥ 1.5m libres a la derecha
      2. Izquierda — si hay ≥ 1.5m libres a la izquierda
      3. Arriba   — solo si ambos lados están bloqueados
    """
    print(f"  [EVASIÓN] izq={iz:.2f} cen={ce:.2f} der={de:.2f}")
    LIBRE = 1.5

    # Intentar lateral primero — escoge el lado con más espacio
    if de >= LIBRE or iz >= LIBRE:
        if de >= iz:
            return "derecha"
        else:
            return "izquierda"

    # Ambos lados bloqueados → subir
    print(f"  [EVASIÓN] ambos lados bloqueados → subiendo")
    return "arriba"


def angulo_a(px, py, dx, dy): return math.atan2(dy-py, dx-px)
def dist_a(px, py, dx, dy):   return math.sqrt((dx-px)**2+(dy-py)**2)


def nav_hacia(px, py, yaw, dx, dy):
    """Calcula yaw_disturbance y pitch_disturbance hacia (dx,dy)."""
    yd = angulo_a(px, py, dx, dy)
    ey = yd - yaw
    while ey >  math.pi: ey -= 2*math.pi
    while ey < -math.pi: ey += 2*math.pi
    yd_dist = clamp(ey * K_YAW_P, -1.3, 1.3)
    pd_dist = -VELOCIDAD_CRUCERO if abs(ey) < math.radians(25) else 0.0
    return yd_dist, pd_dist


# ════════════════════════════════════════════════════════════════════════════════
# BIENVENIDA
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  EXPERIMENTO MULTI-CORRIDA — DETECCIÓN Y EVASIÓN")
print("="*65)
print(f"  Corridas programadas: {NUM_CORRIDAS}")
print(f"  Ruta AB (obstáculos): {PUNTO_A} → {PUNTO_B}")
print(f"  Retorno (libre):      {PUNTO_B} → {PUNTO_C} → {PUNTO_D} → {PUNTO_A}")
print(f"  Distancia crítica:    {DISTANCIA_CRITICA} m")
print(f"  Logs en:              {LOG_DIR}")
print("="*65 + "\n")

while robot.step(timestep) != -1:
    if robot.getTime() > 1.0: break

# ════════════════════════════════════════════════════════════════════════════════
# BUCLE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

while robot.step(timestep) != -1:

    if fase == Fase.FIN:
        break

    t  = robot.getTime()
    dt = timestep / 1000.0

    # ── Sensores ──────────────────────────────────────────────────────────────
    rpy                    = imu.getRollPitchYaw()
    roll, pitch, yaw       = rpy[0], rpy[1], rpy[2]
    pos                    = gps.getValues()
    pos_x, pos_y, altitud  = pos[0], pos[1], pos[2]
    roll_vel               = gyro.getValues()[0]
    pitch_vel              = gyro.getValues()[1]

    # LEDs
    led = int(t) % 2
    front_left_led.set(led); front_right_led.set(not led)

    # Gimbal
    camera_roll_motor.setPosition(-0.115 * roll_vel)
    camera_pitch_motor.setPosition(-0.1  * pitch_vel)

    # LiDAR
    distancia_lidar, d_izq, d_cen, d_der = leer_lidar()

    # YOLO (cada N frames)
    frame_contador += 1
    if frame_contador >= YOLO_CADA_N_FRAMES:
        frame_contador = 0
        yolo_detecta, clase_detectada, confianza_yolo = ejecutar_yolo()
        if yolo_detecta:
            corrida_metricas["detecciones_yolo"] += 1
            cd = corrida_metricas["clases_detectadas"]
            cd[clase_detectada] = cd.get(clase_detectada, 0) + 1

    # ── Distancia recorrida (odometría por GPS) ───────────────────────────────
    if pos_anterior:
        dx = pos_x - pos_anterior[0]
        dy = pos_y - pos_anterior[1]
        corrida_metricas["distancia_recorrida_m"] += math.sqrt(dx*dx + dy*dy)
    pos_anterior = (pos_x, pos_y)

    # Distancia al destino actual según fase
    if fase == Fase.NAVEGAR_AB:
        dist_destino = dist_a(pos_x, pos_y, PUNTO_B[0], PUNTO_B[1])
    elif fase == Fase.NAVEGAR_RET:
        wp = WP_RETORNO[wp_retorno_idx]
        dist_destino = dist_a(pos_x, pos_y, wp[0], wp[1])
    else:
        dist_destino = 0.0

    # ════════════════════════════════════════════════════════════════════════
    # LÓGICA DE FASES
    # ════════════════════════════════════════════════════════════════════════

    roll_disturbance  = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance   = 0.0

    # ── DESPEGAR ──────────────────────────────────────────────────────────────
    if fase == Fase.DESPEGAR:
        if altitud >= ALTURA_CRUCERO - 0.05:
            fase = Fase.NAVEGAR_AB
            t_fase_inicio = t
            corrida_metricas["t_inicio"] = t
            print(f"\n[C{corrida_actual_num}] Corrida iniciada → {PUNTO_A} → {PUNTO_B}")
            registrar_evento(t, corrida_actual_num, "INICIO_CORRIDA",
                             fase, pos_x, pos_y, altitud,
                             distancia_lidar, None, 0.0, None,
                             dist_a(pos_x,pos_y,PUNTO_B[0],PUNTO_B[1]),
                             f"corrida {corrida_actual_num} de {NUM_CORRIDAS}")

    # ── NAVEGAR A→B (con obstáculos) ──────────────────────────────────────────
    elif fase == Fase.NAVEGAR_AB:
        corrida_metricas["tiempo_navegando_s"] += dt

        if dist_destino < TOLERANCIA_WP:
            # Llegó a B
            fase = Fase.NAVEGAR_RET
            wp_retorno_idx = 0
            t_fase_inicio  = t
            print(f"\n[C{corrida_actual_num}] ✓ LLEGADA A PUNTO B — iniciando retorno")
            registrar_evento(t, corrida_actual_num, "LLEGADA_B",
                             fase, pos_x, pos_y, altitud,
                             distancia_lidar, None, 0.0, None, 0.0,
                             f"dist_final={dist_destino:.2f}m")

        elif yolo_detecta and distancia_lidar < DISTANCIA_CRITICA:
            ev_dist_min_actual = distancia_lidar
            altura_pre_evasion = altitud          # guardar altura actual
            corrida_metricas["evasiones_iniciadas"] += 1
            fase = Fase.DETENIDO
            evasion_inicio = t
            t_fase_inicio  = t
            print(f"\n[C{corrida_actual_num}] ⚠ OBSTÁCULO: {clase_detectada} "
                  f"({confianza_yolo*100:.1f}%) LiDAR={distancia_lidar:.2f}m")
            registrar_evento(t, corrida_actual_num, "OBSTACULO_DETECTADO",
                             fase, pos_x, pos_y, altitud,
                             distancia_lidar, clase_detectada, confianza_yolo,
                             None, dist_destino,
                             f"lidar={distancia_lidar:.3f}m")
        else:
            yaw_disturbance, pitch_disturbance = nav_hacia(
                pos_x, pos_y, yaw, PUNTO_B[0], PUNTO_B[1])

            # Si target_altitude se desvió (evasión vertical previa), corregir gradualmente
            if abs(target_altitude - ALTURA_CRUCERO) > 0.05:
                target_altitude += 0.01 * (1 if ALTURA_CRUCERO > target_altitude else -1)

            if int(t*10) % 20 == 0:
                print(f"  [C{corrida_actual_num}] dist={dist_destino:.1f}m | "
                      f"lidar={distancia_lidar:.2f}m | "
                      f"yolo={'✓'+clase_detectada if yolo_detecta else '—'}")
                registrar_evento(t, corrida_actual_num, "NAVEGANDO",
                                 fase, pos_x, pos_y, altitud,
                                 distancia_lidar,
                                 clase_detectada if yolo_detecta else None,
                                 confianza_yolo, None, dist_destino)

    # ── DETENIDO (analizar) ───────────────────────────────────────────────────
    elif fase == Fase.DETENIDO:
        corrida_metricas["tiempo_detenido_s"] += dt
        ev_dist_min_actual = min(ev_dist_min_actual, distancia_lidar)

        if t - evasion_inicio > 0.5:
            evasion_direccion = decidir_evasion(d_izq, d_cen, d_der, altitud)
            corrida_metricas["direcciones_evasion"].append(evasion_direccion)
            fase = Fase.EVADIENDO
            evasion_inicio = t
            print(f"  [C{corrida_actual_num}] → EVADIR {evasion_direccion.upper()}")
            registrar_evento(t, corrida_actual_num, "EVASION_INICIA",
                             fase, pos_x, pos_y, altitud,
                             distancia_lidar, clase_detectada, confianza_yolo,
                             evasion_direccion, dist_destino,
                             f"sectores izq={d_izq:.2f} cen={d_cen:.2f} der={d_der:.2f}")

    # ── EVADIENDO ─────────────────────────────────────────────────────────────
    elif fase == Fase.EVADIENDO:
        corrida_metricas["tiempo_evadiendo_s"] += dt
        ev_dist_min_actual = min(ev_dist_min_actual, distancia_lidar)

        # Aplicar perturbación según dirección
        if evasion_direccion == "derecha":
            roll_disturbance  = -VELOCIDAD_EVASION
        elif evasion_direccion == "izquierda":
            roll_disturbance  =  VELOCIDAD_EVASION
        elif evasion_direccion == "arriba":
            target_altitude  += 0.03   # sube 0.03m por frame

        # Para evasión lateral: salir cuando el LiDAR global se despejó
        # Para evasión vertical: salir cuando subió suficiente (GPS)
        if evasion_direccion in ("derecha", "izquierda"):
            despejado = distancia_lidar > DISTANCIA_CRITICA * 1.5
        else:  # arriba
            despejado = altitud >= altura_pre_evasion + 1.5

        if despejado:
            corrida_metricas["evasiones_exitosas"] += 1
            corrida_metricas["distancias_evasion"].append(round(ev_dist_min_actual, 3))
            corrida_metricas["dist_minima_obstaculo_m"] = min(
                corrida_metricas["dist_minima_obstaculo_m"], ev_dist_min_actual)
            fase           = Fase.DESPEJANDO
            evasion_inicio = t
            print(f"  [C{corrida_actual_num}] ✓ Evasión lograda "
                  f"({evasion_direccion}) — cruzando obstáculo {DELAY_CRUCE}s...")
            registrar_evento(t, corrida_actual_num, "EVASION_EXITOSA",
                             fase, pos_x, pos_y, altitud,
                             distancia_lidar, clase_detectada, confianza_yolo,
                             evasion_direccion, dist_destino,
                             f"dist_min_evasion={ev_dist_min_actual:.3f}m")

        elif t - evasion_inicio > TIEMPO_EVASION * 2:
            corrida_metricas["evasiones_timeout"] += 1
            print(f"  [C{corrida_actual_num}] ⚠ Timeout evasión. Reintentando...")
            registrar_evento(t, corrida_actual_num, "EVASION_TIMEOUT",
                             fase, pos_x, pos_y, altitud,
                             distancia_lidar, clase_detectada, confianza_yolo,
                             evasion_direccion, dist_destino)
            fase           = Fase.DETENIDO
            evasion_inicio = t

    # ── DESPEJANDO ────────────────────────────────────────────────────────────
    elif fase == Fase.DESPEJANDO:
        # Avanzar hacia el destino durante DELAY_CRUCE segundos
        # Esto garantiza que el dron pase completamente el obstáculo
        # antes de volver a la altura/posición original
        yaw_disturbance, pitch_disturbance = nav_hacia(
            pos_x, pos_y, yaw, PUNTO_B[0], PUNTO_B[1])

        tiempo_cruzando = t - evasion_inicio

        # Para evasión vertical: bajar solo después de DELAY_CRUCE
        if evasion_direccion == "arriba":
            if tiempo_cruzando > DELAY_CRUCE:
                diff_h = altura_pre_evasion - target_altitude
                if abs(diff_h) > 0.05:
                    target_altitude += 0.02 * (1 if diff_h > 0 else -1)
                else:
                    target_altitude = altura_pre_evasion

        # Salir de DESPEJANDO después de DELAY_CRUCE
        # Para vertical: además esperar a que la altura esté recuperada
        if evasion_direccion == "arriba":
            altura_ok = abs(altitud - altura_pre_evasion) < 0.20
            listo = tiempo_cruzando > DELAY_CRUCE and altura_ok
        else:
            listo = tiempo_cruzando > DELAY_CRUCE

        if listo:
            target_altitude    = altura_pre_evasion
            fase               = Fase.NAVEGAR_AB
            yolo_detecta       = False
            ev_dist_min_actual = 99.0
            print(f"  [C{corrida_actual_num}] ✓ Cruce completado → reanudando ruta"
                  f" | alt={altitud:.2f}m")

    # ── NAVEGAR RETORNO (B→C→D→A) ─────────────────────────────────────────────
    elif fase == Fase.NAVEGAR_RET:
        wp = WP_RETORNO[wp_retorno_idx]
        dist_wp = dist_a(pos_x, pos_y, wp[0], wp[1])

        if dist_wp < TOLERANCIA_WP:
            wp_retorno_idx += 1
            if wp_retorno_idx >= len(WP_RETORNO):
                # Llegó de regreso a A → corrida completada
                corrida_metricas["t_fin"]      = t
                corrida_metricas["duracion_s"] = round(t - corrida_metricas["t_inicio"], 2)
                corrida_metricas["completada"] = True
                corrida_metricas["distancia_recorrida_m"] = round(
                    corrida_metricas["distancia_recorrida_m"], 2)
                sesion["corridas"].append(corrida_metricas)
                guardar_metricas_json()

                print(f"\n[C{corrida_actual_num}] ✓ CORRIDA COMPLETADA")
                print(f"  Duración: {corrida_metricas['duracion_s']:.1f}s | "
                      f"Evasiones: {corrida_metricas['evasiones_exitosas']}/"
                      f"{corrida_metricas['evasiones_iniciadas']} | "
                      f"Dist.mín: {corrida_metricas['dist_minima_obstaculo_m']:.3f}m")

                registrar_evento(t, corrida_actual_num, "CORRIDA_COMPLETADA",
                                 fase, pos_x, pos_y, altitud,
                                 distancia_lidar, None, 0.0, None, 0.0,
                                 f"dur={corrida_metricas['duracion_s']}s "
                                 f"ev={corrida_metricas['evasiones_exitosas']}/"
                                 f"{corrida_metricas['evasiones_iniciadas']}")

                if corrida_actual_num >= NUM_CORRIDAS:
                    fase = Fase.FIN
                else:
                    corrida_actual_num += 1
                    corrida_metricas   = nueva_corrida_metricas(corrida_actual_num)
                    fase               = Fase.PAUSA
                    pausa_inicio       = t
                    print(f"\n  Pausa de {PAUSA_ENTRE}s antes de corrida #{corrida_actual_num}...")
            else:
                print(f"  [RET] → WP{wp_retorno_idx}: {WP_RETORNO[wp_retorno_idx]}")
        else:
            # Navegar al siguiente waypoint de retorno (sin evasión)
            yaw_disturbance, pitch_disturbance = nav_hacia(
                pos_x, pos_y, yaw, wp[0], wp[1])

    # ── PAUSA entre corridas ──────────────────────────────────────────────────
    elif fase == Fase.PAUSA:
        if t - pausa_inicio > PAUSA_ENTRE:
            fase = Fase.NAVEGAR_AB
            t_fase_inicio = t
            corrida_metricas["t_inicio"] = t
            print(f"\n[C{corrida_actual_num}] ▶ Iniciando corrida #{corrida_actual_num}")
            registrar_evento(t, corrida_actual_num, "INICIO_CORRIDA",
                             fase, pos_x, pos_y, altitud,
                             distancia_lidar, None, 0.0, None,
                             dist_a(pos_x,pos_y,PUNTO_B[0],PUNTO_B[1]),
                             f"corrida {corrida_actual_num} de {NUM_CORRIDAS}")

    # ════════════════════════════════════════════════════════════════════════
    # PID DE VUELO
    # ════════════════════════════════════════════════════════════════════════

    roll_input  = K_ROLL_P  * clamp(roll,  -1.0, 1.0) + roll_vel  + roll_disturbance
    pitch_input = K_PITCH_P * clamp(pitch, -1.0, 1.0) + pitch_vel + pitch_disturbance
    yaw_input   = yaw_disturbance

    diff_alt       = clamp(target_altitude - altitud + K_VERTICAL_OFFSET, -1.0, 1.0)
    vertical_input = K_VERTICAL_P * (diff_alt ** 3)

    fl = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
    fr = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
    rl = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
    rr = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input

    motors[0].setVelocity( fl)
    motors[1].setVelocity(-fr)
    motors[2].setVelocity(-rl)
    motors[3].setVelocity( rr)

# ════════════════════════════════════════════════════════════════════════════════
# CIERRE
# ════════════════════════════════════════════════════════════════════════════════

csv_file.close()
guardar_metricas_json()
generar_reporte_txt()
