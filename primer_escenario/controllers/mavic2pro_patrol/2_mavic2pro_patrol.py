"""
Controlador: mavic2pro_patrol.py
Navegación autónoma punto A → B con:
  - Despegue automático a 1 metro de altura
  - Navegación GPS hacia punto destino
  - Detección YOLO + LiDAR (evasión si objeto < 0.60m)
  - Reanuda ruta después de evadir

Universidad Autónoma de Querétaro - Maestría en IA
Autor: Ing. Michael Louie Guerra Huante
"""

from controller import Robot
import math

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURACION — modifica aquí
# ════════════════════════════════════════════════════════════════════════════════

# Punto A (origen del dron en el mundo .wbt)
PUNTO_A = (0.0, 0.0)          # (x, y) en metros

# Punto B (destino final)
PUNTO_B = (45.0, 0.0)          # (x, y) en metros — dirección de los obstáculos

# Altura de crucero (metros)
ALTURA_CRUCERO = 1.0

# Velocidad de avance hacia el destino
VELOCIDAD_CRUCERO = 0.5        # perturbación de pitch hacia adelante

# Tolerancia para considerar que llegó al destino
TOLERANCIA_LLEGADA = 1.0       # metros

# ── Evasión ────────────────────────────────────────────────────────────────────
DISTANCIA_CRITICA    = 0.80    # LiDAR: metros para activar evasión
YOLO_CONFIANZA_MIN   = 0.50    # Confianza mínima YOLO
VELOCIDAD_EVASION    = 1.2     # Intensidad de la maniobra
TIEMPO_EVASION       = 5.0     # Segundos máximos evadiendo
YOLO_CADA_N_FRAMES   = 5       # Ejecutar YOLO cada N frames

# Ruta al modelo entrenado
MODELO_YOLO_PATH = r"D:\proyectro_webots\runs\detect\train_dataset\weights\best.pt"

# ════════════════════════════════════════════════════════════════════════════════
# CARGA YOLO
# ════════════════════════════════════════════════════════════════════════════════

try:
    from ultralytics import YOLO
    import numpy as np
    modelo_yolo   = YOLO(MODELO_YOLO_PATH)
    YOLO_ACTIVO   = True
    print(f"✓ YOLO cargado | clases: {list(modelo_yolo.names.values())}")
except Exception as e:
    YOLO_ACTIVO   = False
    print(f"⚠  YOLO no disponible: {e}")

# ── Utilidades ────────────────────────────────────────────────────────────────
def clamp(value, low, high):
    return max(low, min(value, high))

# ════════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN WEBOTS
# ════════════════════════════════════════════════════════════════════════════════

robot    = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)

lidar = robot.getDevice("lidar_frontal")
lidar.enable(timestep)
lidar.enablePointCloud()   # necesario en R2025a para lecturas continuas

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
K_NAV_P           = 0.4        # ganancia de navegación GPS
K_NAV_MAX         = 1.5        # perturbación máxima de navegación

# ════════════════════════════════════════════════════════════════════════════════
# MÁQUINA DE ESTADOS
# ════════════════════════════════════════════════════════════════════════════════

class Fase:
    DESPEGAR    = "despegar"      # Subir a ALTURA_CRUCERO
    NAVEGAR     = "navegar"       # Volar de A hacia B
    DETENIDO    = "detenido"      # Obstáculo detectado, esperar
    EVADIENDO   = "evadiendo"     # Maniobra de evasión
    DESPEJANDO  = "despejando"    # Breve avance tras evadir
    LLEGADA     = "llegada"       # Llegó al punto B, hovering

fase                  = Fase.DESPEGAR
target_altitude       = ALTURA_CRUCERO
evasion_inicio        = 0.0
evasion_direccion     = None
regresa_a_navegar     = False     # bandera para retomar ruta tras evasión

# Estado YOLO / LiDAR
frame_contador        = 0
yolo_detecta          = False
clase_detectada       = None
confianza_yolo        = 0.0
distancia_lidar       = 99.0

# ════════════════════════════════════════════════════════════════════════════════
# FUNCIONES
# ════════════════════════════════════════════════════════════════════════════════

### Estado interno para detectar lecturas congeladas
_lidar_ultima_lectura  = None
_lidar_contador_igual  = 0
_LIDAR_MAX_IGUAL       = 10    # si el valor no cambia en N frames = congelado

def leer_lidar():
    """
    Lee el LiDAR y devuelve (global, izq, centro, der).
    Detecta lecturas congeladas comparando con la lectura anterior.
    Si el LiDAR lleva N frames sin cambiar → retorna 99.0 (sin obstáculo).
    """
    global _lidar_ultima_lectura, _lidar_contador_igual

    try:
        ranges = lidar.getRangeImage()

        # Webots devuelve None o lista vacía si el sensor no está listo
        if not ranges or len(ranges) == 0:
            return 99.0, 99.0, 99.0, 99.0

        # Filtrar valores válidos (descarta inf, nan, 0, fuera de rango)
        import math as _math
        def _valido(r):
            return (r is not None and
                    not _math.isnan(r) and
                    not _math.isinf(r) and
                    0.05 < r < 10.0)

        def _min_seg(seg):
            v = [r for r in seg if _valido(r)]
            return round(min(v), 3) if v else 99.0

        n   = len(ranges)
        t   = n // 3
        izq = _min_seg(ranges[:t])
        cen = _min_seg(ranges[t:2*t])
        der = _min_seg(ranges[2*t:])
        glob = _min_seg(ranges)

        # ── Detección de congelamiento ────────────────────────────────────────
        # Comparar checksum simple (suma de primeros 10 valores)
        checksum_actual = round(sum(r for r in ranges[:10] if _valido(r)), 4)

        if checksum_actual == _lidar_ultima_lectura:
            _lidar_contador_igual += 1
            if _lidar_contador_igual == _LIDAR_MAX_IGUAL:
                print(f"  [LIDAR] ⚠ Lectura congelada detectada "
                      f"(mismo valor {_lidar_contador_igual} frames seguidos)")
            if _lidar_contador_igual > _LIDAR_MAX_IGUAL:
                # Lectura congelada — probablemente el LiDAR no está actualizando
                # Retornamos 99.0 para no disparar evasión por un valor falso
                return 99.0, 99.0, 99.0, 99.0
        else:
            _lidar_contador_igual = 0

        _lidar_ultima_lectura = checksum_actual

        return glob, izq, cen, der

    except Exception as e:
        return 99.0, 99.0, 99.0, 99.0


def ejecutar_yolo():
    """Retorna (detecta, clase, confianza)."""
    if not YOLO_ACTIVO:
        return False, None, 0.0
    try:
        import numpy as np
        img = camera.getImage()
        if img is None:
            return False, None, 0.0
        w, h = camera.getWidth(), camera.getHeight()
        arr  = np.frombuffer(img, dtype=np.uint8).reshape((h, w, 4))
        rgb  = arr[:, :, :3]
        res  = modelo_yolo.predict(rgb, conf=YOLO_CONFIANZA_MIN,
                                   verbose=False, imgsz=320)
        boxes = res[0].boxes
        if len(boxes) == 0:
            return False, None, 0.0
        idx  = boxes.conf.argmax().item()
        cid  = int(boxes.cls[idx])
        conf = float(boxes.conf[idx])
        name = res[0].names[cid]
        return True, name, conf
    except Exception as e:
        print(f"  [YOLO] Error: {e}")
        return False, None, 0.0


def decidir_evasion(izq, cen, der, alt):
    """Elige la dirección de evasión con más espacio libre."""
    print(f"  [EVASIÓN] sectores → izq={izq:.2f}m  cen={cen:.2f}m  der={der:.2f}m")
    LIBRE = 1.5
    if der > izq and der > LIBRE:
        return "derecha"
    if izq > der and izq > LIBRE:
        return "izquierda"
    if alt < 4.0:
        return "arriba"
    return "abajo"


def angulo_hacia_destino(px, py, dx, dy):
    """Yaw necesario para apuntar del punto (px,py) al destino (dx,dy)."""
    return math.atan2(dy - py, dx - px)


def distancia_a_destino(px, py, dx, dy):
    return math.sqrt((dx - px)**2 + (dy - py)**2)


# ════════════════════════════════════════════════════════════════════════════════
# BIENVENIDA
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  DRON MAVIC 2 PRO  —  NAVEGACIÓN A → B CON EVASIÓN")
print("="*65)
print(f"  Punto A (inicio):  {PUNTO_A}")
print(f"  Punto B (destino): {PUNTO_B}")
print(f"  Altura crucero:    {ALTURA_CRUCERO} m")
print(f"  Velocidad:         {VELOCIDAD_CRUCERO}")
print(f"  Distancia crítica: {DISTANCIA_CRITICA} m")
print(f"  YOLO:              {'✓ Activo' if YOLO_ACTIVO else '⚠ No disponible'}")
print("="*65 + "\n")

# Esperar 1 s para que los sensores estén listos
while robot.step(timestep) != -1:
    if robot.getTime() > 1.0:
        break

# ════════════════════════════════════════════════════════════════════════════════
# BUCLE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

while robot.step(timestep) != -1:
    t   = robot.getTime()
    dt  = timestep / 1000.0

    # ── Sensores ──────────────────────────────────────────────────────────────
    rpy            = imu.getRollPitchYaw()
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    pos            = gps.getValues()
    pos_x, pos_y, altitud = pos[0], pos[1], pos[2]
    roll_vel       = gyro.getValues()[0]
    pitch_vel      = gyro.getValues()[1]

    # ── LEDs ──────────────────────────────────────────────────────────────────
    led = int(t) % 2
    front_left_led.set(led)
    front_right_led.set(not led)

    # ── Gimbal ────────────────────────────────────────────────────────────────
    camera_roll_motor.setPosition(-0.115 * roll_vel)
    camera_pitch_motor.setPosition(-0.1  * pitch_vel)

    # ── LiDAR ─────────────────────────────────────────────────────────────────
    distancia_lidar, d_izq, d_cen, d_der = leer_lidar()

    # ── YOLO (cada N frames) ──────────────────────────────────────────────────
    frame_contador += 1
    if frame_contador >= YOLO_CADA_N_FRAMES:
        frame_contador = 0
        yolo_detecta, clase_detectada, confianza_yolo = ejecutar_yolo()

    # ── Distancia al destino ──────────────────────────────────────────────────
    dist_destino = distancia_a_destino(pos_x, pos_y, PUNTO_B[0], PUNTO_B[1])

    # ════════════════════════════════════════════════════════════════════════
    # LÓGICA DE FASES
    # ════════════════════════════════════════════════════════════════════════

    roll_disturbance  = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance   = 0.0

    # ── FASE: DESPEGAR ────────────────────────────────────────────────────────
    if fase == Fase.DESPEGAR:
        # Solo subir, sin moverse en XY
        if altitud >= ALTURA_CRUCERO - 0.05:
            fase = Fase.NAVEGAR
            print(f"✓ Despegue completado a {altitud:.2f}m → Navegando hacia {PUNTO_B}")

    # ── FASE: NAVEGAR ─────────────────────────────────────────────────────────
    elif fase == Fase.NAVEGAR:

        # ¿Llegó al destino?
        if dist_destino < TOLERANCIA_LLEGADA:
            fase = Fase.LLEGADA
            print(f"\n✓ ¡LLEGADA AL PUNTO B! Distancia final: {dist_destino:.2f}m")
            print(f"  Iniciando hovering en {PUNTO_B}")

        # ¿Obstáculo peligroso? (YOLO + LiDAR)
        elif yolo_detecta and distancia_lidar < DISTANCIA_CRITICA:
            fase          = Fase.DETENIDO
            evasion_inicio = t
            print(f"\n⚠  OBSTÁCULO EN RUTA")
            print(f"   Clase:    {clase_detectada}  ({confianza_yolo*100:.1f}%)")
            print(f"   LiDAR:    {distancia_lidar:.2f}m")
            print(f"   Acción:   DETENIÉNDOSE...")

        else:
            # Navegación GPS: calcular yaw deseado hacia el destino
            yaw_deseado = angulo_hacia_destino(pos_x, pos_y,
                                               PUNTO_B[0], PUNTO_B[1])
            error_yaw   = yaw_deseado - yaw

            # Normalizar error yaw [-π, π]
            while error_yaw >  math.pi: error_yaw -= 2 * math.pi
            while error_yaw < -math.pi: error_yaw += 2 * math.pi

            yaw_disturbance = clamp(error_yaw * K_YAW_P, -1.3, 1.3)

            # Solo avanzar si el dron ya está apuntando aprox. al destino
            if abs(error_yaw) < math.radians(25):
                pitch_disturbance = -VELOCIDAD_CRUCERO   # avanzar
            else:
                pitch_disturbance = 0.0                  # girar primero

            # Debug cada 2 s
            if int(t * 10) % 20 == 0:
                print(f"  → Navegando | dist={dist_destino:.1f}m | "
                      f"yaw_err={math.degrees(error_yaw):.1f}° | "
                      f"LiDAR={distancia_lidar:.2f}m | "
                      f"YOLO={'✓ '+clase_detectada if yolo_detecta else '—'}")

    # ── FASE: DETENIDO (0.5s para estabilizarse y analizar) ──────────────────
    elif fase == Fase.DETENIDO:
        # Sin perturbaciones → dron flota
        if t - evasion_inicio > 0.5:
            evasion_direccion = decidir_evasion(d_izq, d_cen, d_der, altitud)
            fase              = Fase.EVADIENDO
            evasion_inicio    = t
            print(f"   Decisión: EVADIR → {evasion_direccion.upper()}")

    # ── FASE: EVADIENDO ───────────────────────────────────────────────────────
    elif fase == Fase.EVADIENDO:

        if evasion_direccion == "derecha":
            roll_disturbance = -VELOCIDAD_EVASION
        elif evasion_direccion == "izquierda":
            roll_disturbance =  VELOCIDAD_EVASION
        elif evasion_direccion == "arriba":
            target_altitude += 0.02
        elif evasion_direccion == "abajo":
            target_altitude  = max(0.5, target_altitude - 0.02)

        # ¿Se despejó el obstáculo?
        if distancia_lidar > DISTANCIA_CRITICA * 1.5:
            fase           = Fase.DESPEJANDO
            evasion_inicio = t
            print(f"   Obstáculo despejado ({distancia_lidar:.2f}m) → despejando...")

        # Timeout de seguridad
        elif t - evasion_inicio > TIEMPO_EVASION * 2:
            print(f"   Timeout de evasión. Reintentando análisis...")
            fase           = Fase.DETENIDO
            evasion_inicio = t

    # ── FASE: DESPEJANDO (avanza brevemente para alejarse) ────────────────────
    elif fase == Fase.DESPEJANDO:
        pitch_disturbance = -0.5     # suave impulso hacia adelante

        if t - evasion_inicio > 0.8:
            fase         = Fase.NAVEGAR
            yolo_detecta = False
            print(f"✓  Evasión completada. Reanudando ruta hacia {PUNTO_B}\n")

    # ── FASE: LLEGADA (hovering en punto B) ───────────────────────────────────
    elif fase == Fase.LLEGADA:
        # Mantener posición — sin perturbaciones
        # Aquí podrías agregar más waypoints si quisieras
        pass

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
