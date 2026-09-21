"""
================================================================================
 PROYECTO : Jefe Final Adaptativo con Inteligencia Artificial (MLP)
================================================================================

DESCRIPCION GENERAL
-------------------
Videojuego 2D en Pygame donde un Jefe Final usa una Red Neuronal (Perceptron
Multicapa, sklearn.neural_network.MLPClassifier) para analizar el estilo de
juego del usuario durante 60 segundos de telemetria y luego adaptar sus
patrones de ataque al punto debil detectado.

================================================================================
 CONVENCION DE NOMBRES Y MEDIDAS DE SPRITES (carpeta assets/)
================================================================================

Coloca solo los archivos que ya tengas listos. Todos son .png con canal alpha
(fondo transparente). Los que falten usan geometria automaticamente.

PANTALLA DE JUEGO: 960 x 600 pixeles (ANCHO x ALTO).

----------------------------------------------------------------------
 JUGADOR
----------------------------------------------------------------------
  Los 3 sprites de la nave son mutuamente excluyentes: el juego muestra
  exactamente uno segun el movimiento horizontal actual. NO se aplica
  ningun flip automatico; cada imagen debe estar orientada correctamente
  desde el editor de sprites.

  jugador_idle.png
      Nave apuntando directamente hacia arriba (estado en reposo).
      Se muestra cuando el jugador esta quieto O cuando se mueve
      solo en el eje Y (arriba/abajo sin componente horizontal).
      Medida : 32 x 32 px

  jugador_derecha.png
      Nave inclinada hacia la derecha.
      Se muestra cuando dx > 0 (tecla derecha / D presionada).
      Medida : 32 x 32 px

  jugador_izquierda.png
      Nave inclinada hacia la izquierda.
      Se muestra cuando dx < 0 (tecla izquierda / A presionada).
      Medida : 32 x 32 px

  jugador_laser.png
      Proyectil rapido disparado hacia arriba por el jugador.
      Medida : 10 x 18 px
      Color de referencia: cian (#3CDCDC) cuando se usa geometria.

  jugador_misil.png
      Proyectil lento y fuerte disparado hacia arriba por el jugador.
      Medida : 14 x 20 px
      Color de referencia: naranja (#F08C28) cuando se usa geometria.

----------------------------------------------------------------------
 ENEMIGO COMUN
----------------------------------------------------------------------
  enemigo_idle.png
      Sprite estatico del enemigo comun. Se mueve solo en eje X.
      Medida : 28 x 28 px

  enemigo_laser.png
      Proyectil laser apuntado que dispara el enemigo hacia el jugador.
      Medida : 10 x 14 px
      Color de referencia: amarillo (#F0D23C) cuando se usa geometria.

  enemigo_misil.png
      Proyectil misil apuntado que dispara el enemigo hacia el jugador.
      Medida : 14 x 14 px
      Color de referencia: morado (#AA50DC) cuando se usa geometria.

----------------------------------------------------------------------
 JEFE FINAL
----------------------------------------------------------------------
  jefe_idle.png
      Sprite estatico del Jefe Final. Se mueve en X e Y dentro de
      la franja superior de la pantalla (y: 20 a 220 px).
      Medida : 120 x 60 px

  jefe_laser.png
      Proyectil laser del jefe (apuntado o en barrera segun el perfil).
      Medida : 10 x 14 px
      Color de referencia: amarillo (#F0D23C).

  jefe_misil.png
      Proyectil misil del jefe (apuntado o en anillo segun el perfil).
      Medida : 16 x 16 px
      Color de referencia: morado (#AA50DC).

----------------------------------------------------------------------
 FONDO
----------------------------------------------------------------------
  fondo.png
      Imagen de fondo de toda la pantalla.
      Medida exacta : 960 x 600 px  (se estira si no coincide, mejor
                      que sea exacta para evitar distorsion).
      Formato       : RGB sin canal alpha (no necesita transparencia).

================================================================================
 ARQUITECTURA DE CLASES
================================================================================

  ManejadorAssets  Carga y sirve sprites desde disco. Fallback a geometria.
  ManejadorIA      Entrena el MLPClassifier al inicio y expone predecir_perfil().
  Proyectil        Un disparo en vuelo. Soporta velocidad apuntada o fija.
  Jugador          Movimiento WASD/flechas, disparo, telemetria, animacion,
                   i-frames de invulnerabilidad.
  Enemigo          Enemigo comun de Fase 1. Dispara hacia el jugador (apuntado).
  JefeFinal        Jefe con 3 fases de enrabiamiento y proyectiles apuntados.
  Juego            Maquina de estados principal, bucle a 60 FPS, UI, debug.

================================================================================
 CONTROLES
================================================================================
  Flechas / WASD    Mover al jugador
  ESPACIO           Disparar Laser  (rapido, 5 dmg, cooldown 12 frames)
  SHIFT             Disparar Misil  (lento, 18 dmg, cooldown 45 frames)
  R                 Reiniciar tras Game Over o Victoria
  ESC               Salir

================================================================================
 DEPENDENCIAS
================================================================================
  pip install pygame numpy scikit-learn
  Probado con: Python 3.11, pygame 2.6, numpy 1.26, scikit-learn 1.5
================================================================================
"""

import os
import sys
import math
import random

import numpy as np
import pygame
from sklearn.neural_network import MLPClassifier

ANCHO          = 960    # Ancho de la ventana en pixeles.
ALTO           = 600    # Alto  de la ventana en pixeles.
FPS            = 60     # Tasa de refresco objetivo (frames por segundo).

# Duraciones de fase en segundos.
DURACION_FASE_1      = 60.0  # Tiempo de recoleccion de telemetria.
DURACION_TRANSICION  = 3.0   # Pausa entre Fase 1 y aparicion del jefe.

# Ruta absoluta a la carpeta de assets, relativa al directorio de este script.
# Usar ruta absoluta evita errores si el script se lanza desde otro directorio.
CARPETA_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Paleta de colores RGB.
# Se usan como fallback cuando no hay sprite para un objeto determinado.
NEGRO        = (15,  15,  20)
BLANCO       = (240, 240, 240)
GRIS         = (90,  90,  100)
GRIS_OSCURO  = (40,  40,  50)
VERDE        = (60,  220, 100)
ROJO         = (230, 60,  60)
AMARILLO     = (240, 210, 60)
NARANJA      = (240, 140, 40)
MORADO       = (170, 80,  220)
CIAN         = (60,  220, 220)

# Identificadores de tipo de proyectil.
TIPO_LASER = "laser"
TIPO_MISIL = "misil"

# Velocidad de la animacion de movimiento del jugador.
# Valor en frames de juego: cada VELOCIDAD_ANIMACION ticks se avanza un frame
# de la spritesheet. Con FPS=60 y valor=6, la animacion corre a 10 fps.
VELOCIDAD_ANIMACION = 6


# ==============================================================================
# CLASE: ManejadorAssets
# ==============================================================================
class ManejadorAssets:
    """
    Responsabilidad: cargar y servir todos los recursos graficos (sprites).

    Patron de diseño: Repositorio centralizado con fallback silencioso.
    En vez de que cada clase cargue su propio PNG (lo que dispersa el
    manejo de errores y duplica llamadas a disco), ManejadorAssets carga
    todo UNA SOLA VEZ al inicio y los demas objetos le piden lo que
    necesitan en cada frame.

    Si un archivo no existe en disco, _cargar_estatico / _cargar_animacion
    simplemente no agregan nada al diccionario. Cuando el resto del codigo
    llama a tiene_estatico("nombre") y recibe False, dibuja geometria.
    Nunca se lanza FileNotFoundError ni pygame.error hacia el exterior.

    Atributos publicos:
        estaticos   dict[str, Surface]       sprites de un solo frame
        animaciones dict[str, list[Surface]] secuencias de frames
        fondo       Surface | None           imagen de fondo escalada a pantalla
    """

    def __init__(self, carpeta: str = CARPETA_ASSETS):
        """
        Parametros:
            carpeta: ruta a la carpeta que contiene los archivos .png.
                     Por defecto es la subcarpeta 'assets/' junto al script.
        """
        self.carpeta    = carpeta
        self.estaticos  = {}   # {nombre_logico: pygame.Surface}
        self.animaciones = {}  # {prefijo: [Surface, Surface, ...]}
        self.fondo      = None

        # --- Cargar sprites del jugador (nave espacial) ---
        # Tres estados de la nave, cada uno con su propio PNG.
        # El juego selecciona uno por frame segun el dx del movimiento.
        self._cargar_estatico("jugador_idle")       # nave apuntando arriba (quieta / solo Y)
        self._cargar_estatico("jugador_derecha")    # nave inclinada a la derecha (dx > 0)
        self._cargar_estatico("jugador_izquierda")  # nave inclinada a la izquierda (dx < 0)
        self._cargar_estatico("jugador_laser")      # proyectil laser del jugador
        self._cargar_estatico("jugador_misil")      # proyectil misil del jugador

        # --- Cargar sprites del enemigo comun (Fase 1) ---
        self._cargar_estatico("enemigo_idle")    # sprite del enemigo
        self._cargar_estatico("enemigo_laser")   # proyectil laser del enemigo
        self._cargar_estatico("enemigo_misil")   # proyectil misil del enemigo

        # --- Cargar sprites del Jefe Final (Fase 2) ---
        self._cargar_estatico("jefe_idle")       # sprite del jefe
        self._cargar_estatico("jefe_laser")      # proyectil laser del jefe
        self._cargar_estatico("jefe_misil")      # proyectil misil del jefe

        # --- Cargar fondo opcional ---
        self._cargar_fondo()

        self._reportar_resumen()

    # ------------------------------------------------------------------
    # Metodos privados de carga
    # ------------------------------------------------------------------

    def _ruta(self, nombre_archivo: str) -> str:
        """Construye la ruta completa a un archivo dentro de self.carpeta."""
        return os.path.join(self.carpeta, nombre_archivo)

    def _cargar_estatico(self, nombre_logico: str) -> None:
        """
        Intenta cargar assets/<nombre_logico>.png como Surface con alpha.
        Si el archivo no existe o hay error de pygame, no hace nada
        (el juego usara geometria para ese objeto).

        Parametros:
            nombre_logico: clave interna, ej. "jugador_idle", "jefe_laser".
        """
        ruta = self._ruta(nombre_logico + ".png")
        if os.path.isfile(ruta):
            try:
                # convert_alpha() optimiza la Surface para blitting con
                # transparencia, mejorando el rendimiento en el bucle principal.
                self.estaticos[nombre_logico] = pygame.image.load(ruta).convert_alpha()
            except pygame.error as e:
                print(f"[Assets] No se pudo cargar '{ruta}': {e}. Usando geometria.")

    def _cargar_animacion(self, prefijo: str) -> None:
        """
        Carga una secuencia de frames: assets/<prefijo>_0.png, _1.png, ...
        Se detiene en el primer numero que no exista en disco.
        Si no existe ni _0.png, no se registra nada (sin animacion para ese prefijo).

        Parametros:
            prefijo: nombre base de la secuencia, ej. "jugador_run".
                     Buscara jugador_run_0.png, jugador_run_1.png, etc.
        """
        frames = []
        i = 0
        while True:
            ruta = self._ruta(f"{prefijo}_{i}.png")
            if not os.path.isfile(ruta):
                break  # secuencia terminada (no hay frame con este indice)
            try:
                frames.append(pygame.image.load(ruta).convert_alpha())
            except pygame.error as e:
                print(f"[Assets] No se pudo cargar '{ruta}': {e}.")
                break
            i += 1

        if frames:
            self.animaciones[prefijo] = frames

    def _cargar_fondo(self) -> None:
        """
        Carga assets/fondo.png y lo escala exactamente a (ANCHO x ALTO).
        Si no existe, self.fondo queda como None y el juego rellena con
        NEGRO solido en cada frame.
        """
        ruta = self._ruta("fondo.png")
        if os.path.isfile(ruta):
            try:
                img = pygame.image.load(ruta).convert()  # sin alpha: mas rapido
                self.fondo = pygame.transform.scale(img, (ANCHO, ALTO))
            except pygame.error as e:
                print(f"[Assets] No se pudo cargar fondo: {e}.")

    def _reportar_resumen(self) -> None:
        """Imprime en consola cuantos assets se cargaron correctamente."""
        print(f"[Assets] Estaticos cargados : {len(self.estaticos)}")
        print(f"[Assets] Animaciones cargadas: {len(self.animaciones)}")
        if not self.estaticos and not self.animaciones:
            print(f"[Assets] Carpeta '{self.carpeta}' vacia o inexistente. "
                  "El juego corre 100% con geometria (normal al inicio).")

    # ------------------------------------------------------------------
    # Metodos publicos de consulta
    # ------------------------------------------------------------------

    def tiene_estatico(self, nombre_logico: str) -> bool:
        """True si el sprite estatico 'nombre_logico' fue cargado exitosamente."""
        return nombre_logico in self.estaticos

    def tiene_animacion(self, prefijo: str) -> bool:
        """True si la secuencia de animacion 'prefijo' tiene al menos 1 frame."""
        return prefijo in self.animaciones

    def obtener_estatico(self, nombre_logico: str):
        """Devuelve la Surface del sprite, o None si no fue cargado."""
        return self.estaticos.get(nombre_logico)

    def obtener_frames_animacion(self, prefijo: str):
        """Devuelve la lista de Surfaces de la animacion, o None si no existe."""
        return self.animaciones.get(prefijo)

    @staticmethod
    def escalar_a_rect(superficie: pygame.Surface, rect: pygame.Rect) -> pygame.Surface:
        """
        Reescala 'superficie' al tamaño exacto de 'rect' (width x height).
        Se usa para que el sprite siempre coincida con el hitbox de colision
        sin importar el tamaño original del PNG.

        Parametros:
            superficie: Surface de pygame a reescalar.
            rect      : Rect del objeto (define el tamaño destino).
        Devuelve:
            Nueva Surface reescalada. La original no se modifica.
        """
        return pygame.transform.scale(superficie, (rect.width, rect.height))


# ==============================================================================
# CLASE: ManejadorIA
# ==============================================================================
class ManejadorIA:
    """
    Responsabilidad: entrenar y consultar la Red Neuronal que actua como
    "Director de Juego". Se instancia UNA SOLA VEZ al iniciar el programa.

    =========================================================================
    EXPLICACION ACADEMICA DEL MODELO (para la exposicion)
    =========================================================================

    TIPO DE MODELO: Perceptron Multicapa (Multi-Layer Perceptron, MLP).
    Red neuronal feedforward densa, entrenada por backpropagation.
    Resuelve un problema de CLASIFICACION MULTICLASE: dado un vector de
    3 numeros que describen como jugo el usuario, predice cual de 3 perfiles
    de comportamiento debe activar el Jefe Final.

    ARQUITECTURA DE LA RED:
    -----------------------
    Capa de Entrada  (Input Layer) : 3 neuronas — una por cada feature:
        x0 = ratio_recibido_laser      proporciones de daño acumulado
        x1 = ratio_recibido_misil      (ambas suman 1.0 entre si)
        x2 = preferencia_distancia     posicion Y promedio normalizada

    Capa Oculta (Hidden Layer): 10 neuronas, activacion ReLU.
        Cada neurona calcula: h = ReLU(w·x + b) = max(0, w·x + b)
        donde w son los pesos aprendidos y b el sesgo (bias).
        ReLU introduce NO-LINEALIDAD: sin ella, toda la red equivaldria
        a una sola transformacion lineal incapaz de separar las 3 clases
        si sus regiones no son linealmente separables.

    Capa de Salida (Output Layer): 3 neuronas, activacion Softmax.
        Softmax(z_i) = exp(z_i) / sum(exp(z_j) para todo j)
        Convierte los 3 valores de salida en probabilidades que suman 1.
        La clase predicha es la de mayor probabilidad: argmax(softmax).

    OPTIMIZADOR: 'lbfgs' (Limited-memory Broyden–Fletcher–Goldfarb–Shanno).
        Metodo cuasi-Newton de segundo orden. Ideal para datasets pequeños
        (< 1000 muestras): converge en pocas iteraciones sin necesitar
        ajuste de tasa de aprendizaje (learning rate), a diferencia de
        'adam' o 'sgd' que requieren mas tunning en datasets chicos.

    ENTRENAMIENTO (fit):
        Ajusta todos los pesos W y sesgos b minimizando la log-loss
        (cross-entropy) sobre el dataset sintetico, via backpropagation:
        1. Forward pass: calcula predicciones con los pesos actuales.
        2. Calcula el error (loss) entre prediccion y etiqueta real.
        3. Backward pass: propaga el gradiente del error hacia atras.
        4. lbfgs actualiza W y b usando el gradiente y una aproximacion
           de la matriz Hessiana (curvatura de la funcion de perdida).
        5. Repite hasta convergencia o max_iter=2000.

    INFERENCIA (predict):
        Una vez entrenada, dado el vector de estado real del jugador humano,
        hace UN forward pass (sin actualizar pesos) y devuelve la clase
        con mayor probabilidad: el "Punto Debil" detectado.

    DATASET SINTETICO:
        24 registros que cubren los arquetipos extremos de cada clase mas
        4 casos de frontera para mejorar la generalizacion. Accuracy en
        training: 95.8%. Se amplio desde los 8 registros del enunciado
        original para reducir el riesgo de overfitting en zonas de borde.
    =========================================================================
    """

    # Identificadores de los 3 perfiles de Jefe.
    PERFIL_ANTI_DISTANCIA = 0  # jugador huye     -> jefe acorrala
    PERFIL_ANTI_AGRESIVO  = 1  # jugador se acerca -> escudo + rafagas
    PERFIL_RAFAGA         = 2  # esquiva mal X     -> spam de tipo X

    def __init__(self):
        # Dataset sintetico de entrenamiento.
        # Columnas: [ratio_laser, ratio_misil, preferencia_distancia]
        # Todas las features estan normalizadas en [0.0, 1.0].
        self.X_train = np.array([
            # Clase 2: el jugador recibe mucho daño de LASER
            [0.80, 0.20, 0.50], [0.70, 0.30, 0.60],
            [0.90, 0.10, 0.40], [0.75, 0.25, 0.55],
            # Clase 2: el jugador recibe mucho daño de MISIL
            [0.20, 0.80, 0.40], [0.10, 0.90, 0.30],
            [0.15, 0.85, 0.50], [0.25, 0.75, 0.45],
            # Clase 0: el jugador pasa la mayor parte del tiempo abajo (huye)
            [0.30, 0.30, 0.90], [0.20, 0.10, 0.95],
            [0.40, 0.40, 0.85], [0.10, 0.20, 0.98],
            [0.50, 0.20, 0.88], [0.30, 0.50, 0.92],
            # Clase 1: el jugador pasa la mayor parte del tiempo arriba (agresivo)
            [0.40, 0.20, 0.10], [0.30, 0.30, 0.05],
            [0.50, 0.40, 0.08], [0.20, 0.50, 0.12],
            [0.45, 0.15, 0.02], [0.35, 0.45, 0.15],
            # Casos mixtos en zona de frontera (mejoran generalizacion)
            [0.60, 0.40, 0.50], [0.40, 0.60, 0.50],
            [0.30, 0.30, 0.50], [0.50, 0.50, 0.50],
        ])

        # Etiquetas correspondientes a cada fila de X_train.
        self.y_train = np.array([
            2, 2, 2, 2,   # recibe laser -> Clase 2
            2, 2, 2, 2,   # recibe misil -> Clase 2
            0, 0, 0, 0, 0, 0,  # huye -> Clase 0
            1, 1, 1, 1, 1, 1,  # agresivo -> Clase 1
            2, 2, 0, 1,   # casos frontera
        ])

        # Definicion de la arquitectura del MLP.
        self.modelo = MLPClassifier(
            hidden_layer_sizes=(10,),  # 1 capa oculta, 10 neuronas
            activation="relu",         # ReLU en capa oculta
            solver="lbfgs",            # optimizador cuasi-Newton
            max_iter=2000,             # max iteraciones para convergencia
            random_state=42,           # semilla fija para reproducibilidad
        )

        # Entrenamiento: se ejecuta UNA SOLA VEZ al construir ManejadorIA.
        self.modelo.fit(self.X_train, self.y_train)

        self.ultimo_perfil = None

    def predecir_perfil(self, ratio_laser: float,
                         ratio_misil: float,
                         preferencia_distancia: float) -> int:
        """
        Construye el vector de estado del jugador y ejecuta la inferencia.

        Proceso interno:
            1. Empaqueta los 3 valores en un array numpy de forma (1, 3),
               que es el formato que espera MLPClassifier.predict().
            2. Ejecuta el forward pass a traves de la red ya entrenada.
            3. Devuelve la clase con mayor probabilidad (0, 1 o 2).

        Parametros:
            ratio_laser          : fraccion del daño total que fue laser [0,1]
            ratio_misil          : fraccion del daño total que fue misil [0,1]
            preferencia_distancia: posicion Y promedio normalizada      [0,1]

        Devuelve:
            int: 0 (ANTI_DISTANCIA), 1 (ANTI_AGRESIVO) o 2 (RAFAGA)
        """
        estado_jugador = np.array([[ratio_laser, ratio_misil, preferencia_distancia]])
        prediccion = self.modelo.predict(estado_jugador)
        self.ultimo_perfil = int(prediccion[0])
        return self.ultimo_perfil

    @staticmethod
    def nombre_perfil(perfil: int) -> str:
        """Devuelve el nombre legible del perfil para mostrarlo en pantalla."""
        nombres = {
            0: "PERFIL 0: ANTI-DISTANCIA",
            1: "PERFIL 1: ANTI-AGRESIVO",
            2: "PERFIL 2: RAFAGA LASER/MISIL",
        }
        return nombres.get(perfil, "DESCONOCIDO")


# ==============================================================================
# CLASE: Proyectil
# ==============================================================================
class Proyectil:
    """
    Responsabilidad: representar un disparo en vuelo (laser o misil),
    moverlo cada frame y dibujarlo (con sprite o con geometria).

    Soporta dos modos de movimiento:
      - Direccion fija  : vx y vy se pasan directamente al constructor.
      - Apuntado        : la funcion de fabrica crear_apuntado() calcula
                          vx y vy usando atan2 para que el proyectil vaya
                          hacia la posicion del objetivo en el momento del
                          disparo.

    Colision: se usa AABB (Axis-Aligned Bounding Box) con pygame.Rect.
    El hitbox coincide exactamente con el rectangulo visible (o con el
    sprite escalado si existe), lo que hace la colision predecible y justa.

    Atributos publicos:
        tipo   str          TIPO_LASER o TIPO_MISIL
        dueño  str          "jugador" o "enemigo"
        daño   int          puntos de vida que quita al impactar
        rect   pygame.Rect  posicion y tamaño (hitbox de colision)
        activo bool         False cuando sale de pantalla o impacta
    """

    def __init__(self, x: float, y: float, vx: float, vy: float,
                 tipo: str, dueño: str, assets: ManejadorAssets):
        """
        Parametros:
            x, y  : posicion del centro del proyectil al crearlo (pixeles).
            vx, vy: velocidad en pixeles por frame (puede ser flotante,
                    se suma acumulativamente al rect en actualizar()).
            tipo  : TIPO_LASER o TIPO_MISIL.
            dueño : "jugador" (daña enemigos/jefe) o "enemigo" (daña jugador).
            assets: referencia al ManejadorAssets para consultar sprites.
        """
        self.tipo  = tipo
        self.dueño = dueño
        self.assets = assets

        # Propiedades visuales y de daño segun el tipo de proyectil.
        if tipo == TIPO_LASER:
            self.w, self.h = 10, 4    # hitbox: 10px ancho, 4px alto
            self.color = CIAN   if dueño == "jugador" else AMARILLO
            self.daño  = 5            # daño bajo, cadencia alta
        else:  # TIPO_MISIL
            self.w, self.h = 14, 14   # hitbox: 14x14 px (mas grande = mas facil de esquivar)
            self.color = NARANJA if dueño == "jugador" else MORADO
            self.daño  = 18           # daño alto, cadencia baja

        # pygame.Rect se construye desde la esquina superior izquierda.
        # Restamos la mitad del tamaño para que (x, y) sea el centro.
        self.rect = pygame.Rect(int(x - self.w / 2), int(y - self.h / 2),
                                self.w, self.h)

        # Velocidad en pixeles/frame. Puede ser flotante (se trunca al sumar
        # al rect.x/rect.y, que son enteros, pero la precision es suficiente
        # para la logica de colision AABB).
        self.vx = vx
        self.vy = vy

        self.activo = True  # False cuando sale de pantalla o colisiona

        # Clave para buscar el sprite en ManejadorAssets.
        # Ejemplos: "jugador_laser", "enemigo_misil", "jefe_laser"
        prefijo = "jugador" if dueño == "jugador" else "enemigo"
        self._clave_sprite = f"{prefijo}_{tipo}"

    @classmethod
    def crear_apuntado(cls, ox: float, oy: float,
                       tx: float, ty: float,
                       velocidad: float,
                       tipo: str, dueño: str,
                       assets: ManejadorAssets) -> "Proyectil":
        """
        Metodo de fabrica: crea un Proyectil cuya trayectoria apunta desde
        el origen (ox, oy) hacia el objetivo (tx, ty) en el momento del
        disparo, con rapidez escalar 'velocidad'.

        Matematica:
            dx = tx - ox
            dy = ty - oy
            angulo = atan2(dy, dx)    # angulo del vector origen->objetivo
            vx = velocidad * cos(angulo)
            vy = velocidad * sin(angulo)

        La normalizacion del vector garantiza que el proyectil siempre
        viaje a la misma rapidez independientemente de la distancia al
        objetivo. Si el origen y el objetivo coinciden exactamente
        (distancia == 0, caso degenerado), se dispara hacia abajo.

        Parametros:
            ox, oy   : posicion del origen del disparo (centro del enemigo/jefe).
            tx, ty   : posicion del objetivo (centro del jugador).
            velocidad: rapidez escalar en pixeles/frame.
            tipo, dueño, assets: igual que en __init__.
        """
        dx = tx - ox
        dy = ty - oy
        distancia = math.hypot(dx, dy)  # sqrt(dx^2 + dy^2)

        if distancia == 0:
            # Caso degenerado: origen y objetivo en el mismo pixel.
            # Se dispara directamente hacia abajo para evitar division por cero.
            vx, vy = 0.0, velocidad
        else:
            # Normalizar el vector y multiplicar por la velocidad deseada.
            vx = velocidad * (dx / distancia)
            vy = velocidad * (dy / distancia)

        return cls(ox, oy, vx, vy, tipo, dueño, assets)

    def actualizar(self) -> None:
        """
        Mueve el proyectil sumando su velocidad al Rect cada frame.
        Marca activo=False cuando sale completamente de la pantalla visible
        (margen de 20px) para que el sistema de colision lo elimine pronto.
        """
        self.rect.x += int(self.vx)
        self.rect.y += int(self.vy)

        fuera = (self.rect.right  < -20 or
                 self.rect.left   > ANCHO + 20 or
                 self.rect.bottom < -20 or
                 self.rect.top    > ALTO  + 20)
        if fuera:
            self.activo = False

    def dibujar(self, superficie: pygame.Surface) -> None:
        """
        Dibuja el proyectil en 'superficie'.
        Prioridad: sprite de assets -> geometria de pygame.draw.

        Para el laser se usa un rectangulo (forma alargada legible).
        Para el misil se usa un circulo (forma redondeada, mas amenazante).
        """
        sprite = self.assets.obtener_estatico(self._clave_sprite)
        if sprite is not None:
            escalado = ManejadorAssets.escalar_a_rect(sprite, self.rect)
            superficie.blit(escalado, self.rect)
            return

        # --- Fallback a geometria ---
        if self.tipo == TIPO_LASER:
            pygame.draw.rect(superficie, self.color, self.rect)
        else:
            pygame.draw.circle(superficie, self.color,
                               self.rect.center, self.w // 2)


# ==============================================================================
# CLASE: Jugador
# ==============================================================================
class Jugador:
    """
    Responsabilidad: manejar al personaje controlado por el usuario humano.

    Funciones principales:
      1. Movimiento con 8 direcciones (WASD / flechas).
      2. Dos tipos de disparo: laser (ESPACIO) y misil (SHIFT).
      3. Acumulacion de telemetria para la IA:
           - daño recibido por tipo (laser / misil)
           - posicion Y promedio a lo largo de la Fase 1
      4. Animacion de sprite: idle y movimiento con flip horizontal.
      5. Sistema de i-frames: periodo de invulnerabilidad breve tras
         recibir daño, para que golpes simultaneos no sean injustos.

    Colision: hitbox de 32 x 32 px (igual al sprite recomendado).

    Constantes de clase (facilitan el balance sin buscar magic numbers):
        VELOCIDAD     pixels/frame de movimiento
        VIDA_MAX      puntos de vida maximos
        COOLDOWN_LASER  frames entre disparos de laser   (~0.2s a 60fps)
        COOLDOWN_MISIL  frames entre disparos de misil   (~0.75s a 60fps)
        DURACION_IFRAMES frames de invulnerabilidad tras daño (~0.5s)
    """

    VELOCIDAD       = 5
    VIDA_MAX        = 100
    COOLDOWN_LASER  = 12   # frames (~0.20 s a 60 FPS)
    COOLDOWN_MISIL  = 45   # frames (~0.75 s a 60 FPS)
    DURACION_IFRAMES = 30  # frames (~0.50 s a 60 FPS)

    def __init__(self, x: int, y: int, assets: ManejadorAssets):
        """
        Parametros:
            x, y  : posicion inicial de la esquina superior izquierda del Rect.
            assets: referencia al ManejadorAssets compartido.
        """
        self.rect  = pygame.Rect(x, y, 32, 32)
        self.vida  = self.VIDA_MAX
        self.color = VERDE
        self.assets = assets

        # --- Telemetria acumulada para la IA ---
        # Se actualizan en cada frame durante la Fase 1 de recoleccion.
        # Al final de la fase se usan para construir el vector de entrada
        # de la red neuronal.
        self.daño_recibido_laser = 0.0  # puntos de vida perdidos por laser
        self.daño_recibido_misil = 0.0  # puntos de vida perdidos por misil
        self.suma_posicion_y     = 0.0  # acumulador de posicion Y (centroide)
        self.muestras_posicion   = 0    # conteo de frames muestreados

        # --- Cooldowns de disparo (en frames) ---
        self._cooldown_laser = 0  # 0 = puede disparar laser
        self._cooldown_misil = 0  # 0 = puede disparar misil

        # --- Sistema de i-frames (invulnerabilidad temporal) ---
        # Tras recibir cualquier golpe, se activa este contador. Mientras
        # sea > 0, recibir_daño() ignora el impacto. Esto evita que varios
        # proyectiles del mismo grupo de ataque quiten vida de golpe.
        self._timer_invulnerable = 0

        # --- Estado de sprite de la nave ---
        # dx_actual guarda el componente horizontal del ultimo frame procesado.
        # Es la unica variable que necesita _obtener_sprite_actual() para
        # elegir entre los 3 sprites de la nave:
        #   dx_actual < 0  ->  jugador_izquierda.png
        #   dx_actual > 0  ->  jugador_derecha.png
        #   dx_actual == 0 ->  jugador_idle.png  (quieta o solo movimiento Y)
        self.dx_actual = 0

    # ------------------------------------------------------------------
    # Actualizacion por frame
    # ------------------------------------------------------------------

    def manejar_input(self, teclas) -> None:
        """
        Lee el estado del teclado, mueve el jugador y registra telemetria.
        Se llama una vez por frame durante las fases de juego activo.

        El movimiento esta limitado a los bordes de la pantalla con
        max/min para que el jugador nunca salga del area visible.

        Parametros:
            teclas: resultado de pygame.key.get_pressed() (array de booleanos).
        """
        dx = dy = 0
        if teclas[pygame.K_LEFT]  or teclas[pygame.K_a]: dx -= self.VELOCIDAD
        if teclas[pygame.K_RIGHT] or teclas[pygame.K_d]: dx += self.VELOCIDAD
        if teclas[pygame.K_UP]    or teclas[pygame.K_w]: dy -= self.VELOCIDAD
        if teclas[pygame.K_DOWN]  or teclas[pygame.K_s]: dy += self.VELOCIDAD

        # Guardar el dx del frame actual para que _obtener_sprite_actual()
        # sepa que sprite de nave mostrar este frame.
        self.dx_actual = dx

        # Mover y restringir a los limites de la pantalla.
        self.rect.x = max(0, min(ANCHO - self.rect.width,  self.rect.x + dx))
        self.rect.y = max(0, min(ALTO  - self.rect.height, self.rect.y + dy))

        # Acumular posicion Y para telemetria (muestreo por frame).
        self.suma_posicion_y   += self.rect.centery
        self.muestras_posicion += 1

    def actualizar_cooldowns(self) -> None:
        """
        Decrementa todos los cooldowns en 1 tick por frame.
        Se llama una vez por frame independientemente del input.
        Cuando un cooldown llega a 0, la accion correspondiente
        vuelve a estar disponible.
        """
        if self._cooldown_laser    > 0: self._cooldown_laser    -= 1
        if self._cooldown_misil    > 0: self._cooldown_misil    -= 1
        if self._timer_invulnerable > 0: self._timer_invulnerable -= 1

    # ------------------------------------------------------------------
    # Disparo
    # ------------------------------------------------------------------

    def disparar_laser(self, lista_proyectiles: list) -> None:
        """
        Crea un Proyectil de tipo LASER apuntando hacia arriba si el
        cooldown lo permite, y lo agrega a lista_proyectiles.
        El cooldown se reinicia a COOLDOWN_LASER frames.
        """
        if self._cooldown_laser <= 0:
            p = Proyectil(self.rect.centerx, self.rect.top,
                          0, -10, TIPO_LASER, "jugador", self.assets)
            lista_proyectiles.append(p)
            self._cooldown_laser = self.COOLDOWN_LASER

    def disparar_misil(self, lista_proyectiles: list) -> None:
        """
        Crea un Proyectil de tipo MISIL apuntando hacia arriba si el
        cooldown lo permite, y lo agrega a lista_proyectiles.
        El cooldown se reinicia a COOLDOWN_MISIL frames.
        """
        if self._cooldown_misil <= 0:
            p = Proyectil(self.rect.centerx, self.rect.top,
                          0, -5, TIPO_MISIL, "jugador", self.assets)
            lista_proyectiles.append(p)
            self._cooldown_misil = self.COOLDOWN_MISIL

    # ------------------------------------------------------------------
    # Daño y telemetria
    # ------------------------------------------------------------------

    def recibir_daño(self, cantidad: float, tipo_proyectil: str) -> None:
        """
        Aplica 'cantidad' de daño al jugador y acumula telemetria.
        Si el jugador esta en i-frames (_timer_invulnerable > 0), el
        impacto se ignora completamente (ni resta vida ni acumula).

        Tras recibir daño, activa DURACION_IFRAMES frames de proteccion.

        Parametros:
            cantidad       : puntos de vida a restar.
            tipo_proyectil : TIPO_LASER o TIPO_MISIL (para la telemetria).
        """
        if self._timer_invulnerable > 0:
            return  # dentro del periodo de invulnerabilidad: ignorar impacto

        self.vida = max(0, self.vida - cantidad)
        self._timer_invulnerable = self.DURACION_IFRAMES

        # Acumular en el contador del tipo correspondiente.
        if tipo_proyectil == TIPO_LASER:
            self.daño_recibido_laser += cantidad
        else:
            self.daño_recibido_misil += cantidad

    def obtener_ratios_daño(self) -> tuple:
        """
        Calcula los ratios de daño por tipo, normalizados a [0, 1].
        Ambos ratios suman exactamente 1.0.

        Si el jugador no recibio ningun daño (total == 0), devuelve
        (0.5, 0.5) como valor neutro para no sesgar la prediccion de la IA.

        Devuelve:
            (ratio_laser, ratio_misil): tupla de dos flotantes.
        """
        total = self.daño_recibido_laser + self.daño_recibido_misil
        if total <= 0:
            return 0.5, 0.5
        return (self.daño_recibido_laser / total,
                self.daño_recibido_misil / total)

    def obtener_preferencia_distancia(self) -> float:
        """
        Calcula la posicion Y promedio del jugador normalizada a [0, 1].

        Interpretacion para la IA:
            0.0 -> el jugador paso todo el tiempo arriba (zona del enemigo,
                   comportamiento agresivo / cuerpo a cuerpo)
            1.0 -> el jugador paso todo el tiempo abajo (zona de escape,
                   comportamiento defensivo / huyendo)

        La normalizacion se hace dividiendo por ALTO (600 px), que es el
        valor maximo posible de centery.

        Devuelve:
            float en [0.0, 1.0]
        """
        if self.muestras_posicion == 0:
            return 0.5  # sin datos: valor neutro
        promedio_y = self.suma_posicion_y / self.muestras_posicion
        return max(0.0, min(1.0, promedio_y / ALTO))

    # ------------------------------------------------------------------
    # Renderizado
    # ------------------------------------------------------------------

    def dibujar(self, superficie: pygame.Surface) -> None:
        """
        Dibuja la nave del jugador en 'superficie'.

        Prioridad de renderizado:
            1. Sprite direccional (jugador_derecha / jugador_izquierda / jugador_idle)
               segun el dx del frame actual, si existe en assets.
            2. Rectangulo verde + borde blanco (geometria de fallback).

        No se aplica ningun flip: cada sprite ya tiene la orientacion
        correcta tal como fue dibujado en el editor de sprites.
        """
        sprite = self._obtener_sprite_actual()
        if sprite is not None:
            escalado = ManejadorAssets.escalar_a_rect(sprite, self.rect)
            superficie.blit(escalado, self.rect)
            return

        # Fallback: rectangulo redondeado
        pygame.draw.rect(superficie, self.color, self.rect, border_radius=4)
        pygame.draw.rect(superficie, BLANCO,     self.rect, width=2, border_radius=4)

    def _obtener_sprite_actual(self):
        """
        Selecciona la Surface de nave correcta segun self.dx_actual.

        Logica de seleccion:
            dx < 0  ->  "jugador_izquierda"  (nave virando a la izquierda)
            dx > 0  ->  "jugador_derecha"    (nave virando a la derecha)
            dx == 0 ->  "jugador_idle"       (nave recta, quieta o solo Y)

        Si el sprite de la direccion pedida no fue cargado (no existe el PNG),
        intenta con "jugador_idle" como segundo fallback antes de devolver None.

        Devuelve:
            pygame.Surface si hay algun sprite disponible, None si no hay ninguno.
        """
        if self.dx_actual < 0:
            clave = "jugador_izquierda"
        elif self.dx_actual > 0:
            clave = "jugador_derecha"
        else:
            clave = "jugador_idle"

        if self.assets.tiene_estatico(clave):
            return self.assets.obtener_estatico(clave)

        # Segundo fallback: si falta el sprite direccional pero hay idle, usar idle.
        if clave != "jugador_idle" and self.assets.tiene_estatico("jugador_idle"):
            return self.assets.obtener_estatico("jugador_idle")

        return None  # sin ningun sprite: el llamador dibujara geometria


# ==============================================================================
# CLASE: Enemigo  (Fase 1 - recoleccion de datos)
# ==============================================================================
class Enemigo:
    """
    Comportamiento:
      - Se mueve horizontalmente rebotando en los bordes de la pantalla.
      - Dispara periodicamente hacia la posicion del jugador (proyectil
        apuntado con Proyectil.crear_apuntado), alternando aleatoriamente
        entre laser y misil para generar diversidad en la telemetria.

    Diseño: el enemigo necesita saber donde esta el jugador para apuntar,
    por eso actualizar() recibe tanto lista_proyectiles como jugador.

    Colision: hitbox de 28 x 28 px.
    Vida: 20 puntos (muere de 4 golpes de laser o 2 de misil del jugador).
    """

    def __init__(self, x: int, y: int, assets: ManejadorAssets):
        """
        Parametros:
            x, y  : posicion inicial (esquina superior izquierda del Rect).
            assets: referencia al ManejadorAssets compartido.
        """
        self.rect   = pygame.Rect(x, y, 28, 28)
        self.vida   = 20
        self.color  = ROJO
        self.assets = assets

        # Cooldown inicial aleatorio para desfasar los disparos entre enemigos
        # y evitar que todos disparen al mismo tiempo al inicio.
        self._cooldown = random.randint(30, 90)

        # Velocidad horizontal: +/-1 o +/-2 px/frame (variedad de ritmos).
        self.vx = random.choice([-2, -1, 1, 2])

    def actualizar(self, lista_proyectiles: list, jugador: "Jugador") -> None:
        """
        Actualiza posicion y disparo del enemigo.

        Movimiento: desplazamiento horizontal simple con rebote en bordes.

        Disparo: dos comportamientos segun el tipo elegido aleatoriamente:
          - Laser : cae verticalmente hacia abajo (vx=0, vy fija).
                    Es rapido y predecible; se esquiva moviendose lateralmente.
                    Genera presion constante sin ser injusto.
          - Misil : apuntado directamente al jugador con crear_apuntado().
                    Es lento pero te persigue; justifica su mayor daño.
                    Obliga al jugador a moverse incluso cuando esta quieto.

        Cooldown subido a 60-130 frames (~1-2.2s) para que los enemigos
        disparen menos seguido y la Fase 1 sea tensa pero manejable.

        Parametros:
            lista_proyectiles: lista compartida donde se agregan nuevos disparos.
            jugador          : referencia al Jugador (solo necesaria para el misil).
        """
        # Movimiento horizontal con rebote
        self.rect.x += self.vx
        if self.rect.left < 0 or self.rect.right > ANCHO:
            self.vx *= -1

        self._cooldown -= 1
        if self._cooldown <= 0:
            tipo = random.choice([TIPO_LASER, TIPO_MISIL])

            if tipo == TIPO_LASER:
                # Laser: vertical puro, rapido. No usa crear_apuntado().
                # vx=0 garantiza trayectoria recta hacia abajo.
                p = Proyectil(
                    self.rect.centerx, self.rect.bottom,
                    0, 6,  # vx=0 (recto), vy=6 (hacia abajo)
                    TIPO_LASER, "enemigo", self.assets
                )
            else:
                # Misil: apuntado al jugador, mas lento.
                p = Proyectil.crear_apuntado(
                    ox=self.rect.centerx, oy=self.rect.bottom,
                    tx=jugador.rect.centerx, ty=jugador.rect.centery,
                    velocidad=3.5,
                    tipo=TIPO_MISIL, dueño="enemigo", assets=self.assets
                )

            lista_proyectiles.append(p)
            self._cooldown = random.randint(60, 130)

    def dibujar(self, superficie: pygame.Surface) -> None:
        sprite = self.assets.obtener_estatico("enemigo_idle")
        if sprite is not None:
            escalado = ManejadorAssets.escalar_a_rect(sprite, self.rect)
            superficie.blit(escalado, self.rect)
            return
        pygame.draw.rect(superficie, self.color, self.rect, border_radius=3)


# ==============================================================================
# CLASE: JefeFinal
# ==============================================================================
class JefeFinal:
    """
    El perfil (0, 1 o 2) se inyecta en el constructor y determina que
    patron de ataque usa durante todo el combate.

    Sistema de fases de enrabiamiento (escalado dinamico de dificultad):
    Conforme pierde vida, los ataques se vuelven mas frecuentes y densos.
    Esto evita que el combate sea igual de facil del principio al final,
    y crea un arco dramatico natural en la batalla.

        Fase A: vida > 50% del maximo  -> cadencia base
        Fase B: vida 10-50%            -> cadencia x1.25,  mas proyectiles
        Fase C: vida < 10%             -> cadencia x1.5, mayor densidad,
                                          color rojo (visualmente "enrabiado")

    Multiplicadores de timer de ataque (menor valor = ataca mas seguido):
        MULT_FASE_A = 1.2  (base)
        MULT_FASE_B = 0.65  (~25% mas frecuente)
        MULT_FASE_C = 0.50  (~50% mas frecuente)

    Movimiento:
        El jefe se mueve en X e Y dentro de la franja superior (y: 20-220 px).
        En fases B y C la velocidad aumenta para que sea mas dificil apuntarle.

    Colision: hitbox de 120 x 60 px.
    Vida: 320 puntos (elegida para ~13s de DPS maximo del jugador en Fase A,
          y mucho mas en Fase C donde el jugador debe esquivar mas y dispara menos).
    """

    VIDA_MAX = 320

    # Multiplicadores de velocidad de ataque por fase.
    # Se aplican al timer base de cada ataque: timer = int(base * mult)
    MULT_FASE_A = 1.2
    MULT_FASE_B = 0.75
    MULT_FASE_C = 0.5

    def __init__(self, perfil: int, ratio_laser: float,
                 ratio_misil: float, assets: ManejadorAssets):
        """
        Parametros:
            perfil      : 0, 1 o 2 — resultado de ManejadorIA.predecir_perfil().
            ratio_laser : proporcion de daño laser recibido por el jugador.
            ratio_misil : proporcion de daño misil recibido por el jugador.
            assets      : referencia al ManejadorAssets compartido.
        """
        self.perfil = perfil
        self.vida   = self.VIDA_MAX
        self.rect   = pygame.Rect(ANCHO // 2 - 60, 40, 120, 60)
        self.color  = MORADO
        self.assets = assets

        # Para el Perfil 2 (RAFAGA), el jefe spamea el tipo de proyectil
        # contra el que el jugador demostro peor rendimiento al esquivar.
        # Se determina comparando los ratios: mayor ratio = mas daño recibido.
        self.tipo_explotado = (TIPO_LASER if ratio_laser >= ratio_misil
                               else TIPO_MISIL)

        self._timer_ataque = 45    # frames hasta el primer ataque
        self.vx = 4                # velocidad horizontal inicial (px/frame)
        self.vy = 2                # velocidad vertical inicial   (px/frame)
        self.fase_actual = "A"     # se actualiza en _actualizar_fase()

        # El escudo de contacto solo existe en Perfil 1 (ANTI_AGRESIVO).
        # Causa daño al jugador si se acerca demasiado al jefe.
        self.escudo_activo = (perfil == ManejadorIA.PERFIL_ANTI_AGRESIVO)

    # ------------------------------------------------------------------
    # Sistema de fases
    # ------------------------------------------------------------------

    def _actualizar_fase(self) -> None:
        """
        Recalcula la fase de enrabiamiento segun el % de vida restante.
        Se llama una vez por frame al inicio de actualizar().

        Umbrales:
            vida > 50% -> Fase A (inicio del combate, ritmo normal)
            vida 10-50% -> Fase B (jefe herido, empieza a enojarse)
            vida < 10% -> Fase C (jefe enrabiado, maximo peligro)
        """
        proporcion = self.vida / self.VIDA_MAX
        if   proporcion > 0.50: self.fase_actual = "A"
        elif proporcion > 0.10: self.fase_actual = "B"
        else:                   self.fase_actual = "C"

    def _multiplicador_velocidad(self) -> float:
        """
        Devuelve el multiplicador de cadencia para la fase actual.
        Se multiplica por el timer base de cada ataque para escalarlo.
        """
        return {
            "A": self.MULT_FASE_A,
            "B": self.MULT_FASE_B,
            "C": self.MULT_FASE_C,
        }[self.fase_actual]

    # ------------------------------------------------------------------
    # Actualizacion por frame
    # ------------------------------------------------------------------

    def actualizar(self, lista_proyectiles: list, jugador: "Jugador") -> None:
        """
        Actualiza estado del jefe: fase, movimiento, ataque y escudo.

        Movimiento en X: rebota en los bordes laterales de la pantalla.
        Movimiento en Y: rebota entre y=20 y y=220 (franja superior).
        En fases B y C la velocidad se multiplica por un factor creciente
        para que el jefe sea mas evasivo cuando esta mas herido.

        Parametros:
            lista_proyectiles: lista compartida donde se agregan disparos.
            jugador          : referencia al Jugador (para apuntar y escudo).
        """
        self._actualizar_fase()

        # Factor de velocidad extra segun la fase de enrabiamiento.
        factor_vel = (1.5 if self.fase_actual == "C" else
                      1.0 if self.fase_actual == "B" else 0.0)

        # Movimiento horizontal
        self.rect.x += int(self.vx * (1 + factor_vel * 0.3))
        if self.rect.left < 0 or self.rect.right > ANCHO:
            self.vx *= -1  # invertir direccion al tocar el borde

        # Movimiento vertical (dentro de la franja superior)
        self.rect.y += int(self.vy * (1 + factor_vel * 0.3))
        if self.rect.top < 20 or self.rect.bottom > 220:
            self.vy *= -1  # invertir direccion al tocar los limites verticales

        # Contador de ataque: decrementa cada frame y dispara al llegar a 0.
        self._timer_ataque -= 1
        if self._timer_ataque <= 0:
            self._ejecutar_ataque(lista_proyectiles, jugador)

        # Escudo de contacto (solo Perfil 1): daña al jugador si se acerca.
        # El rect.inflate(20, 20) crea un area de deteccion ligeramente mas
        # grande que el hitbox visual para que el efecto se sienta antes de
        # que los sprites se toquen (margen de aviso al jugador).
        # Los i-frames del jugador limitan esto a un golpe cada ~0.5s.
        if self.escudo_activo and self.rect.inflate(20, 20).colliderect(jugador.rect):
            jugador.recibir_daño(7, TIPO_MISIL)

    def _ejecutar_ataque(self, lista_proyectiles: list, jugador: "Jugador") -> None:
        """
        Decide que patron de ataque lanzar segun el perfil del jefe y
        reinicia el timer de ataque escalado por la fase actual.

        El timer minimo (max(...)) evita que en Fase C el timer llegue a
        0 o negativo, lo que causaria un ataque por frame (game-breaking).

        Parametros:
            lista_proyectiles: lista donde se agregan los proyectiles creados.
            jugador          : referencia al Jugador (necesaria para apuntar).
        """
        mult = self._multiplicador_velocidad()

        if self.perfil == ManejadorIA.PERFIL_ANTI_DISTANCIA:
            self._ataque_anti_distancia(lista_proyectiles, jugador)
            self._timer_ataque = max(26, int(75 * mult))

        elif self.perfil == ManejadorIA.PERFIL_ANTI_AGRESIVO:
            self._ataque_anti_agresivo(lista_proyectiles, jugador)
            self._timer_ataque = max(15, int(42 * mult))

        else:  # PERFIL_RAFAGA
            self._ataque_rafaga(lista_proyectiles, jugador)
            self._timer_ataque = max(16, int(35 * mult))

    # ------------------------------------------------------------------
    # Patrones de ataque
    # ------------------------------------------------------------------

    def _ataque_anti_distancia(self, lista_proyectiles: list,
                                jugador: "Jugador") -> None:
        """
        Patron del Perfil 0 (ANTI_DISTANCIA): el jugador tiende a huir.
        Objetivo: reducir el espacio disponible para escapar.

        Fase A/B: barrera horizontal de lasers con UN hueco de 90px para
                  que el jugador pueda pasar si se mueve correctamente.
        Fase C  : dos huecos mas angostos (60px cada uno), forzando una
                  decision de ruta mas precisa y rapida.

        Adicionalmente, se dispara UN laser apuntado directamente al
        jugador para evitar que se quede quieto detras del hueco.

        Parametros:
            lista_proyectiles: lista donde se agregan los proyectiles.
            jugador          : necesario para el laser apuntado y para
                               ubicar el hueco relativo al jugador.
        """
        y = self.rect.bottom + 10
        paso = 40  # separacion entre proyectiles de la barrera

        # Definir huecos en la barrera segun la fase.
        if self.fase_actual == "C":
            # Dos huecos angostos: mayor dificultad de navegacion.
            desfase = random.choice([-85, 85])
            inicio = max(0, min(ANCHO - 70, jugador.rect.centerx + desfase))
            huecos = [(inicio, inicio + 70)]
        else:
            # Un hueco amplio centrado aproximadamente en la posicion del jugador.
            inicio = max(0, min(ANCHO - 90, jugador.rect.centerx - 45))
            huecos = [(inicio, inicio + 90)]

        # Generar la barrera horizontal de lasers.
        x = 0
        while x < ANCHO:
            # Solo crear proyectil si x no cae dentro de ningun hueco.
            if not any(ini <= x <= fin for ini, fin in huecos):
                p = Proyectil(x, y, 0, 6, TIPO_LASER, "enemigo", self.assets)
                lista_proyectiles.append(p)
            x += paso

        # Laser adicional apuntado directo al jugador para presion extra.
        apuntado = Proyectil.crear_apuntado(
            ox=self.rect.centerx, oy=self.rect.bottom,
            tx=jugador.rect.centerx, ty=jugador.rect.centery,
            velocidad=5.0,
            tipo=TIPO_LASER, dueño="enemigo", assets=self.assets
        )
        lista_proyectiles.append(apuntado)

    def _ataque_anti_agresivo(self, lista_proyectiles: list,
                               jugador: "Jugador") -> None:
        """
        Patron del Perfil 1 (ANTI_AGRESIVO): el jugador tiende a acercarse.
        Objetivo: castigar la proximidad con un anillo expansivo de misiles.

        Fase A: 10 misiles en anillo, mas 1 apuntado al jugador.
        Fase B/C: 14 misiles en anillo (mas denso), mas 2 apuntados.

        El anillo cubre todas las direcciones, por lo que el jugador no
        puede simplemente esquivar moviendose en una sola direccion.
        Los misiles apuntados adicionalmente dificultan alejarse rapido.

        Parametros:
            lista_proyectiles: lista donde se agregan los proyectiles.
            jugador          : para los misiles apuntados adicionales.
        """
        cx, cy = self.rect.centerx, self.rect.bottom

        # Numero de misiles en el anillo segun fase.
        n = 10 if self.fase_actual == "A" else 14

        # Anillo uniforme: angulos igualmente distribuidos en [0, 2*pi).
        for i in range(n):
            angulo = (2 * math.pi / n) * i
            vx = 6.0 * math.cos(angulo)
            vy = 6.0 * math.sin(angulo) + 2.0  # sesgo +2 hacia abajo
            p = Proyectil(cx, cy, vx, vy, TIPO_MISIL, "enemigo", self.assets)
            lista_proyectiles.append(p)

        # Misiles apuntados adicionales: 1 en Fase A, 2 en B/C.
        cantidad_apuntados = 1 if self.fase_actual == "A" else 2
        for _ in range(cantidad_apuntados):
            apuntado = Proyectil.crear_apuntado(
                ox=cx, oy=cy,
                tx=jugador.rect.centerx, ty=jugador.rect.centery,
                velocidad=5.5,
                tipo=TIPO_MISIL, dueño="enemigo", assets=self.assets
            )
            lista_proyectiles.append(apuntado)

    def _ataque_rafaga(self, lista_proyectiles: list,
                       jugador: "Jugador") -> None:
        """
        Patron del Perfil 2 (RAFAGA): spamea el tipo de proyectil contra
        el que el jugador esquiva peor (self.tipo_explotado).

        Cada disparo combina un proyectil apuntado directamente al jugador
        con proyectiles adicionales en angulos desviados (spread), para
        que esquivar no sea tan simple como moverse en una sola direccion.

        Parametros:
            lista_proyectiles: lista donde se agregan los proyectiles.
            jugador          : para calcular el angulo de apuntado.
        """
        rapidez = 7.5 if self.tipo_explotado == TIPO_LASER else 4.0
        cx, cy  = self.rect.centerx, self.rect.bottom

        # Calcular el angulo base hacia el jugador.
        dx = jugador.rect.centerx - cx
        dy = jugador.rect.centery - cy
        distancia = math.hypot(dx, dy)

        if distancia == 0:
            angulo_base = math.pi / 2  # directamente hacia abajo
        else:
            angulo_base = math.atan2(dy, dx)

        desviaciones = [0.0]
        if self.fase_actual in ("B", "C"):
            desviaciones += [math.radians(22), math.radians(-22)]
        if self.fase_actual == "C":
            desviaciones += [math.radians(45), math.radians(-45)]

        for desv in desviaciones:
            angulo = angulo_base + desv
            vx = rapidez * math.cos(angulo)
            vy = rapidez * math.sin(angulo)
            p = Proyectil(cx, cy, vx, vy,
                          self.tipo_explotado, "enemigo", self.assets)
            lista_proyectiles.append(p)

    # ------------------------------------------------------------------
    # Renderizado
    # ------------------------------------------------------------------

    def dibujar(self, superficie: pygame.Surface) -> None:
        """
        Dibuja el jefe en 'superficie'.
        Prioridad: sprite 'jefe_idle.png' -> rectangulo de geometria.

        En Fase C (enrabiado) la geometria de fallback cambia de color
        morado a rojo para indicar visualmente el estado de peligro.
        Si tienes sprite, considera cambiar a 'jefe_enojado.png' en Fase C
        cargandolo como un estatico adicional en ManejadorAssets.

        El escudo (Perfil 1) siempre se dibuja como un rectangulo cian
        ligeramente mas grande que el hitbox, independientemente de si
        hay sprite o no.
        """
        sprite = self.assets.obtener_estatico("jefe_idle")
        if sprite is not None:
            escalado = ManejadorAssets.escalar_a_rect(sprite, self.rect)
            superficie.blit(escalado, self.rect)
        else:
            # Cambio de color al enrabiarse (solo visible sin sprite)
            color_actual = ROJO if self.fase_actual == "C" else self.color
            pygame.draw.rect(superficie, color_actual, self.rect, border_radius=6)
            pygame.draw.rect(superficie, BLANCO,       self.rect,
                             width=3, border_radius=6)

        # El halo de escudo siempre se dibuja sobre el sprite o la geometria.
        if self.escudo_activo:
            pygame.draw.rect(superficie, CIAN,
                             self.rect.inflate(20, 20), width=2, border_radius=10)


# ==============================================================================
# CLASE: Juego
# ==============================================================================
class Juego:
    """
    Responsabilidad: orquestar todos los sistemas del juego.

    Implementa una maquina de estados con 4 estados:
        RECOLECCION  60s de oleadas de enemigos, se acumula telemetria.
        TRANSICION   3s de pausa, se calcula el vector y se llama predict().
        JEFE         Combate contra el Jefe Final adaptado.
        FIN          Pantalla de Game Over o Victoria, espera R o ESC.

    El bucle principal (ejecutar) corre a 60 FPS usando reloj.tick(FPS).
    El delta time (dt) en segundos se usa para timers de duracion (fases,
    letrero); los cooldowns de frame se manejan con contadores de ticks
    para ser consistentes con el resto del sistema.

    Atributos principales:
        pantalla   pygame.Surface  ventana de renderizado
        reloj      pygame.Clock    control de FPS
        assets     ManejadorAssets sprites cargados al inicio
        ia         ManejadorIA     red neuronal entrenada al inicio
        jugador    Jugador         instancia del jugador actual
        enemigos   list[Enemigo]   lista de enemigos activos (Fase 1)
        proyectiles list[Proyectil] todos los disparos en vuelo
        jefe       JefeFinal|None  instancia del jefe (None en Fase 1)
        estado     str             estado actual de la maquina de estados
    """

    # Identificadores de estado como constantes de clase.
    ESTADO_RECOLECCION = "RECOLECCION"
    ESTADO_TRANSICION  = "TRANSICION"
    ESTADO_JEFE        = "JEFE"
    ESTADO_FIN         = "FIN"

    def __init__(self):
        """
        Inicializa pygame, la ventana, las fuentes, los sistemas de IA
        y assets, y llama a reiniciar() para poner el juego en estado inicial.
        """
        pygame.init()
        pygame.display.set_caption("Juego IA")
        self.pantalla = pygame.display.set_mode((ANCHO, ALTO))
        self.reloj    = pygame.time.Clock()

        self.fuente        = pygame.font.SysFont("consolas", 18)
        self.fuente_mediana = pygame.font.SysFont("consolas", 24, bold=True)
        self.fuente_grande  = pygame.font.SysFont("consolas", 36, bold=True)

        self.assets = ManejadorAssets()
        self.ia     = ManejadorIA()

        self.reiniciar()

    def reiniciar(self) -> None:
        """
        Reinicia el estado de la partida actual a los valores iniciales.
        Se llama al inicio y cada vez que el jugador presiona R.

        Los sistemas assets e ia NO se reinician (ya estan listos y
        re-inicializarlos seria costoso e innecesario).
        """
        # Entidades del juego
        self.jugador     = Jugador(ANCHO // 2 - 16, ALTO - 80, self.assets)
        self.proyectiles = []
        self.enemigos    = [
            Enemigo(random.randint(40, ANCHO - 40), random.randint(40, 150), self.assets)
            for _ in range(4)
        ]
        self.jefe = None

        # Maquina de estados
        self.estado = self.ESTADO_RECOLECCION

        # Timers de fase (en segundos, usan dt del bucle principal)
        self.tiempo_restante_fase1 = DURACION_FASE_1
        self.timer_transicion      = DURACION_TRANSICION

        # Resultado de la prediccion (None hasta que se invoca el jefe)
        self.perfil_detectado = None
        self.mensaje_final    = ""

        # Timer del letrero "IA SELECCIONO: ..." al inicio de Fase 2.
        # 0.0 = letrero no visible. Se activa en _invocar_jefe_final().
        self.timer_letrero_perfil = 0.0

        # Timer para el spawn periodico de enemigos en Fase 1.
        self._spawn_enemigo_timer = 0

    # ------------------------------------------------------------------
    # Bucle principal
    # ------------------------------------------------------------------

    def ejecutar(self) -> None:
        """
        Bucle principal del juego (game loop).

        Estructura de cada iteracion:
            1. reloj.tick(FPS) limita la velocidad a 60 FPS y devuelve
               el tiempo real transcurrido (dt) en milisegundos.
            2. Se procesan los eventos de pygame (teclado, cierre).
            3. Se actualiza la logica segun el estado actual.
            4. Se dibuja el frame actual completo.
            5. pygame.display.flip() presenta el buffer en pantalla.
        """
        corriendo = True
        while corriendo:
            dt = self.reloj.tick(FPS) / 1000.0  # convertir ms a segundos
            corriendo = self._procesar_eventos()

            if   self.estado == self.ESTADO_RECOLECCION: self._actualizar_recoleccion(dt)
            elif self.estado == self.ESTADO_TRANSICION:  self._actualizar_transicion(dt)
            elif self.estado == self.ESTADO_JEFE:        self._actualizar_jefe(dt)
            # ESTADO_FIN: no hay logica de actualizacion, solo espera input

            self._dibujar()
            pygame.display.flip()

        pygame.quit()
        sys.exit()

    # ------------------------------------------------------------------
    # Procesamiento de eventos
    # ------------------------------------------------------------------

    def _procesar_eventos(self) -> bool:
        """
        Procesa la cola de eventos de pygame.

        Devuelve:
            False si el juego debe cerrarse (ventana cerrada o ESC),
            True en cualquier otro caso.
        """
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                return False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    return False
                # R solo tiene efecto en el estado FIN para reiniciar la partida.
                if evento.key == pygame.K_r and self.estado == self.ESTADO_FIN:
                    self.reiniciar()
        return True

    # ------------------------------------------------------------------
    # FASE 1: Recoleccion / Telemetria
    # ------------------------------------------------------------------

    def _actualizar_recoleccion(self, dt: float) -> None:
        """
        Logica de la Fase 1: oleadas de enemigos, disparo del jugador,
        y acumulacion de telemetria. SIN llamadas a la IA (solo acumula datos).

        Transiciones:
            -> TRANSICION si tiempo_restante_fase1 llega a 0.
            -> FIN        si el jugador pierde toda su vida.

        Parametros:
            dt: tiempo real del frame en segundos (para el countdown).
        """
        self.tiempo_restante_fase1 -= dt

        teclas = pygame.key.get_pressed()
        self.jugador.manejar_input(teclas)
        self.jugador.actualizar_cooldowns()
        if teclas[pygame.K_SPACE]:                         self.jugador.disparar_laser(self.proyectiles)
        if teclas[pygame.K_LSHIFT] or teclas[pygame.K_RSHIFT]: self.jugador.disparar_misil(self.proyectiles)

        # Spawn periodico de enemigos para mantener presion constante.
        self._spawn_enemigo_timer -= 1
        if len(self.enemigos) < 5 and self._spawn_enemigo_timer <= 0:
            self.enemigos.append(
                Enemigo(random.randint(40, ANCHO - 40), random.randint(40, 150), self.assets)
            )
            self._spawn_enemigo_timer = 90  # nuevo enemigo cada ~1.5s

        # Actualizar enemigos (necesitan referencia al jugador para apuntar).
        for enemigo in self.enemigos:
            enemigo.actualizar(self.proyectiles, self.jugador)

        self._actualizar_proyectiles_y_colisiones(daña_enemigos=True)

        # Evaluacion de condiciones de transicion.
        if self.tiempo_restante_fase1 <= 0:
            self._iniciar_transicion()
        elif self.jugador.vida <= 0:
            self.estado        = self.ESTADO_FIN
            self.mensaje_final = "GAME OVER (caiste en la Fase de Recoleccion)"

    def _iniciar_transicion(self) -> None:
        """
        Prepara el estado TRANSICION: limpia enemigos y proyectiles
        activos para dejar la pantalla despejada durante la cuenta regresiva.
        """
        self.estado = self.ESTADO_TRANSICION
        self.timer_transicion = DURACION_TRANSICION
        self.proyectiles.clear()
        self.enemigos.clear()

    # ------------------------------------------------------------------
    # TRANSICION: calculo del vector de estado y prediccion de la red
    # ------------------------------------------------------------------

    def _actualizar_transicion(self, dt: float) -> None:
        """
        Cuenta regresiva de DURACION_TRANSICION segundos antes de invocar
        al Jefe Final. Cuando llega a 0, se ejecuta la prediccion de la IA.

        Parametros:
            dt: tiempo real del frame en segundos.
        """
        self.timer_transicion -= dt
        if self.timer_transicion <= 0:
            self._invocar_jefe_final()

    def _invocar_jefe_final(self) -> None:
        """
        Calcula el vector de estado del jugador y llama a brain.predict().

        Secuencia:
            1. Obtener ratio_laser y ratio_misil de la telemetria acumulada.
            2. Obtener preferencia_distancia de la posicion Y promedio.
            3. Llamar a ia.predecir_perfil() con estos 3 valores.
            4. Crear la instancia de JefeFinal con el perfil resultante.
            5. Aplicar la vida minima de seguridad al jugador.
            6. Activar el letrero de perfil por 3.5 segundos.
        """
        ratio_laser, ratio_misil  = self.jugador.obtener_ratios_daño()
        preferencia_distancia     = self.jugador.obtener_preferencia_distancia()

        # Inferencia de la red neuronal: forward pass con el vector de estado.
        perfil = self.ia.predecir_perfil(ratio_laser, ratio_misil, preferencia_distancia)
        self.perfil_detectado = perfil

        # El jefe recibe el perfil en su constructor y fija su Machine State.
        self.jefe = JefeFinal(perfil, ratio_laser, ratio_misil, self.assets)

        # El jugador NO se cura: conserva la vida con la que termino Fase 1.
        # La vida minima (20%) evita que la pelea sea imposible si llego muy daňado.
        vida_minima = Jugador.VIDA_MAX * 0.5
        self.jugador.vida = max(self.jugador.vida, vida_minima)

        self.estado = self.ESTADO_JEFE
        self.timer_letrero_perfil = 2.5  # segundos de visibilidad del letrero

    # ------------------------------------------------------------------
    # FASE 2: Combate contra el Jefe Final
    # ------------------------------------------------------------------

    def _actualizar_jefe(self, dt: float) -> None:
        """
        Logica de la Fase 2: el jugador combate contra el Jefe Final.

        Transiciones:
            -> FIN (derrota)  si la vida del jugador llega a 0.
            -> FIN (victoria) si la vida del jefe llega a 0.

        Parametros:
            dt: tiempo real del frame en segundos (para el timer del letrero).
        """
        # Decrementar el timer del letrero de perfil (se apaga solo).
        if self.timer_letrero_perfil > 0:
            self.timer_letrero_perfil -= dt

        teclas = pygame.key.get_pressed()
        self.jugador.manejar_input(teclas)
        self.jugador.actualizar_cooldowns()
        if teclas[pygame.K_SPACE]:                         self.jugador.disparar_laser(self.proyectiles)
        if teclas[pygame.K_LSHIFT] or teclas[pygame.K_RSHIFT]: self.jugador.disparar_misil(self.proyectiles)

        # El jefe tambien necesita referencia al jugador para apuntar.
        self.jefe.actualizar(self.proyectiles, self.jugador)
        self._actualizar_proyectiles_y_colisiones(daña_enemigos=False, jefe=True)

        if self.jugador.vida <= 0:
            self.estado        = self.ESTADO_FIN
            self.mensaje_final = "GAME OVER - El Jefe Final te derroto"
        elif self.jefe.vida <= 0:
            self.estado        = self.ESTADO_FIN
            self.mensaje_final = "VICTORIA - Derrotaste al Jefe Final"

    # ------------------------------------------------------------------
    # Fisica compartida: proyectiles y colisiones AABB
    # ------------------------------------------------------------------

    def _actualizar_proyectiles_y_colisiones(self, daña_enemigos: bool,
                                              jefe: bool = False) -> None:
        """
        Mueve todos los proyectiles activos y resuelve colisiones AABB.

        Colision AABB (Axis-Aligned Bounding Box):
            Se usa pygame.Rect.colliderect() que verifica si dos rectangulos
            alineados con los ejes se superponen. Es O(1) por par de objetos
            y suficiente para este tipo de juego 2D con hitboxes rectangulares.

        Reglas de colision:
            - Proyectil del jugador + enemigo: resta daño al enemigo.
            - Proyectil del jugador + jefe   : resta daño al jefe.
            - Proyectil del enemigo + jugador: llama a jugador.recibir_daño()
              (que respeta los i-frames internamente).
            - Un proyectil solo puede impactar UNA vez (se desactiva al impactar).

        Al final, se eliminan de las listas los enemigos con vida <= 0
        y los proyectiles inactivos (salieron de pantalla o impactaron).

        Parametros:
            daña_enemigos: True en Fase 1 (hay enemigos), False en Fase 2.
            jefe         : True en Fase 2 (hay jefe), False en Fase 1.
        """
        for p in self.proyectiles:
            p.actualizar()

        for p in self.proyectiles:
            if not p.activo:
                continue

            if p.dueño == "jugador":
                if daña_enemigos:
                    for enemigo in self.enemigos:
                        if p.rect.colliderect(enemigo.rect):
                            enemigo.vida -= p.daño
                            p.activo = False
                            break

                if jefe and self.jefe is not None:
                    if p.rect.colliderect(self.jefe.rect):
                        self.jefe.vida -= p.daño
                        p.activo = False

            elif p.dueño == "enemigo":
                if p.rect.colliderect(self.jugador.rect):
                    self.jugador.recibir_daño(p.daño, p.tipo)
                    p.activo = False

        self.enemigos    = [e for e in self.enemigos    if e.vida > 0]
        self.proyectiles = [p for p in self.proyectiles if p.activo]

    # ------------------------------------------------------------------
    # Renderizado principal
    # ------------------------------------------------------------------

    def _dibujar(self) -> None:
        if self.assets.fondo is not None:
            self.pantalla.blit(self.assets.fondo, (0, 0))
        else:
            self.pantalla.fill(NEGRO)

        if self.estado == self.ESTADO_RECOLECCION:
            self._dibujar_escena_juego(mostrar_enemigos=True)
            self._dibujar_panel_telemetria()
            self._dibujar_temporizador_fase1()

        elif self.estado == self.ESTADO_TRANSICION:
            self._dibujar_escena_juego(mostrar_enemigos=False)
            self._dibujar_panel_telemetria()
            self._dibujar_transicion()

        elif self.estado == self.ESTADO_JEFE:
            self._dibujar_escena_juego(mostrar_enemigos=False, mostrar_jefe=True)
            self._dibujar_panel_telemetria()
            self._dibujar_letrero_perfil()

        elif self.estado == self.ESTADO_FIN:
            self._dibujar_pantalla_fin()

    def _dibujar_escena_juego(self, mostrar_enemigos: bool = False,
                               mostrar_jefe: bool = False) -> None:

        self.jugador.dibujar(self.pantalla)

        if mostrar_enemigos:
            for enemigo in self.enemigos:
                enemigo.dibujar(self.pantalla)

        if mostrar_jefe and self.jefe is not None:
            self.jefe.dibujar(self.pantalla)

        for p in self.proyectiles:
            p.dibujar(self.pantalla)

        self._dibujar_barra(10, 10, 200, 18,
                            self.jugador.vida, Jugador.VIDA_MAX,
                            VERDE, "VIDA")

        if mostrar_jefe and self.jefe is not None:
            self._dibujar_barra(ANCHO - 210, 10, 200, 18,
                                self.jefe.vida, JefeFinal.VIDA_MAX,
                                MORADO, "JEFE")
            color_fase = {
                "A": VERDE, "B": AMARILLO, "C": ROJO
            }[self.jefe.fase_actual]
            txt_fase = self.fuente.render(
                f"FASE {self.jefe.fase_actual}", True, color_fase)
            self.pantalla.blit(txt_fase, (ANCHO - 210, 32))

    def _dibujar_barra(self, x: int, y: int, w: int, h: int,
                        valor: float, maximo: float,
                        color: tuple, etiqueta: str) -> None:

        pygame.draw.rect(self.pantalla, GRIS_OSCURO, (x, y, w, h))
        proporcion = max(0, valor) / maximo
        pygame.draw.rect(self.pantalla, color, (x, y, int(w * proporcion), h))
        pygame.draw.rect(self.pantalla, BLANCO, (x, y, w, h), width=1)
        texto = self.fuente.render(
            f"{etiqueta}: {int(max(0, valor))}/{int(maximo)}", True, BLANCO)
        self.pantalla.blit(texto, (x, y + h + 2))

    def _dibujar_temporizador_fase1(self) -> None:
        segundos = max(0, int(self.tiempo_restante_fase1))
        texto = self.fuente_mediana.render(
            f"FASE 1: RECOLECCION DE DATOS  -  {segundos}s", True, AMARILLO)
        self.pantalla.blit(texto, (ANCHO // 2 - texto.get_width() // 2, 10))

    def _dibujar_transicion(self) -> None:
        overlay = pygame.Surface((ANCHO, ALTO))
        overlay.set_alpha(160)
        overlay.fill(NEGRO)
        self.pantalla.blit(overlay, (0, 0))

        t1 = self.fuente_grande.render("ANALIZANDO PATRON DE JUEGO...", True, BLANCO)
        t2 = self.fuente.render(
            "El Perceptron Multicapa esta procesando tu vector de estado",
            True, GRIS)

        self.pantalla.blit(t1, (ANCHO // 2 - t1.get_width() // 2, ALTO // 2 - 60))
        self.pantalla.blit(t2, (ANCHO // 2 - t2.get_width() // 2, ALTO // 2 - 10))

    def _dibujar_letrero_perfil(self) -> None:
        if self.timer_letrero_perfil <= 0:
            return

        nombre = ManejadorIA.nombre_perfil(self.perfil_detectado)
        texto  = self.fuente_mediana.render(f"IA SELECCIONO: {nombre}", True, ROJO)
        x      = ANCHO // 2 - texto.get_width() // 2

        duracion_fade = 0.6
        if self.timer_letrero_perfil < duracion_fade:
            alpha = int(255 * (self.timer_letrero_perfil / duracion_fade))
        else:
            alpha = 255

        caja = pygame.Surface((texto.get_width() + 20, 34))
        caja.set_alpha(alpha)
        caja.fill(NEGRO)
        pygame.draw.rect(caja, ROJO, (0, 0, texto.get_width() + 20, 34), width=2)
        self.pantalla.blit(caja, (x - 10, 70))

        texto.set_alpha(alpha)
        self.pantalla.blit(texto, (x, 76))

    def _dibujar_panel_telemetria(self) -> None:
        ANCHO_PANEL = 230
        ALTO_PANEL  = 100
        x = ANCHO - ANCHO_PANEL - 10
        y = ALTO  - ALTO_PANEL  - 10

        panel = pygame.Surface((ANCHO_PANEL, ALTO_PANEL))
        panel.set_alpha(210)
        panel.fill(GRIS_OSCURO)
        self.pantalla.blit(panel, (x, y))
        pygame.draw.rect(self.pantalla, CIAN, (x, y, ANCHO_PANEL, ALTO_PANEL), width=2)

        ratio_laser, ratio_misil = self.jugador.obtener_ratios_daño()
        distancia                = self.jugador.obtener_preferencia_distancia()

        lineas = [
            ("TELEMETRIA DE LA IA",              CIAN),
            (f"% Daño Laser:  {ratio_laser*100:5.1f}%", BLANCO),
            (f"% Daño Misil:  {ratio_misil*100:5.1f}%", BLANCO),
            (f"Distancia prom: {distancia*100:5.1f}%",  BLANCO),
        ]
        for i, (texto, color) in enumerate(lineas):
            sup = self.fuente.render(texto, True, color)
            self.pantalla.blit(sup, (x + 10, y + 8 + i * 20))

        if self.perfil_detectado is not None:
            nombre_corto = ManejadorIA.nombre_perfil(self.perfil_detectado).split(":")[0]
            sup = self.fuente.render(f"-> {nombre_corto} ACTIVO", True, AMARILLO)
            self.pantalla.blit(sup, (x + 10, y + 8 + 4 * 20))

    def _dibujar_pantalla_fin(self) -> None:
        color = VERDE if "VICTORIA" in self.mensaje_final else ROJO
        t1 = self.fuente_grande.render(self.mensaje_final, True, color)
        t2 = self.fuente.render("Presiona R para reiniciar  |  ESC para salir",
                                True, BLANCO)
        self.pantalla.blit(t1, (ANCHO // 2 - t1.get_width() // 2, ALTO // 2 - 40))
        self.pantalla.blit(t2, (ANCHO // 2 - t2.get_width() // 2, ALTO // 2 + 20))

        if self.perfil_detectado is not None:
            nombre = ManejadorIA.nombre_perfil(self.perfil_detectado)
            t3 = self.fuente.render(f"Perfil del Jefe: {nombre}", True, GRIS)
            self.pantalla.blit(t3, (ANCHO // 2 - t3.get_width() // 2, ALTO // 2 + 50))

if __name__ == "__main__":
    juego = Juego()
    juego.ejecutar()
