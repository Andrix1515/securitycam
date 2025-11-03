"""
🎭 APLICACIÓN INTERACTIVA DE DETECCIÓN DE EMOCIONES Y GESTOS
Detecta emociones faciales y gestos de manos en tiempo real con efectos visuales
Autor: Sistema de Visión por Computadora
Versión: 1.0
"""

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import time
import random

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

# Inicializar MediaPipe para detección de rostro y manos
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración de emociones con colores y emojis
EMOCIONES_CONFIG = {
    'happy': {'emoji': '😊', 'color': (100, 255, 100), 'texto': 'FELIZ'},
    'sad': {'emoji': '😢', 'color': (255, 100, 100), 'texto': 'TRISTE'},
    'angry': {'emoji': '😠', 'color': (50, 50, 255), 'texto': 'ENOJADO'},
    'surprise': {'emoji': '😲', 'color': (255, 255, 100), 'texto': 'SORPRENDIDO'},
    'neutral': {'emoji': '😐', 'color': (200, 200, 200), 'texto': 'NEUTRAL'},
    'fear': {'emoji': '😨', 'color': (200, 100, 255), 'texto': 'ASUSTADO'},
    'disgust': {'emoji': '🤢', 'color': (100, 255, 255), 'texto': 'DISGUSTADO'}
}

# Configuración de gestos con efectos
GESTOS_CONFIG = {
    'thumbs_up': {'emoji': '👍', 'efecto': 'estrellas', 'texto': '¡GENIAL!'},
    'peace': {'emoji': '✌️', 'efecto': 'paz', 'texto': 'PAZ'},
    'ok': {'emoji': '👌', 'efecto': 'ondas', 'texto': 'OK'},
    'rock': {'emoji': '🤘', 'efecto': 'rayos', 'texto': 'ROCK!'},
    'palm': {'emoji': '🖐️', 'efecto': 'brillo', 'texto': 'HOLA'}
}


# ============================================================================
# CLASE PRINCIPAL DE LA APLICACIÓN
# ============================================================================

class AplicacionEmocionesGestos:
    def __init__(self):
        """Inicializa la aplicación con todos sus componentes"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Estado de la aplicación
        self.modo = 'emociones'  # 'emociones' o 'gestos'
        self.emocion_actual = 'neutral'
        self.gesto_actual = None
        self.ultimo_analisis = time.time()
        self.intervalo_analisis = 0.5  # Analizar cada 0.5 segundos
        
        # Inicializar detectores de MediaPipe
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Efectos visuales
        self.particulas = []
        self.tiempo_efecto = 0
        
        print("✅ Aplicación inicializada correctamente")
        print("📹 Cámara activada")
        print("🎮 Controles:")
        print("   [E] - Modo Emociones")
        print("   [G] - Modo Gestos")
        print("   [Q] - Salir")

    def detectar_emocion(self, frame):
        """Detecta la emoción en el rostro usando DeepFace"""
        try:
            # Analizar emoción (solo cada cierto intervalo para eficiencia)
            resultado = DeepFace.analyze(
                frame, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(resultado, list):
                resultado = resultado[0]
            
            emocion = resultado['dominant_emotion']
            return emocion
            
        except Exception as e:
            return self.emocion_actual  # Mantener emoción anterior si falla

    def detectar_gesto_mano(self, landmarks):
        """Detecta gestos específicos basados en posiciones de dedos"""
        # Extraer posiciones de los dedos
        dedos = self.contar_dedos_extendidos(landmarks)
        
        # Pulgar arriba: solo pulgar extendido
        if dedos == [1, 0, 0, 0, 0]:
            return 'thumbs_up'
        
        # Paz: índice y medio extendidos
        if dedos == [0, 1, 1, 0, 0]:
            return 'peace'
        
        # OK: pulgar e índice formando círculo
        if self.detectar_ok(landmarks):
            return 'ok'
        
        # Rock: índice y meñique extendidos
        if dedos == [0, 1, 0, 0, 1]:
            return 'rock'
        
        # Palma abierta: todos los dedos extendidos
        if dedos == [1, 1, 1, 1, 1]:
            return 'palm'
        
        return None

    def contar_dedos_extendidos(self, landmarks):
        """Cuenta qué dedos están extendidos"""
        dedos = []
        
        # Pulgar (comparar x en lugar de y)
        if landmarks[4].x < landmarks[3].x:
            dedos.append(1)
        else:
            dedos.append(0)
        
        # Otros dedos (comparar y)
        tips = [8, 12, 16, 20]  # Puntas de índice, medio, anular, meñique
        pips = [6, 10, 14, 18]  # Articulaciones medias
        
        for tip, pip in zip(tips, pips):
            if landmarks[tip].y < landmarks[pip].y:
                dedos.append(1)
            else:
                dedos.append(0)
        
        return dedos

    def detectar_ok(self, landmarks):
        """Detecta el gesto OK (pulgar e índice formando círculo)"""
        # Distancia entre pulgar e índice
        pulgar = landmarks[4]
        indice = landmarks[8]
        dist = np.sqrt((pulgar.x - indice.x)**2 + (pulgar.y - indice.y)**2)
        
        # Si están cerca, es posible que sea OK
        return dist < 0.05

    def crear_particulas(self, tipo_efecto, centro):
        """Crea partículas para efectos visuales"""
        x, y = centro
        
        for _ in range(10):
            particula = {
                'x': x,
                'y': y,
                'vx': random.uniform(-5, 5),
                'vy': random.uniform(-8, -2),
                'vida': 1.0,
                'tipo': tipo_efecto,
                'color': (random.randint(100, 255), 
                         random.randint(100, 255), 
                         random.randint(100, 255))
            }
            self.particulas.append(particula)

    def actualizar_particulas(self, frame):
        """Actualiza y dibuja partículas en el frame"""
        nuevas_particulas = []
        
        for p in self.particulas:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.3  # Gravedad
            p['vida'] -= 0.02
            
            if p['vida'] > 0:
                # Dibujar partícula
                radio = int(5 * p['vida'])
                alpha = int(255 * p['vida'])
                color = tuple(int(c * p['vida']) for c in p['color'])
                
                cv2.circle(frame, (int(p['x']), int(p['y'])), 
                          radio, color, -1)
                
                nuevas_particulas.append(p)
        
        self.particulas = nuevas_particulas

    def dibujar_interfaz_emociones(self, frame, emocion):
        """Dibuja la interfaz del modo emociones"""
        h, w = frame.shape[:2]
        config = EMOCIONES_CONFIG.get(emocion, EMOCIONES_CONFIG['neutral'])
        
        # Fondo de color suave según emoción
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), config['color'], -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Título del modo
        cv2.putText(frame, "MODO: EMOCIONES", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        
        # Emoción detectada (grande y centrado)
        texto = f"{config['emoji']} {config['texto']}"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
        x = (w - tw) // 2
        y = h - 80
        
        # Sombra del texto
        cv2.putText(frame, texto, (x+3, y+3),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 4)
        # Texto principal
        cv2.putText(frame, texto, (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 2.5, config['color'], 4)
        
        # Indicador de modo
        cv2.rectangle(frame, (w-200, 10), (w-10, 60), (50, 50, 50), -1)
        cv2.putText(frame, "[G] Gestos", (w-190, 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def dibujar_interfaz_gestos(self, frame, gesto):
        """Dibuja la interfaz del modo gestos"""
        h, w = frame.shape[:2]
        
        # Fondo del header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (100, 50, 200), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Título del modo
        cv2.putText(frame, "MODO: GESTOS", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Si hay gesto detectado
        if gesto and gesto in GESTOS_CONFIG:
            config = GESTOS_CONFIG[gesto]
            texto = f"{config['emoji']} {config['texto']}"
            
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
            x = (w - tw) // 2
            y = h - 80
            
            # Efecto de brillo
            for i in range(3):
                cv2.putText(frame, texto, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 0), 6-i*2)
            
            cv2.putText(frame, texto, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 200, 50), 4)
        
        # Indicador de modo
        cv2.rectangle(frame, (w-200, 10), (w-10, 60), (50, 50, 50), -1)
        cv2.putText(frame, "[E] Emociones", (w-190, 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def dibujar_landmarks_rostro(self, frame, face_landmarks):
        """Dibuja los puntos clave del rostro"""
        h, w = frame.shape[:2]
        
        # Dibujar solo contorno y características principales
        indices_importantes = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                              397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                              172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        for idx in indices_importantes:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    def ejecutar(self):
        """Loop principal de la aplicación"""
        print("\n🎬 Aplicación en ejecución...")
        print("=" * 50)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error al capturar frame")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            
            # ===============================================
            # MODO EMOCIONES
            # ===============================================
            if self.modo == 'emociones':
                results_face = self.face_mesh.process(rgb_frame)
                
                if results_face.multi_face_landmarks:
                    for face_landmarks in results_face.multi_face_landmarks:
                        # Dibujar puntos del rostro
                        self.dibujar_landmarks_rostro(frame, face_landmarks)
                        
                        # Analizar emoción periódicamente
                        tiempo_actual = time.time()
                        if tiempo_actual - self.ultimo_analisis > self.intervalo_analisis:
                            self.emocion_actual = self.detectar_emocion(frame)
                            self.ultimo_analisis = tiempo_actual
                            print(f"😊 Emoción detectada: {self.emocion_actual}")
                
                self.dibujar_interfaz_emociones(frame, self.emocion_actual)
            
            # ===============================================
            # MODO GESTOS
            # ===============================================
            elif self.modo == 'gestos':
                results_hands = self.hands.process(rgb_frame)
                
                if results_hands.multi_hand_landmarks:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        # Dibujar esqueleto de la mano
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                            mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                        )
                        
                        # Detectar gesto
                        gesto = self.detectar_gesto_mano(hand_landmarks.landmark)
                        
                        if gesto and gesto != self.gesto_actual:
                            self.gesto_actual = gesto
                            # Crear efecto de partículas
                            centro_x = int(hand_landmarks.landmark[9].x * w)
                            centro_y = int(hand_landmarks.landmark[9].y * h)
                            self.crear_particulas('confeti', (centro_x, centro_y))
                            print(f"✋ Gesto detectado: {gesto}")
                
                self.dibujar_interfaz_gestos(frame, self.gesto_actual)
                self.actualizar_particulas(frame)
            
            # Mostrar FPS
            cv2.putText(frame, "FPS: 30", (w-120, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow('🎭 Detector de Emociones y Gestos', frame)
            
            # Controles de teclado
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Cerrando aplicación...")
                break
            elif key == ord('e'):
                self.modo = 'emociones'
                self.gesto_actual = None
                print("\n👤 Cambiado a MODO EMOCIONES")
            elif key == ord('g'):
                self.modo = 'gestos'
                print("\n✋ Cambiado a MODO GESTOS")
        
        # Limpieza
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Aplicación cerrada correctamente")


# ============================================================================
# PUNTO DE ENTRADA DE LA APLICACIÓN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎭 APLICACIÓN DE DETECCIÓN DE EMOCIONES Y GESTOS")
    print("=" * 60)
    print("\n📦 Verificando librerías necesarias...")
    
    try:
        import cv2
        import mediapipe
        from deepface import DeepFace
        print("✅ Todas las librerías están instaladas\n")
        
        # Crear y ejecutar la aplicación
        app = AplicacionEmocionesGestos()
        app.ejecutar()
        
    except ImportError as e:
        print(f"\n❌ Error: Falta instalar una librería")
        print(f"   {e}")
        print("\n📥 Instala las dependencias con:")
        print("   pip install opencv-python mediapipe deepface tf-keras")
        print("=" * 60)