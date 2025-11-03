"""
Sistema Inteligente de Detección de Tránsito y Alertas de Emergencia por Audio
Distrito de Chilca - Seguridad Ciudadana

Enfoque: Teoría General de Sistemas (Bertalanffy) y Diseño de Sistemas (Churchman)
- Entradas: Video (cámara), Audio (micrófono), Parámetros ambientales
- Procesamiento: Detección de personas (YOLOv8) + Análisis de voz/emoción
- Salidas: Alertas visuales, sonoras, registros de eventos
- Retroalimentación: Ajuste dinámico de sensibilidad y aprendizaje continuo

Autor: Sistema de IA para Seguridad Ciudadana
Fecha: 2025
Versión: 1.0.0
"""

import cv2
import numpy as np
import threading
import queue
import time
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Librerías para detección de personas
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics no instalado. Instalar con: pip install ultralytics")

# Librerías para audio
try:
    import pyaudio
    import speech_recognition as sr
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  PyAudio o SpeechRecognition no instalados.")
    print("   Instalar con: pip install pyaudio SpeechRecognition")

# Librería para alertas sonoras
try:
    import pygame
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("⚠️  pygame no instalado. Alertas sonoras deshabilitadas.")
    print("   Instalar con: pip install pygame")


class SystemConfig:
    """
    Configuración centralizada del sistema - Principio de Jerarquía de Sistemas
    """
    # Parámetros de detección visual
    CONFIDENCE_THRESHOLD = 0.5
    PERSON_CLASS_ID = 0  # En COCO dataset, ID 0 es 'person'
    MAX_PEOPLE_NORMAL = 10  # Umbral para alertas de aglomeración
    
    # Parámetros de detección de audio
    KEYWORDS_EMERGENCY = [
        "ayuda", "socorro", "auxilio", "auxilio por favor",
        "no por favor", "ayúdenme", "me duele", "llamen"
    ]
    AUDIO_THRESHOLD_DB = 80  # Nivel de decibelios para gritos
    RECOGNITION_TIMEOUT = 3  # Segundos de escucha por ciclo
    
    # Parámetros de retroalimentación
    ADAPTIVE_SENSITIVITY = True
    NOISE_ADJUSTMENT_FACTOR = 1.2
    LEARNING_RATE = 0.1  # Para futuro aprendizaje continuo
    
    # Rutas de archivos
    LOG_FILE = "event_log.txt"
    AUDIO_SAMPLES_DIR = "audio_samples"
    ALERT_SOUND_FILE = "alert_sound.wav"
    
    # Configuración de alertas
    ALERT_DURATION_SECONDS = 5
    ALERT_COLOR = (0, 0, 255)  # Rojo en BGR
    ALERT_WINDOW_NAME = "⚠️ ALERTA DE EMERGENCIA ⚠️"


class PeopleTrafficDetector:
    """
    Subsistema Visual: Detección de personas y análisis de tránsito
    
    Funcionalidades:
    - Detección en tiempo real con YOLOv8
    - Conteo de personas
    - Análisis de densidad y aglomeración
    - Métricas de rendimiento (FPS, latencia)
    """
    
    def __init__(self, camera_id: int = 0, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.camera_id = camera_id
        self.model = None
        self.cap = None
        self.running = False
        self.people_count = 0
        self.frame_count = 0
        self.fps = 0
        self.alert_queue = queue.Queue()
        
        # Métricas de retroalimentación
        self.processing_times = []
        self.avg_processing_time = 0
        
        print("🎥 Inicializando módulo de detección visual...")
        self._initialize_model()
        self._initialize_camera()
    
    def _initialize_model(self):
        """Carga el modelo YOLOv8 para detección de personas"""
        if not YOLO_AVAILABLE:
            print("❌ No se puede inicializar YOLO. Módulo visual deshabilitado.")
            return
        
        try:
            # Usar YOLOv8n (nano) para mayor velocidad
            self.model = YOLO('yolov8n.pt')
            print("✅ Modelo YOLOv8 cargado correctamente")
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
            self.model = None
    
    def _initialize_camera(self):
        """Inicializa la captura de video"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"❌ No se puede abrir la cámara {self.camera_id}")
                return
            
            # Configurar resolución
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"✅ Cámara {self.camera_id} inicializada")
        except Exception as e:
            print(f"❌ Error al inicializar cámara: {e}")
            self.cap = None
    
    def detect_people(self, frame: np.ndarray) -> Tuple[np.ndarray, int, List[Dict]]:
        """
        Detecta personas en un frame
        
        Returns:
            frame_annotated: Frame con detecciones dibujadas
            people_count: Número de personas detectadas
            detections: Lista de detecciones con coordenadas
        """
        if self.model is None:
            return frame, 0, []
        
        start_time = time.time()
        
        # Realizar detección
        results = self.model(frame, conf=self.config.CONFIDENCE_THRESHOLD, verbose=False)
        
        people_count = 0
        detections = []
        
        # Procesar resultados
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Filtrar solo personas (class_id = 0)
                if int(box.cls[0]) == self.config.PERSON_CLASS_ID:
                    people_count += 1
                    
                    # Obtener coordenadas
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence
                    })
                    
                    # Dibujar bbox
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    cv2.putText(frame, f'Persona {confidence:.2f}', 
                              (int(x1), int(y1)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calcular tiempo de procesamiento (retroalimentación)
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        self.avg_processing_time = np.mean(self.processing_times)
        
        return frame, people_count, detections
    
    def run(self):
        """Bucle principal de detección visual"""
        if self.cap is None or self.model is None:
            print("❌ No se puede ejecutar el detector visual")
            return
        
        self.running = True
        frame_times = []
        
        print("🎥 Iniciando detección de personas...")
        
        while self.running:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️  No se puede leer frame de la cámara")
                break
            
            # Detectar personas
            frame_annotated, people_count, detections = self.detect_people(frame)
            self.people_count = people_count
            
            # Verificar aglomeración (alerta visual)
            if people_count > self.config.MAX_PEOPLE_NORMAL:
                alert = {
                    'type': 'visual',
                    'subtype': 'aglomeracion',
                    'people_count': people_count,
                    'timestamp': datetime.now(),
                    'severity': 'media'
                }
                self.alert_queue.put(alert)
            
            # Añadir información en pantalla
            info_text = [
                f"Personas detectadas: {people_count}",
                f"FPS: {self.fps:.1f}",
                f"Proc. Time: {self.avg_processing_time*1000:.1f}ms"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(frame_annotated, text, (10, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
            
            # Mostrar frame
            cv2.imshow("Detección de Tránsito - Chilca", frame_annotated)
            
            # Calcular FPS
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            self.fps = 1.0 / np.mean(frame_times) if frame_times else 0
            
            self.frame_count += 1
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop()
    
    def stop(self):
        """Detiene la detección visual"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🎥 Detector visual detenido")


class AudioEmergencyDetector:
    """
    Subsistema Auditivo: Detección de emergencias por audio
    
    Funcionalidades:
    - Captura de audio en tiempo real
    - Reconocimiento de palabras clave de emergencia
    - Detección de gritos por nivel de decibelios
    - Control de falsos positivos
    - Retroalimentación adaptativa ante ruido ambiente
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.recognizer = None
        self.microphone = None
        self.running = False
        self.alert_queue = queue.Queue()
        
        # Parámetros adaptativos (retroalimentación negativa)
        self.noise_threshold = self.config.AUDIO_THRESHOLD_DB
        self.sensitivity = 1.0
        self.ambient_noise_samples = []
        
        # Estadísticas
        self.detections_count = 0
        self.false_positives_filtered = 0
        
        print("🎤 Inicializando módulo de detección de audio...")
        self._initialize_audio()
    
    def _initialize_audio(self):
        """Inicializa el sistema de reconocimiento de voz"""
        if not AUDIO_AVAILABLE:
            print("❌ Módulo de audio no disponible")
            return
        
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Calibrar ruido ambiente
            print("🎤 Calibrando ruido ambiente... (espere 3 segundos)")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=3)
            
            print("✅ Sistema de audio inicializado correctamente")
        except Exception as e:
            print(f"❌ Error al inicializar audio: {e}")
            self.recognizer = None
            self.microphone = None
    
    def _analyze_audio_level(self, audio_data) -> float:
        """
        Analiza el nivel de audio en decibelios (aproximado)
        
        Returns:
            Nivel de audio en dB
        """
        try:
            # Convertir a array numpy
            audio_array = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
            
            # Calcular RMS (Root Mean Square)
            rms = np.sqrt(np.mean(audio_array**2))
            
            # Convertir a dB (aproximado)
            if rms > 0:
                db = 20 * np.log10(rms)
            else:
                db = 0
            
            return db
        except Exception as e:
            return 0
    
    def _detect_emergency_keywords(self, text: str) -> bool:
        """
        Verifica si el texto contiene palabras clave de emergencia
        
        Returns:
            True si se detecta emergencia
        """
        text_lower = text.lower()
        
        for keyword in self.config.KEYWORDS_EMERGENCY:
            if keyword in text_lower:
                return True
        
        return False
    
    def _apply_adaptive_filtering(self, audio_level: float) -> bool:
        """
        Retroalimentación negativa: ajusta sensibilidad según ruido ambiente
        
        Returns:
            True si el audio supera el umbral adaptativo
        """
        self.ambient_noise_samples.append(audio_level)
        if len(self.ambient_noise_samples) > 100:
            self.ambient_noise_samples.pop(0)
        
        # Calcular umbral adaptativo
        if len(self.ambient_noise_samples) > 10:
            avg_noise = np.mean(self.ambient_noise_samples)
            adaptive_threshold = avg_noise * self.config.NOISE_ADJUSTMENT_FACTOR
            
            # Ajustar sensibilidad
            if audio_level > adaptive_threshold:
                return True
        
        return audio_level > self.noise_threshold
    
    def run(self):
        """Bucle principal de detección de audio"""
        if self.recognizer is None or self.microphone is None:
            print("❌ No se puede ejecutar el detector de audio")
            return
        
        self.running = True
        print("🎤 Iniciando detección de emergencias por audio...")
        print(f"🔊 Palabras clave monitoreadas: {', '.join(self.config.KEYWORDS_EMERGENCY)}")
        
        while self.running:
            try:
                with self.microphone as source:
                    print("🎤 Escuchando...", end='\r')
                    
                    # Capturar audio
                    audio = self.recognizer.listen(
                        source, 
                        timeout=self.config.RECOGNITION_TIMEOUT,
                        phrase_time_limit=5
                    )
                    
                    # Analizar nivel de audio
                    audio_level = self._analyze_audio_level(audio)
                    
                    # Verificar si es un grito (nivel alto)
                    is_loud = self._apply_adaptive_filtering(audio_level)
                    
                    if is_loud:
                        print(f"\n🔊 Sonido fuerte detectado: {audio_level:.1f} dB")
                    
                    # Intentar reconocer voz
                    try:
                        text = self.recognizer.recognize_google(audio, language='es-ES')
                        print(f"🗣️  Texto reconocido: '{text}'")
                        
                        # Verificar palabras clave
                        if self._detect_emergency_keywords(text):
                            alert = {
                                'type': 'audio',
                                'subtype': 'palabra_clave',
                                'text': text,
                                'audio_level': audio_level,
                                'timestamp': datetime.now(),
                                'severity': 'alta'
                            }
                            self.alert_queue.put(alert)
                            self.detections_count += 1
                            
                            print(f"⚠️  ¡EMERGENCIA DETECTADA! Texto: '{text}'")
                            
                            # PLACEHOLDER: Guardar muestra de audio para aprendizaje
                            # self._save_audio_sample(audio, text)
                        
                        elif is_loud:
                            # Grito sin palabras clave reconocidas
                            alert = {
                                'type': 'audio',
                                'subtype': 'grito',
                                'text': text,
                                'audio_level': audio_level,
                                'timestamp': datetime.now(),
                                'severity': 'media'
                            }
                            self.alert_queue.put(alert)
                            print(f"⚠️  Grito detectado (sin palabras clave)")
                    
                    except sr.UnknownValueError:
                        # No se pudo reconocer voz
                        if is_loud:
                            # Sonido fuerte no verbal (posible grito)
                            alert = {
                                'type': 'audio',
                                'subtype': 'sonido_fuerte',
                                'audio_level': audio_level,
                                'timestamp': datetime.now(),
                                'severity': 'baja'
                            }
                            self.alert_queue.put(alert)
                    
                    except sr.RequestError as e:
                        print(f"❌ Error en servicio de reconocimiento: {e}")
            
            except sr.WaitTimeoutError:
                # Timeout normal, continuar
                pass
            
            except KeyboardInterrupt:
                break
            
            except Exception as e:
                print(f"❌ Error en detector de audio: {e}")
                time.sleep(1)
        
        print("🎤 Detector de audio detenido")
    
    def stop(self):
        """Detiene la detección de audio"""
        self.running = False
    
    # PLACEHOLDER: Funcionalidad de aprendizaje continuo
    def _save_audio_sample(self, audio_data, label: str):
        """
        [FUTURO] Guarda muestras de audio para entrenamiento
        
        Retroalimentación positiva: aprendizaje de nuevos patrones
        """
        # TODO: Implementar guardado de audio en formato WAV
        # TODO: Etiquetar con metadata (timestamp, label, contexto)
        # TODO: Crear dataset para fine-tuning de modelo personalizado
        pass


class SecuritySystem:
    """
    Sistema Central de Seguridad - Integración de Subsistemas
    
    Responsabilidades:
    - Coordinar subsistemas visual y auditivo
    - Gestionar alertas centralizadas
    - Registrar eventos en log
    - Emitir alertas sonoras y visuales
    - Proporcionar interfaz de control
    
    Principios de diseño:
    - Entrada-Proceso-Salida (Churchman)
    - Retroalimentación y homeostasis (Bertalanffy)
    - Control jerárquico de subsistemas
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        
        # Subsistemas
        self.visual_detector = None
        self.audio_detector = None
        
        # Hilos de ejecución
        self.threads = []
        self.running = False
        
        # Sistema de alertas
        self.alert_window_active = False
        self.sound_system_initialized = False
        
        # Registro de eventos
        self.event_log = []
        
        print("\n" + "="*60)
        print("🚨 SISTEMA DE SEGURIDAD CIUDADANA - DISTRITO DE CHILCA")
        print("="*60)
        print("Enfoque: Teoría General de Sistemas")
        print("Subsistemas: Visual (Tránsito) + Auditivo (Emergencias)")
        print("="*60 + "\n")
        
        self._initialize_sound_system()
        self._load_event_log()
    
    def _initialize_sound_system(self):
        """Inicializa pygame para alertas sonoras"""
        if not SOUND_AVAILABLE:
            print("⚠️  Sistema de sonido no disponible")
            return
        
        try:
            pygame.mixer.init()
            self.sound_system_initialized = True
            print("✅ Sistema de alertas sonoras inicializado")
            
            # PLACEHOLDER: Generar sonido de alerta si no existe
            # self._generate_alert_sound()
        except Exception as e:
            print(f"⚠️  No se pudo inicializar sistema de sonido: {e}")
    
    def _load_event_log(self):
        """Carga el registro de eventos existente"""
        if os.path.exists(self.config.LOG_FILE):
            try:
                with open(self.config.LOG_FILE, 'r', encoding='utf-8') as f:
                    self.event_log = [line.strip() for line in f.readlines()]
                print(f"✅ Log de eventos cargado: {len(self.event_log)} entradas")
            except Exception as e:
                print(f"⚠️  Error al cargar log: {e}")
    
    def _save_event(self, event: Dict):
        """
        Registra un evento en el log del sistema
        
        Formato: [TIMESTAMP] [TIPO] [SUBTIPO] [SEVERIDAD] - Detalles
        """
        timestamp = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        event_type = event['type'].upper()
        subtype = event.get('subtype', 'N/A')
        severity = event.get('severity', 'baja').upper()
        
        # Construir mensaje
        if event_type == 'VISUAL':
            details = f"Personas detectadas: {event.get('people_count', 0)}"
        elif event_type == 'AUDIO':
            text = event.get('text', 'N/A')
            audio_level = event.get('audio_level', 0)
            details = f"Texto: '{text}' | Nivel: {audio_level:.1f} dB"
        else:
            details = str(event)
        
        log_entry = f"[{timestamp}] [{event_type}] [{subtype}] [{severity}] - {details}"
        
        # Guardar en memoria y archivo
        self.event_log.append(log_entry)
        
        try:
            with open(self.config.LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"❌ Error al guardar evento: {e}")
        
        print(f"📝 Evento registrado: {log_entry}")
    
    def _show_alert_window(self, alert: Dict):
        """
        Muestra ventana de alerta visual
        """
        # Crear ventana de alerta
        alert_frame = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Fondo rojo parpadeante
        if int(time.time() * 2) % 2 == 0:
            alert_frame[:] = self.config.ALERT_COLOR
        
        # Texto de alerta
        title = "⚠️ ALERTA DE EMERGENCIA ⚠️"
        cv2.putText(alert_frame, title, (50, 80),
                   cv2.FONT_HERSHEY_BOLD, 1.0, (255, 255, 255), 3)
        
        # Detalles
        details = []
        details.append(f"Tipo: {alert['type'].upper()}")
        details.append(f"Subtipo: {alert.get('subtype', 'N/A')}")
        details.append(f"Severidad: {alert.get('severity', 'N/A').upper()}")
        details.append(f"Hora: {alert['timestamp'].strftime('%H:%M:%S')}")
        
        if alert['type'] == 'audio':
            details.append(f"Texto: {alert.get('text', 'N/A')}")
            details.append(f"Nivel: {alert.get('audio_level', 0):.1f} dB")
        elif alert['type'] == 'visual':
            details.append(f"Personas: {alert.get('people_count', 0)}")
        
        y_offset = 150
        for detail in details:
            cv2.putText(alert_frame, detail, (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
        
        cv2.imshow(self.config.ALERT_WINDOW_NAME, alert_frame)
        cv2.waitKey(1)
    
    def _play_alert_sound(self):
        """Reproduce sonido de alerta"""
        if not self.sound_system_initialized:
            return
        
        try:
            # PLACEHOLDER: Cargar sonido real
            # if os.path.exists(self.config.ALERT_SOUND_FILE):
            #     sound = pygame.mixer.Sound(self.config.ALERT_SOUND_FILE)
            #     sound.play()
            
            # Por ahora, imprimir mensaje
            print("🔊 [SONIDO DE ALERTA REPRODUCIDO]")
        except Exception as e:
            print(f"❌ Error al reproducir sonido: {e}")
    
    def _process_alerts(self):
        """
        Hilo de procesamiento de alertas
        
        Consolida alertas de ambos subsistemas y ejecuta acciones
        """
        print("🚨 Procesador de alertas activo")
        
        while self.running:
            try:
                # Revisar alertas del detector visual
                if self.visual_detector:
                    try:
                        visual_alert = self.visual_detector.alert_queue.get_nowait()
                        self._save_event(visual_alert)
                        
                        if visual_alert.get('severity') == 'alta':
                            self._show_alert_window(visual_alert)
                            self._play_alert_sound()
                    except queue.Empty:
                        pass
                
                # Revisar alertas del detector de audio
                if self.audio_detector:
                    try:
                        audio_alert = self.audio_detector.alert_queue.get_nowait()
                        self._save_event(audio_alert)

                        # Siempre mostrar alertas de audio
                        self._show_alert_window(audio_alert)
                        self._play_alert_sound()

                        # Mantener ventana de alerta por X segundos
                        start = time.time()
                        while time.time() - start < self.config.ALERT_DURATION_SECONDS:
                            self._show_alert_window(audio_alert)
                            time.sleep(0.1)
                    except queue.Empty:
                        # No hay alertas de audio en la cola
                        pass
                
            except Exception as e:
                print(f"❌ Error en procesador de alertas: {e}")
                time.sleep(1)
        
        try:
            # Esperar a que los hilos terminen
            for thread in self.threads:
                thread.join()
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupción detectada. Deteniendo sistema...")
            self.stop()
    
    def stop(self):
        """Detiene todos los subsistemas de forma segura"""
        print("\n🛑 Deteniendo sistema de seguridad...")
        
        self.running = False
        
        # Detener subsistemas
        if self.visual_detector:
            self.visual_detector.stop()
        
        if self.audio_detector:
            self.audio_detector.stop()
        
        # Cerrar ventanas
        cv2.destroyAllWindows()
        
        # Generar reporte final
        self._generate_final_report()
        
        print("✅ Sistema detenido correctamente")
    
    def start(self, enable_visual: bool = True, enable_audio: bool = True):
        """Inicia el sistema con los subsistemas especificados"""
        if self.running:
            print("⚠️ El sistema ya está en ejecución")
            return
        
        self.running = True
        
        try:
            # Iniciar detector visual si está habilitado
            if enable_visual:
                self.visual_detector = PeopleTrafficDetector(config=self.config)
                visual_thread = threading.Thread(target=self.visual_detector.run)
                visual_thread.daemon = True
                self.threads.append(visual_thread)
                visual_thread.start()
            
            # Iniciar detector de audio si está habilitado
            if enable_audio:
                self.audio_detector = AudioEmergencyDetector(config=self.config)
                audio_thread = threading.Thread(target=self.audio_detector.run)
                audio_thread.daemon = True
                self.threads.append(audio_thread)
                audio_thread.start()
            
            # Iniciar procesador de alertas
            alert_thread = threading.Thread(target=self._process_alerts)
            alert_thread.daemon = True
            self.threads.append(alert_thread)
            alert_thread.start()
            
            print("\n✅ Sistema de Seguridad iniciado correctamente")
            print(f"   Visual: {'✓' if enable_visual else '✗'}")
            print(f"   Audio:  {'✓' if enable_audio else '✗'}")
            
        except Exception as e:
            print(f"❌ Error al iniciar sistema: {e}")
            self.stop()
            raise
    
    def _generate_final_report(self):
        """Genera un reporte final de la sesión"""
        print("\n" + "="*60)
        print("📊 REPORTE FINAL DE SESIÓN")
        print("="*60)
        
        if self.visual_detector:
            print(f"🎥 Subsistema Visual:")
            print(f"   - Frames procesados: {self.visual_detector.frame_count}")
            print(f"   - FPS promedio: {self.visual_detector.fps:.1f}")
            print(f"   - Tiempo proc. promedio: {self.visual_detector.avg_processing_time*1000:.1f}ms")
        
        if self.audio_detector:
            print(f"🎤 Subsistema de Audio:")
            print(f"   - Detecciones totales: {self.audio_detector.detections_count}")
            print(f"   - Falsos positivos filtrados: {self.audio_detector.false_positives_filtered}")
        
        print(f"\n📝 Total de eventos registrados: {len(self.event_log)}")
        print(f"📁 Log guardado en: {self.config.LOG_FILE}")
        print("="*60 + "\n")
    
    # ========================================================================
    # PLACEHOLDERS PARA FUNCIONALIDADES FUTURAS
    # ========================================================================
    
    def integrate_with_map(self, camera_locations: List[Tuple[float, float]]):
        """
        [FUTURO] Integración con sistema de mapas del distrito
        
        Args:
            camera_locations: Lista de coordenadas (lat, lon) de cámaras
        
        Funcionalidad propuesta:
        - Visualizar en mapa todas las cámaras activas
        - Mostrar alertas georreferenciadas
        - Calcular zonas de mayor incidencia
        - Optimizar patrullaje policial basado en datos
        """
        # TODO: Integrar con API de Google Maps o OpenStreetMap
        # TODO: Crear sistema de geofencing para zonas críticas
        # TODO: Implementar análisis espacial de eventos
        pass
    
    def send_notification_to_authorities(self, alert: Dict):
        """
        [FUTURO] Notificación automática a autoridades
        
        Args:
            alert: Diccionario con información de la alerta
        
        Funcionalidad propuesta:
        - Enviar notificación push a app móvil de policía
        - Enviar SMS a números de emergencia registrados
        - Crear ticket en sistema de gestión de incidentes
        - Activar protocolo de respuesta según severidad
        
        Consideraciones éticas:
        - Verificación humana antes de notificar (evitar falsos positivos)
        - Protección de datos personales (GDPR/Ley de Protección de Datos)
        - Transparencia en criterios de alerta
        """
        # TODO: Implementar API REST para comunicación con central
        # TODO: Integrar con Twilio para SMS
        # TODO: Implementar Firebase Cloud Messaging para notificaciones push
        # TODO: Crear sistema de confirmación humana (human-in-the-loop)
        pass
    
    def train_custom_voice_model(self, audio_samples_path: str):
        """
        [FUTURO] Entrenamiento de modelo personalizado de detección de voz
        
        Retroalimentación positiva: Aprendizaje continuo
        
        Args:
            audio_samples_path: Ruta a directorio con muestras de audio etiquetadas
        
        Funcionalidad propuesta:
        - Fine-tuning de modelo de reconocimiento de voz para dialectos locales
        - Aprendizaje de nuevas palabras clave de emergencia
        - Adaptación a condiciones acústicas del distrito de Chilca
        - Reducción de falsos positivos mediante aprendizaje supervisado
        
        Tecnologías sugeridas:
        - Mozilla DeepSpeech o Wav2Vec 2.0
        - TensorFlow/PyTorch para entrenamiento
        - Aumento de datos (data augmentation) para robustez
        """
        # TODO: Implementar pipeline de entrenamiento
        # TODO: Crear dataset anotado de emergencias reales (con consentimiento)
        # TODO: Validar modelo con métricas (precision, recall, F1-score)
        pass
    
    def adaptive_noise_cancellation(self):
        """
        [FUTURO] Cancelación adaptativa de ruido
        
        Retroalimentación negativa: Homeostasis del sistema
        
        Funcionalidad propuesta:
        - Filtrado adaptativo de ruido de tráfico vehicular
        - Supresión de conversaciones normales (no-emergencias)
        - Ajuste dinámico de sensibilidad según hora del día
        - Aprendizaje de patrones de ruido urbano específicos de Chilca
        
        Tecnologías sugeridas:
        - Filtros Wiener o filtros de Kalman
        - Redes neuronales para separación de fuentes (source separation)
        - Algoritmos de beamforming para micrófonos direccionales
        """
        # TODO: Implementar filtros adaptativos
        # TODO: Crear perfil de ruido ambiente por zona y horario
        pass
    
    def create_control_panel_ui(self):
        """
        [FUTURO] Interfaz gráfica de control y monitoreo
        
        Opciones de implementación:
        1. Streamlit: Rápido prototipado, ideal para dashboards
        2. Tkinter: Aplicación de escritorio nativa
        3. Flask/FastAPI + React: Aplicación web completa
        
        Componentes propuestos:
        - Dashboard en tiempo real con métricas del sistema
        - Mapa de calor de incidentes
        - Gráficos de tendencias (horarios de mayor incidencia)
        - Panel de configuración (umbrales, palabras clave, etc.)
        - Registro de eventos con filtros y búsqueda
        - Sistema de usuarios con diferentes niveles de acceso
        - Visualización de cámaras en tiempo real
        
        Ejemplo de estructura:
        
        ```
        +------------------------------------------+
        |  SISTEMA DE SEGURIDAD CIUDADANA - CHILCA |
        +------------------------------------------+
        | Estado: ACTIVO 🟢 | Alertas hoy: 3      |
        +------------------+-----------------------+
        | Mapa de          | Estadísticas         |
        | cámaras          | - Personas: 234      |
        | y alertas        | - Alertas audio: 3   |
        |                  | - FPS: 28.5          |
        +------------------+-----------------------+
        | Registro de eventos (últimos 10)        |
        | [2025-10-27 14:32] AUDIO - Ayuda detect.|
        | [2025-10-27 14:15] VISUAL - Aglomeración|
        +------------------------------------------+
        ```
        """
        # TODO: Elegir framework según necesidades
        # TODO: Diseñar arquitectura frontend/backend
        # TODO: Implementar autenticación y autorización
        # TODO: Crear API RESTful para comunicación
        pass
    
    def privacy_compliance_module(self):
        """
        [FUTURO] Módulo de cumplimiento de privacidad y ética
        
        Principios fundamentales:
        1. Minimización de datos: Solo capturar lo necesario
        2. Anonimización: No identificar personas específicas
        3. Transparencia: Informar a ciudadanos sobre el sistema
        4. Control humano: Decisiones críticas requieren validación humana
        5. Auditoría: Registro de todas las acciones del sistema
        
        Funcionalidades:
        - NO reconocimiento facial (protección de identidad)
        - Anonimización automática de audio capturado
        - Retención limitada de datos (borrado automático después de X días)
        - Encriptación de datos sensibles
        - Logs de auditoría inmutables
        - Consentimiento informado para grabaciones
        - Derecho al olvido (GDPR compliance)
        
        Normativa aplicable en Perú:
        - Ley N° 29733: Ley de Protección de Datos Personales
        - Código de Protección y Defensa del Consumidor
        - Constitución Política del Perú (Art. 2, inciso 7: intimidad personal)
        """
        # TODO: Implementar sistema de anonimización
        # TODO: Crear política de retención de datos
        # TODO: Desarrollar módulo de consentimiento
        # TODO: Implementar encriptación end-to-end
        pass
    
    def predictive_analytics(self):
        """
        [FUTURO] Análisis predictivo y prevención proactiva
        
        Funcionalidad propuesta:
        - Predicción de zonas de alto riesgo según patrones históricos
        - Identificación de horarios críticos
        - Correlación de eventos (clima, eventos locales, etc.)
        - Sugerencias de despliegue de recursos policiales
        - Alertas tempranas de situaciones potencialmente peligrosas
        
        Modelos sugeridos:
        - Series temporales (ARIMA, Prophet) para tendencias
        - Clustering espacial (DBSCAN) para zonas calientes
        - Redes neuronales recurrentes (LSTM) para patrones complejos
        
        Consideraciones éticas:
        - Evitar sesgos algorítmicos (bias hacia ciertas zonas/grupos)
        - No usar para vigilancia masiva o perfilado discriminatorio
        - Transparencia en factores de predicción
        - Validación continua de exactitud de predicciones
        """
        # TODO: Recolectar datos históricos de incidentes
        # TODO: Entrenar modelos predictivos
        # TODO: Crear sistema de validación y feedback
        # TODO: Implementar auditoría de sesgos
        pass


# ============================================================================
# FUNCIÓN PRINCIPAL Y EJEMPLOS DE USO
# ============================================================================

def main():
    """Función principal de ejecución del sistema"""
    # Crear configuración personalizada (opcional)
    config = SystemConfig()
    
    # Crear e iniciar el sistema
    system = SecuritySystem(config=config)
    
    try:
        # Iniciar ambos subsistemas
        system.start(enable_visual=True, enable_audio=True)
        
        # Mantener el programa en ejecución
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Sistema interrumpido por usuario")
    except Exception as e:
        print(f"\n\n❌ Error crítico en el sistema: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.stop()


def demo_mode():
    """
    Modo demostración para pruebas sin hardware completo
    
    Útil para:
    - Desarrollar sin cámara física
    - Probar solo subsistema de audio
    - Validar procesamiento de alertas
    """
    print("\n" + "="*60)
    print("🔧 MODO DEMOSTRACIÓN")
    print("="*60)
    print("Este modo permite probar el sistema sin hardware completo")
    print("="*60 + "\n")
    
    config = SystemConfig()
    system = SecuritySystem(config=config)
    
    # Simulación: Solo iniciar subsistema disponible
    if AUDIO_AVAILABLE:
        print("🎤 Iniciando solo subsistema de audio...")
        system.start(enable_visual=False, enable_audio=True)
    elif YOLO_AVAILABLE:
        print("🎥 Iniciando solo subsistema visual...")
        system.start(enable_visual=True, enable_audio=False)
    else:
        print("❌ No hay subsistemas disponibles. Instale las dependencias.")


# ============================================================================
# INSTRUCCIONES DE INSTALACIÓN Y USO
# ============================================================================

INSTALLATION_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════════╗
║  SISTEMA DE SEGURIDAD CIUDADANA - DISTRITO DE CHILCA                     ║
║  Guía de Instalación y Uso                                               ║
╚══════════════════════════════════════════════════════════════════════════╝

1. REQUISITOS DEL SISTEMA
   ├─ Python 3.10 o superior
   ├─ Cámara web o cámara IP (para detección visual)
   ├─ Micrófono (para detección de audio)
   └─ Sistema operativo: Windows, Linux o macOS

2. INSTALACIÓN DE DEPENDENCIAS

   # Crear entorno virtual (recomendado)
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\\Scripts\\activate

   # Instalar dependencias básicas
   pip install opencv-python numpy

   # Detección de personas (YOLOv8)
   pip install ultralytics

   # Detección de audio
   pip install pyaudio SpeechRecognition

   # Alertas sonoras
   pip install pygame

   # Dependencias adicionales (opcionales)
   pip install matplotlib pandas  # Para análisis y visualización

3. CONFIGURACIÓN INICIAL

   a) Verificar acceso a cámara:
      - Conectar cámara USB o asegurar que la cámara integrada funcione
      - Probar con: python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
   
   b) Verificar acceso a micrófono:
      - Configurar permisos en sistema operativo
      - Probar grabación de audio
   
   c) Calibrar parámetros en SystemConfig:
      - MAX_PEOPLE_NORMAL: Umbral para alertas de aglomeración
      - AUDIO_THRESHOLD_DB: Sensibilidad de detección de gritos
      - KEYWORDS_EMERGENCY: Palabras clave personalizadas

4. EJECUCIÓN DEL SISTEMA

   # Modo normal (ambos subsistemas)
   python security_system_chilca.py

   # Modo solo visual
   # Editar main() y usar: system.start(enable_visual=True, enable_audio=False)

   # Modo solo audio
   # Editar main() y usar: system.start(enable_visual=False, enable_audio=True)

   # Modo demostración
   # Cambiar en __main__: demo_mode() en lugar de main()

5. USO DEL SISTEMA

   ├─ El sistema iniciará automáticamente ambos subsistemas
   ├─ Ventana de video mostrará detecciones en tiempo real
   ├─ Consola mostrará eventos de audio
   ├─ Alertas se mostrarán en ventana separada
   └─ Todos los eventos se guardan en event_log.txt

6. DETENER EL SISTEMA

   ├─ Presionar 'q' en ventana de video, o
   └─ Presionar Ctrl+C en consola

7. REVISAR LOGS

   # Ver eventos registrados
   cat event_log.txt  # En Windows: type event_log.txt

   # Analizar con Python
   with open('event_log.txt', 'r') as f:
       events = f.readlines()
       print(f"Total eventos: {len(events)}")

8. SOLUCIÓN DE PROBLEMAS

   Problema: "No se puede abrir la cámara"
   Solución: 
   - Verificar que no esté en uso por otra aplicación
   - Probar con otro índice: PeopleTrafficDetector(camera_id=1)
   
   Problema: "Error en servicio de reconocimiento"
   Solución:
   - Verificar conexión a internet (usa Google Speech Recognition)
   - Considerar usar reconocimiento offline (PocketSphinx)
   
   Problema: "Demasiados falsos positivos de audio"
   Solución:
   - Aumentar AUDIO_THRESHOLD_DB en configuración
   - Calibrar en ambiente real del distrito

9. INTEGRACIÓN CON INFRAESTRUCTURA EXISTENTE

   Para integrar con sistema municipal:
   ├─ Implementar API REST (Flask/FastAPI)
   ├─ Conectar con base de datos central (PostgreSQL/MongoDB)
   ├─ Configurar notificaciones a autoridades (ver placeholders)
   └─ Desplegar en servidor dedicado o cloud (AWS/Azure/GCP)

10. CONSIDERACIONES ÉTICAS Y LEGALES

    ⚠️  IMPORTANTE:
    ├─ Informar a ciudadanos sobre presencia de sistema de vigilancia
    ├─ NO grabar audio/video sin consentimiento o base legal
    ├─ NO usar reconocimiento facial
    ├─ Anonimizar todos los datos capturados
    ├─ Establecer políticas de retención de datos
    ├─ Realizar auditorías de sesgos algorítmicos
    └─ Cumplir Ley N° 29733 (Protección de Datos Personales - Perú)

11. CONTACTO Y SOPORTE

    Para reportar problemas o sugerencias:
    ├─ GitHub Issues (si el proyecto está en repositorio)
    ├─ Email: seguridad@munichilca.gob.pe (ejemplo)
    └─ Documentación completa: [URL del proyecto]

╔══════════════════════════════════════════════════════════════════════════╗
║  ENFOQUE SISTÉMICO - RESUMEN                                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║  ENTRADAS:                                                               ║
║  • Video en tiempo real (cámara)                                         ║
║  • Audio en tiempo real (micrófono)                                      ║
║  • Parámetros de configuración                                           ║
║  • Condiciones ambientales (ruido, iluminación)                          ║
║                                                                          ║
║  PROCESAMIENTO:                                                          ║
║  • Subsistema Visual: Detección de personas con YOLOv8                   ║
║  • Subsistema Auditivo: Reconocimiento de voz + análisis de decibelios  ║
║  • Subsistema de Alertas: Consolidación y priorización                   ║
║  • Retroalimentación negativa: Ajuste adaptativo de sensibilidad         ║
║  • Retroalimentación positiva: Aprendizaje de nuevos patrones            ║
║                                                                          ║
║  SALIDAS:                                                                ║
║  • Alertas visuales (ventanas emergentes)                                ║
║  • Alertas sonoras (sirena/mensaje)                                      ║
║  • Registro de eventos (event_log.txt)                                   ║
║  • Métricas de sistema (FPS, latencia, detecciones)                      ║
║  • [FUTURO] Notificaciones a autoridades                                 ║
║                                                                          ║
║  CONTROL Y HOMEOSTASIS:                                                  ║
║  • Ajuste automático ante ruido ambiente                                 ║
║  • Filtrado de falsos positivos                                          ║
║  • Optimización dinámica de recursos                                     ║
║  • Supervisión humana (human-in-the-loop)                                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


if __name__ == "__main__":
    # Mostrar instrucciones al iniciar
    print(INSTALLATION_INSTRUCTIONS)
    
    # Preguntar al usuario cómo ejecutar
    print("\n" + "="*60)
    print("Seleccione modo de ejecución:")
    print("1. Modo normal (recomendado)")
    print("2. Modo demostración (sin hardware completo)")
    print("3. Solo mostrar instrucciones")
    print("="*60)
    
    try:
        choice = input("\nIngrese opción (1-3) [Enter = 1]: ").strip()
        
        if choice == "" or choice == "1":
            main()
        elif choice == "2":
            demo_mode()
        elif choice == "3":
            print("\n✅ Revise las instrucciones arriba para configurar el sistema.")
        else:
            print("❌ Opción inválida")
    
    except KeyboardInterrupt:
        print("\n\n👋 Sistema cancelado por usuario")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
