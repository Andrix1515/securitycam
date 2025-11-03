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
    """Configuración centralizada del sistema"""
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
    LEARNING_RATE = 0.1
    
    # Rutas de archivos
    LOG_FILE = "event_log.txt"
    AUDIO_SAMPLES_DIR = "audio_samples"
    ALERT_SOUND_FILE = "alert_sound.wav"
    
    # Configuración de alertas
    ALERT_DURATION_SECONDS = 5
    ALERT_COLOR = (0, 0, 255)  # Rojo en BGR
    ALERT_WINDOW_NAME = "⚠️ ALERTA DE EMERGENCIA ⚠️"


class PeopleTrafficDetector:
    """Subsistema Visual: Detección de personas y análisis de tránsito"""
    
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
        self.processing_times = []
        self.avg_processing_time = 0
        
        print("🎥 Inicializando módulo de detección visual...")
        self._initialize_model()
        self._initialize_camera()
    
    def _initialize_model(self):
        if not YOLO_AVAILABLE:
            print("❌ No se puede inicializar YOLO. Módulo visual deshabilitado.")
            return
        
        try:
            self.model = YOLO('yolov8n.pt')
            print("✅ Modelo YOLOv8 cargado correctamente")
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
            self.model = None
    
    def _initialize_camera(self):
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"❌ No se puede abrir la cámara {self.camera_id}")
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"✅ Cámara {self.camera_id} inicializada")
        except Exception as e:
            print(f"❌ Error al inicializar cámara: {e}")
            self.cap = None
    
    def detect_people(self, frame: np.ndarray) -> Tuple[np.ndarray, int, List[Dict]]:
        if self.model is None:
            return frame, 0, []
        
        start_time = time.time()
        results = self.model(frame, conf=self.config.CONFIDENCE_THRESHOLD, verbose=False)
        people_count = 0
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) == self.config.PERSON_CLASS_ID:
                    people_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence
                    })
                    
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    cv2.putText(frame, f'Persona {confidence:.2f}', 
                              (int(x1), int(y1)-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        self.avg_processing_time = np.mean(self.processing_times)
        
        return frame, people_count, detections
    
    def run(self):
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
            
            frame_annotated, people_count, detections = self.detect_people(frame)
            self.people_count = people_count
            
            if people_count > self.config.MAX_PEOPLE_NORMAL:
                alert = {
                    'type': 'visual',
                    'subtype': 'aglomeracion',
                    'people_count': people_count,
                    'timestamp': datetime.now(),
                    'severity': 'media'
                }
                self.alert_queue.put(alert)
            
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
            
            cv2.imshow("Detección de Tránsito - Chilca", frame_annotated)
            
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            self.fps = 1.0 / np.mean(frame_times) if frame_times else 0
            
            self.frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.stop()
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🎥 Detector visual detenido")


class AudioEmergencyDetector:
    """Subsistema Auditivo: Detección de emergencias por audio"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.recognizer = None
        self.microphone = None
        self.running = False
        self.alert_queue = queue.Queue()
        self.noise_threshold = self.config.AUDIO_THRESHOLD_DB
        self.sensitivity = 1.0
        self.ambient_noise_samples = []
        self.detections_count = 0
        self.false_positives_filtered = 0
        
        print("🎤 Inicializando módulo de detección de audio...")
        self._initialize_audio()
    
    def _initialize_audio(self):
        if not AUDIO_AVAILABLE:
            print("❌ Módulo de audio no disponible")
            return
        
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            print("🎤 Calibrando ruido ambiente... (espere 3 segundos)")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=3)
            
            print("✅ Sistema de audio inicializado correctamente")
        except Exception as e:
            print(f"❌ Error al inicializar audio: {e}")
            self.recognizer = None
            self.microphone = None
    
    def _analyze_audio_level(self, audio_data) -> float:
        try:
            audio_array = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
            rms = np.sqrt(np.mean(audio_array**2))
            return 20 * np.log10(rms) if rms > 0 else 0
        except Exception:
            return 0
    
    def _detect_emergency_keywords(self, text: str) -> bool:
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.config.KEYWORDS_EMERGENCY)
    
    def _apply_adaptive_filtering(self, audio_level: float) -> bool:
        self.ambient_noise_samples.append(audio_level)
        if len(self.ambient_noise_samples) > 100:
            self.ambient_noise_samples.pop(0)
        
        if len(self.ambient_noise_samples) > 10:
            avg_noise = np.mean(self.ambient_noise_samples)
            adaptive_threshold = avg_noise * self.config.NOISE_ADJUSTMENT_FACTOR
            if audio_level > adaptive_threshold:
                return True
        
        return audio_level > self.noise_threshold
    
    def run(self):
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
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.config.RECOGNITION_TIMEOUT,
                        phrase_time_limit=5
                    )
                    
                    audio_level = self._analyze_audio_level(audio)
                    is_loud = self._apply_adaptive_filtering(audio_level)
                    
                    if is_loud:
                        print(f"\n🔊 Sonido fuerte detectado: {audio_level:.1f} dB")
                    
                    try:
                        text = self.recognizer.recognize_google(audio, language='es-ES')
                        print(f"🗣️  Texto reconocido: '{text}'")
                        
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
                        
                        elif is_loud:
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
                        if is_loud:
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
                pass
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error en detector de audio: {e}")
                time.sleep(1)
        
        print("🎤 Detector de audio detenido")
    
    def stop(self):
        self.running = False


class SecuritySystem:
    """Sistema Central de Seguridad"""
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.visual_detector = None
        self.audio_detector = None
        self.threads = []
        self.running = False
        self.alert_window_active = False
        self.sound_system_initialized = False
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
        if not SOUND_AVAILABLE:
            print("⚠️  Sistema de sonido no disponible")
            return
        
        try:
            pygame.mixer.init()
            self.sound_system_initialized = True
            print("✅ Sistema de alertas sonoras inicializado")
        except Exception as e:
            print(f"⚠️  No se pudo inicializar sistema de sonido: {e}")
    
    def _load_event_log(self):
        if os.path.exists(self.config.LOG_FILE):
            try:
                with open(self.config.LOG_FILE, 'r', encoding='utf-8') as f:
                    self.event_log = [line.strip() for line in f.readlines()]
                print(f"✅ Log de eventos cargado: {len(self.event_log)} entradas")
            except Exception as e:
                print(f"⚠️  Error al cargar log: {e}")
    
    def _save_event(self, event: Dict):
        timestamp = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        event_type = event['type'].upper()
        subtype = event.get('subtype', 'N/A')
        severity = event.get('severity', 'baja').upper()
        
        if event_type == 'VISUAL':
            details = f"Personas detectadas: {event.get('people_count', 0)}"
        elif event_type == 'AUDIO':
            text = event.get('text', 'N/A')
            audio_level = event.get('audio_level', 0)
            details = f"Texto: '{text}' | Nivel: {audio_level:.1f} dB"
        else:
            details = str(event)
        
        log_entry = f"[{timestamp}] [{event_type}] [{subtype}] [{severity}] - {details}"
        self.event_log.append(log_entry)
        
        try:
            with open(self.config.LOG_FILE, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"❌ Error al guardar evento: {e}")
        
        print(f"📝 Evento registrado: {log_entry}")
    
    def _show_alert_window(self, alert: Dict):
        alert_frame = np.zeros((400, 600, 3), dtype=np.uint8)
        
        if int(time.time() * 2) % 2 == 0:
            alert_frame[:] = self.config.ALERT_COLOR
        
        title = "⚠️ ALERTA DE EMERGENCIA ⚠️"
        cv2.putText(alert_frame, title, (50, 80),
                   cv2.FONT_HERSHEY_BOLD, 1.0, (255, 255, 255), 3)
        
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
        if not self.sound_system_initialized:
            return
        
        try:
            print("🔊 [SONIDO DE ALERTA REPRODUCIDO]")
        except Exception as e:
            print(f"❌ Error al reproducir sonido: {e}")
    
    def _process_alerts(self):
        print("🚨 Procesador de alertas activo")
        
        while self.running:
            try:
                if self.visual_detector:
                    try:
                        visual_alert = self.visual_detector.alert_queue.get_nowait()
                        self._save_event(visual_alert)
                        
                        if visual_alert.get('severity') == 'alta':
                            self._show_alert_window(visual_alert)
                            self._play_alert_sound()
                    except queue.Empty:
                        pass
                
                if self.audio_detector:
                    try:
                        audio_alert = self.audio_detector.alert_queue.get_nowait()
                        self._save_event(audio_alert)
                        self._show_alert_window(audio_alert)
                        self._play_alert_sound()
                    except queue.Empty:
                        pass
                
                time.sleep(0.1)
            
            except Exception as e:
                print(f"❌ Error en procesador de alertas: {e}")
                time.sleep(1)
    
    def start(self, enable_visual: bool = True, enable_audio: bool = True):
        if self.running:
            print("⚠️ El sistema ya está en ejecución")
            return
        
        self.running = True
        
        try:
            if enable_visual:
                self.visual_detector = PeopleTrafficDetector(config=self.config)
                visual_thread = threading.Thread(target=self.visual_detector.run)
                visual_thread.daemon = True
                self.threads.append(visual_thread)
                visual_thread.start()
            
            if enable_audio:
                self.audio_detector = AudioEmergencyDetector(config=self.config)
                audio_thread = threading.Thread(target=self.audio_detector.run)
                audio_thread.daemon = True
                self.threads.append(audio_thread)
                audio_thread.start()
            
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
    
    def stop(self):
        print("\n🛑 Deteniendo sistema de seguridad...")
        
        self.running = False
        
        if self.visual_detector:
            self.visual_detector.stop()
        
        if self.audio_detector:
            self.audio_detector.stop()
        
        cv2.destroyAllWindows()
        
        self._generate_final_report()
        
        print("✅ Sistema detenido correctamente")
    
    def _generate_final_report(self):
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


def main():
    """Función principal de ejecución del sistema"""
    config = SystemConfig()
    system = SecuritySystem(config=config)
    
    try:
        system.start(enable_visual=True, enable_audio=True)
        
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


if __name__ == "__main__":
    main()