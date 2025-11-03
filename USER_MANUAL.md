USER MANUAL — Sistema de Seguridad Chilca
=========================================

Resumen
-------
Este repositorio contiene un sistema de detección visual (YOLOv8) y detección de audio (SpeechRecognition) para alertas de emergencia. El archivo principal corregido es `security_system_chilca.py`.

Requisitos (Windows)
--------------------
- Python 3.8+ (preferible 3.9-3.11)
- pip
- Visual C++ Build Tools (para instalar paquetes que requieran compilación, p. ej. PyAudio)

Dependencias Python
-------------------
Recomendado crear un entorno virtual:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

Instalar dependencias básicas:

```powershell
pip install --upgrade pip
pip install numpy opencv-python ultralytics SpeechRecognition pyaudio pygame
```

Notas:
- `ultralytics` requiere pip install ultralytics. Asegúrate de tener `yolov8n.pt` en el mismo directorio o ajusta la ruta.
- `pyaudio` en Windows puede requerir la instalación previa de ruedas (.whl). Si falla, descarga la rueda apropiada para tu versión de Python desde: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

Archivos clave
--------------
- `security_system_chilca.py` : Archivo principal (corregido). Contiene las clases:
  - `SystemConfig` — Parámetros configurables.
  - `PeopleTrafficDetector` — Subsistema visual (YOLOv8).
  - `AudioEmergencyDetector` — Subsistema de audio (SpeechRecognition/PyAudio).
  - `SecuritySystem` — Orquestador central, gestión de alertas y logging.
- `security_system_chilca_new.py` : Versión alternativa/backup que se creó durante la depuración.
- `event_log.txt` : Log de eventos generados por el sistema (se crea si no existe).
- `yolov8n.pt` : Modelo YOLOv8 (debe estar presente si se desea detección local).

Uso — modo normal
------------------
1. Activar entorno virtual (ver arriba).
2. Ejecutar el script:

```powershell
python .\security_system_chilca.py
```

El sistema inicializará subsistemas disponibles (visual y audio). Puedes detenerlo con Ctrl+C.

Opciones de inicio desde código
-------------------------------
En la función `main()` se instancia `SecuritySystem` y se llama a:

- `system.start(enable_visual=True, enable_audio=True)` — iniciar ambos subsistemas.
- Para probar solo audio: `system.start(enable_visual=False, enable_audio=True)`.
- Para probar solo visual: `system.start(enable_visual=True, enable_audio=False)`.

Modo demostración (sin cámara o micrófono)
------------------------------------------
Si no cuentas con hardware, puedes modificar `main()` para iniciar solamente el subsistema disponible o crear un modo demo (el repo contiene `demo_mode()` como referencia). Ejemplo para evitar iniciar la cámara:

```python
system.start(enable_visual=False, enable_audio=True)
```

Logs y salida
-------------
- Los eventos se registran en `event_log.txt` con formato:
  [TIMESTAMP] [TIPO] [SUBTIPO] [SEVERIDAD] - Detalles
- Las alertas visuales aparecen como ventanas OpenCV. Pulsa 'q' en la ventana de detección para cerrar el visual.

Solución de problemas
---------------------
1. Error "Try must have at least one except or finally" (como el que reportaste):
   - Ya fue corregido en `security_system_chilca.py`. Si vuelves a ver un error de sintaxis, ejecuta `python -m pyflakes .` o revisa la línea indicada.

2. Problemas al abrir la cámara:
   - Verifica que ninguna otra aplicación esté usando la cámara.
   - Cambia `camera_id` en `PeopleTrafficDetector(camera_id=0)` a 1,2...

3. Errores de PyAudio en Windows:
   - Instala la rueda (.whl) adecuada o instala Microsoft Build Tools.

4. Problemas con el modelo YOLO:
   - Asegúrate de que `yolov8n.pt` exista en el directorio o especifica la ruta completa en `YOLO('ruta/a/yolov8n.pt')`.

5. SpeechRecognition request errors:
   - La API de Google (recognize_google) requiere acceso a internet.
   - Si necesitas reconocimiento offline, considera modelos alternativos (VOSK, Whisper local, etc.).

Comprobación rápida tras la corrección
-------------------------------------
1. Desde PowerShell, en el directorio del proyecto y con el entorno activado:

```powershell
python -c "import importlib, sys; importlib.invalidate_caches(); print('Python OK')"
python -m pyflakes security_system_chilca.py || echo 'pyflakes OK or not installed'
python .\security_system_chilca.py
```

Esto iniciará el sistema. Observa la consola para mensajes de inicialización (YOLO, audio, pygame).

Buenas prácticas
---------------
- Ejecutar en entorno virtual.
- Probar primero con `enable_visual=False` si no tienes cámara.
- Mantener `yolov8n.pt` actualizado según versiones de `ultralytics`.

Contacto y siguientes pasos
--------------------------
Si quieres, puedo:
- Añadir un archivo `requirements.txt` con versiones pinedas.
- Crear un modo demo que reproduzca frames/videos de ejemplo en lugar de cámara real.
- Añadir tests unitarios mínimos para las funciones no dependientes de hardware.

---
Archivo generado el: 2025-10-31
