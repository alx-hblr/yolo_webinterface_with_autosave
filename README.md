# yolo_webinterface_with_autosave
A streamlit webinterface for object detection using ultralytics yolov8. Saves images with detected objects automatically.

## Install and use (on Ubuntu 24.04):

- Create a python virtual environment:

```python -m venv yolo_venv```

```source yolo_venv/bin/activate```

```cd yolo_venv```

- Install ultralytics package:

```pip install ultralytics```

- You can check if ultralytics and yolo object detection works on your machine by using yolo straight in the CLI:

```yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'``` <- This command will also download the yolov8n.pt model if not downloaded yet in the current directory.

To use this yolo_person_detection_app.py webinterface simply download it into the yolo_venv directory (same directory as the yolov8n.pt model) and start it with this command:
  
 ```python3 yolo_person_detection_app.py```
