import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import os
from PIL import Image

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Function to process frame and save image if person detected
def process_frame(frame):
    results = model(frame)
    
    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()
    
    person_detected = False
    
    # Iterate through the detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Check if the detected object is a person (class 0 in COCO dataset)
            if int(box.cls) == 0:
                person_detected = True
                # Get the bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                # Draw the bounding box
                cv2.rectangle(annotated_frame, 
                              (int(x1), int(y1)), 
                              (int(x2), int(y2)), 
                              (255, 0, 0), 2)  # Green color, thickness 2
                # Add label
                cv2.putText(annotated_frame, 
                            'Person', 
                            (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, 
                            (255, 0, 0), 
                            2)

    if person_detected:
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"person_detected_{timestamp}.jpg"
        cv2.imwrite(os.path.join("detected_persons", filename), cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        st.sidebar.write(f"Person detected! Image saved as {filename}")
    
    return annotated_frame

# Function to display saved images in a grid
def display_saved_images():
    saved_images = os.listdir("detected_persons")
    saved_images.sort(reverse=True)  # Show newest first
    
    st.header("Saved Detections Gallery")
    
    # Create a 3-column layout
    cols = st.columns(3)
    
    for idx, img_name in enumerate(saved_images):
        img_path = os.path.join("detected_persons", img_name)
        img = Image.open(img_path)
        cols[idx % 3].image(img, caption=img_name, use_column_width=True)

# Streamlit app
def main():
    st.title("Person Detection with Auto-Save")

    # Create a folder to save detected person images if it doesn't exist
    if not os.path.exists("detected_persons"):
        os.makedirs("detected_persons")

    # Sidebar for showing recent detections
    st.sidebar.title("Recent Detections")
    saved_images = os.listdir("detected_persons")
    saved_images.sort(reverse=True)  # Show newest first
    for img in saved_images[:5]:  # Show last 5 images
        st.sidebar.image(f"detected_persons/{img}", caption=img, use_column_width=True)

    # Main content - live feed and saved images
    tab1, tab2 = st.tabs(["Live Feed", "Saved Images"])
    
    with tab1:
        st.header("Live Feed")
        start_button = st.button("Start Live Feed")
        stop_button = st.button("Stop Live Feed")
        stframe = st.empty()
        
        cap = cv2.VideoCapture(0)  # Use 0 for webcam

        run = st.empty()
        run.text("Click 'Start Live Feed' to begin")

        while start_button and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = process_frame(frame)
            stframe.image(processed_frame, channels="RGB", use_column_width=True)
            run.text("Live feed is running. Click 'Stop Live Feed' to end.")

            if stop_button:
                run.text("Live feed stopped")
                break

        cap.release()
    
    with tab2:
        if st.button("Refresh Saved Images"):
            display_saved_images()

if __name__ == "__main__":
    main()
