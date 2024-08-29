import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO

# Load the YOLOv8 model from the local path
model = YOLO('best.pt')  # Directly reference the best.pt file in the same directory as app.py

def detect_people(image):
    results = model(image)
    return results

def process_results(results, image):
    annotated_frame = image.copy()
    person_detected = False
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # Bounding boxes
        if len(boxes) > 0:
            person_detected = True
        for box in boxes:
            cv2.rectangle(annotated_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
    if person_detected:
        cv2.putText(annotated_frame, 'Canh bao! Co nguoi! Co nguoi', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    return annotated_frame

def main():
    st.title("People Detection Application using YOLOv8")

    option = st.sidebar.selectbox("Choose input type", ("Upload Image/Video", "Use Camera"))

    if option == "Upload Image/Video":
        uploaded_file = st.file_uploader("Upload Image or Video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            if uploaded_file.type.startswith('image'):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                st.image(image, caption='Uploaded Image.', use_column_width=True)
                results = detect_people(image)
                processed_frame = process_results(results, image)
                st.image(processed_frame, caption='Processed Image.', use_column_width=True)
                    
            elif uploaded_file.type.startswith('video'):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                tfile.close()
                st.write(f'Temporary file path: {tfile.name}')
                vidcap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                
                while vidcap.isOpened():
                    success, frame = vidcap.read()
                    if not success:
                        break
                    results = detect_people(frame)
                    processed_frame = process_results(results, frame)
                    stframe.image(processed_frame, channels="BGR")
                vidcap.release()
                os.remove(tfile.name)

    elif option == "Use Camera":
        stframe = st.empty()
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = detect_people(frame)
            processed_frame = process_results(results, frame)
            stframe.image(processed_frame, channels="BGR")
        cap.release()

if __name__ == '__main__':
    main()
