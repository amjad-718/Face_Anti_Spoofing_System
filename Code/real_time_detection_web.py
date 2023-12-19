import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

app = tk.Tk()
app.title("Real-Time Detection")

# Load the trained MobileNet model
# model = load_model("latest_model.h5")
model = load_model("Models\mobilenet_real_vs_spoof.h5")

# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the labels
labels = {0: 'Real', 1: 'Spoof'}

# Open a connection to the webcam (replace 'http://your_phone_ip:8080/video' with your phone's IP and port)
cap = cv2.VideoCapture('http://192.168.137.110:8080/video')
# cap = cv2.VideoCapture(0)

# Create a Tkinter label for displaying the video stream
video_label = ttk.Label(app)
video_label.pack(padx=9, pady=8)

def update_video():
    success, frame = cap.read()

    if success:
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract the face region
            face_roi = frame[y:y+h, x:x+w]

            # Preprocess the face image for the MobileNet model
            face_resized = cv2.resize(face_roi, (224, 224))
            face_resized = np.expand_dims(face_resized, axis=0) / 255.0

            # Predict using the MobileNet model
            prediction = model.predict(face_resized)[0, 0]

            # Determine the label and color for the bounding box
            label = labels[int(round(prediction))]
            color = (0, 255, 0) if label == 'Real' else (0, 0, 255)

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f'{label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a PhotoImage object
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the Tkinter label with the new frame
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Schedule the next update after 10 milliseconds (adjust as needed)
    app.after(10, update_video)

# Start the video update loop
update_video()

# Run the Tkinter main loop
app.mainloop()

# Release the webcam when the Tkinter window is closed
cap.release()
