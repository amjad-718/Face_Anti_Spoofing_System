import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

app = tk.Tk()
app.title("Image Detection")

# Load the trained MobileNet model
# model = load_model("latest_model.h5")
model = load_model("mobilenet_real_vs_spoof.h5")


# Load the Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the labels
labels = {0: 'Real', 1: 'Spoof'}

# Create a Tkinter label for displaying the image
image_label = ttk.Label(app)
image_label.pack(padx=9, pady=8)

def load_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        process_image(file_path)

def process_image(file_path):
    # Read the image
    frame = cv2.imread(file_path)

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

    # Resize the frame to fit the Tkinter window
    frame = resize_frame(frame)

    # Convert the frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PhotoImage object
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)

    # Update the Tkinter label with the new frame
    image_label.imgtk = imgtk
    image_label.configure(image=imgtk)

def resize_frame(frame, max_width=800, max_height=600):
    # Get the original dimensions
    height, width = frame.shape[:2]

    # Calculate the resizing factor to fit within the specified limits
    width_factor = max_width / width
    height_factor = max_height / height
    resize_factor = min(width_factor, height_factor)

    # Resize the frame
    resized_frame = cv2.resize(frame, (int(width * resize_factor), int(height * resize_factor)))

    return resized_frame

# Create a button to load an image
load_button = ttk.Button(app, text="Load Image", command=load_image)
load_button.pack(pady=10)

# Run the Tkinter main loop
app.mainloop()
