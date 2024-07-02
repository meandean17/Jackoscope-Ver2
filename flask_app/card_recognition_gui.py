import tkinter as tk
from PIL import Image, ImageTk
import cv2
from CardDetector import CardDetector, preprocess_image

class CardRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Card Recognition GUI")

        self.card_detector = CardDetector()

        self.canvas = tk.Canvas(self.root, width=1280, height=720)
        self.canvas.pack()

        self.btn_snapshot = tk.Button(self.root, text="Take Snapshot", command=self.take_snapshot)
        self.btn_snapshot.pack()

        self.update()

    def take_snapshot(self):
        # Capture a frame from the card detector
        frame = self.card_detector.read_frame()

        # Do something with the frame (e.g., detect cards)
        processed_frame = preprocess_image(frame)  # Replace this with your card detection logic

        # Display the processed frame
        self.display_image(processed_frame)

    def display_image(self, frame):
        # Convert the frame from OpenCV BGR format to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a format compatible with Tkinter
        img = Image.fromarray(rgb_frame)
        img = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new image
        self.canvas.img = img
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)

    def update(self):
        # Continuously update the display with new frames from the card detector
        frame = self.card_detector.read_frame()
        processed_frame = preprocess_image(frame)  # Replace this with your card detection logic
        self.display_image(processed_frame)
        self.root.after(10, self.update)

    def close(self):
        # Stop the card detector when the GUI is closed
        self.card_detector.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CardRecognitionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.close)  # Handle window close event
    root.mainloop()
