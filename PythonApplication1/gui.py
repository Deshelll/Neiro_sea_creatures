import os
import tkinter as tk
from tkinter import filedialog, messagebox
import tensorflow as tf
from PIL import Image, ImageTk
from data_preparation import load_and_preprocess_data
from model_building import build_model
from training import train_model
from testing import predict_image, draw_prediction

class SeaCreaturesGUI:
    def __init__(self, master):
        self.master = master
        master.title("Sea Creatures Classifier")

        self.label = tk.Label(master, text="Sea Creatures Classifier")
        self.label.grid(row=0, column=0, columnspan=2)

        self.train_button = tk.Button(master, text="Train Model", command=self.train_model)
        self.train_button.grid(row=1, column=0)

        self.train_path_entry = tk.Entry(master)
        self.train_path_entry.grid(row=1, column=1)

        self.load_button = tk.Button(master, text="Load Model", command=self.load_model)
        self.load_button.grid(row=2, column=0)

        self.load_path_entry = tk.Entry(master)
        self.load_path_entry.grid(row=2, column=1)

        self.test_button = tk.Button(master, text="Test Model", command=self.test_model)
        self.test_button.grid(row=3, column=0)

        self.test_path_entry = tk.Entry(master)
        self.test_path_entry.grid(row=3, column=1)

        self.exit_button = tk.Button(master, text="Exit", command=master.quit)
        self.exit_button.grid(row=4, column=0, columnspan=2)

        self.canvas = tk.Canvas(master, width=400, height=300, bg='white', bd=2, relief='solid')
        self.canvas.grid(row=5, column=0, columnspan=2)

        self.model = None
        self.categories = ["fish-citron", "fish-clown", "fish-psevdohromis", "mandarinka", "morsloi petuh", "skat", "shark", "fugu", "fish-zebrasoma"]

    def train_model(self):
        data_path = filedialog.askdirectory(title="Select Training Data Directory")
        if not data_path:
            messagebox.showerror("Error", "No directory selected.")
            return

        data, labels = load_and_preprocess_data(data_path, self.categories)
        self.model = build_model(num_classes=len(self.categories))
        self.model = train_model(self.model, data, labels, batch_size=64, epochs=100)
        self.model.save('sea_creatures_model.keras')
        messagebox.showinfo("Info", "Model trained and saved as 'sea_creatures_model.keras'")

    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Keras Models", "*.keras")])
        if not model_path:
            messagebox.showerror("Error", "No model file selected.")
            return

        self.model = tf.keras.models.load_model(model_path)
        messagebox.showinfo("Info", "Model loaded from 'sea_creatures_model.keras'")

    def test_model(self):
        if not self.model:
            messagebox.showerror("Error", "No model loaded.")
            return

        img_path = filedialog.askopenfilename(title="Select Test Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if not img_path:
            messagebox.showerror("Error", "No image selected.")
            return

        result = predict_image(self.model, img_path, self.categories)
        messagebox.showinfo("Prediction", f"Recognized object: {result}")
        self.display_image(img_path)

    def display_image(self, img_path):
        img = Image.open(img_path)
        img = img.resize((400, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk

def main():
    root = tk.Tk()
    gui = SeaCreaturesGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
