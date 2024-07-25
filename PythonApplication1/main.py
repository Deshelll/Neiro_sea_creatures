import os
import tensorflow as tf
from data_preparation import load_and_preprocess_data
from model_building import build_model
from training import train_model
from testing import predict_image, draw_prediction

def main():
    action = input("Enter 'train' to train the model or 'load' to load the model: ").strip().lower()
    
    if action == 'train':
        data_path = input("Enter the path to your dataset (e.g., C:\\Users\\early\\Desktop\\kursach_neiro\\neiro\\neiro\\dataset): ")
        if not os.path.exists(data_path):
            print(f"Error: The path {data_path} does not exist.")
            return

        categories = ["fish-citron", "fish-clown", "fish-psevdohromis","mandarinka","morsloi petuh","skat", "shark", "fugu", "fish-zebrasoma"]
        print(f"Using data path: {data_path}")
        print(f"Categories: {categories}")

        # Data preparation
        data, labels = load_and_preprocess_data(data_path, categories)

        # Model building
        model = build_model(num_classes=len(categories))

        # Model training
        model = train_model(model, data, labels, batch_size=32, epochs=50)

        # Save the model
        model.save('sea_creatures_model.keras')
        print("Model trained and saved as 'sea_creatures_model.keras'")
    
    elif action == 'load':
        # Load the model
        model = tf.keras.models.load_model('sea_creatures_model.keras')
        print("Model loaded from 'sea_creatures_model.keras'")
    
    else:
        print("Invalid action. Please enter 'train' or 'load'.")
        return

    # Testing loop
    categories = ["fish-citron", "fish-clown", "fish-psevdohromis","mandarinka","morsloi petuh","skat", "shark", "fugu", "fish-zebrasoma"]
    while True:
        img_path = input("Enter the path to the test image (or 'exit' to quit): ")
        if img_path.lower() == 'exit':
            print("Exiting...")
            break
        if not img_path:
            print("Error: Please enter a valid path to the test image.")
            continue
        if not os.path.exists(img_path):
            print(f"Error: The image path {img_path} does not exist.")
            continue
        result = predict_image(model, img_path, categories)
        print(f"Recognized object: {result}")
        draw_prediction(img_path, result)

if __name__ == "__main__":
    main()
