import os
import cv2
import numpy as np

def load_and_preprocess_data(data_path, categories, img_size=224):
    data = []
    labels = []
    
    for category in categories:
        path = os.path.join(data_path, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path)
                if img_array is None:
                    print(f"Warning: Image {img_path} could not be read.")
                    continue
                img_resized = cv2.resize(img_array, (img_size, img_size))
                data.append(img_resized)
                labels.append(class_num)
            except Exception as e:
                print(f"Error processing image {img}: {e}")
    
    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels
