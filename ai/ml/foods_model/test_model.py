import numpy as np
import os
import pathlib
import tensorflow as tf

from tensorflow import keras

BASE_DIR =  os.path.dirname(os.path.abspath(__file__))


class MLModel:
    
    @staticmethod
    def predict(image_path):
        img_height = 180
        img_width = 180
        saved_model_path = f"{BASE_DIR}/output"
        
        model = keras.models.load_model(saved_model_path)

        img = tf.keras.utils.load_img(
            image_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict2(img_array)
        score = tf.nn.softmax(predictions[0])

        return np.argmax(score), np.max(score)

    
if __name__ == "__main__":
    image_path = f"{BASE_DIR}/test_sunflower.jpg"    
    data_dir = pathlib.Path(f"{BASE_DIR}/resources/")
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir)
    class_names = train_ds.class_names
    print(class_names)

    index, score = MLModel.predict(image_path)
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[index], 100 * score)
    )