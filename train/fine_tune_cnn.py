import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from sklearn.model_selection import train_test_split

from train.feature_extraction import build_features_model, load_and_preprocess_image

train_dir = '../data/sample'
metadata_path = '../data/SnakeCLEF2022-TrainMetadata.csv'
target_size = (224, 224)
batch_size = 20  # should be 32
epochs = 2

fine_tuned_cnn_path = 'snake_cnn.h5'


def load_images(file, batch, num_classes):
    images = []
    labels = []
    i = 0

    while i < batch:
        line = file.readline()
        if not line:
            break

        elements = line.split(',')

        image_path = train_dir + '/' + elements[-1].strip()
        if not os.path.exists(image_path):
            continue

        class_id = int(elements[-2])

        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        # TODO: meta = some_encoding_for_countries(elements[4])
        # TODO: meta_array = np.array(meta).reshape(1, -1)
        images.append(img_array)
        labels.append(class_id)

        i += 1

    # TODO: return [np.array(images), metadata], to_categorical(labels, num_classes=num_classes)
    return np.array(images), to_categorical(labels, num_classes=num_classes)


def extract_class_names():
    class_names = []
    years = os.listdir(train_dir)

    for year in years:
        year_dir = train_dir + '/' + year
        class_names += ([class_name for class_name in os.listdir(year_dir)])

    return set(class_names)


def batch_training(model, num_classes):
    print("Starting training on batches")

    current_epoch = 1
    while current_epoch < epochs:
        with open(metadata_path, 'r') as metadata_file:
            metadata_file.readline()  # skip column names
            while True:
                images, labels = load_images(metadata_file, batch_size, num_classes)
                print(f"Loaded a batch of images. Batch size is {len(images)}")

                x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
                model.train_on_batch(x_train, y_train)

                print("Successfully trained model on batch")

                metrics = model.evaluate(x_val, y_val, verbose=0)
                val_loss = metrics[0]
                val_accuracy = metrics[1]

                print(f"Epoch {current_epoch + 1}/{epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

                if len(images) < batch_size:
                    break
        current_epoch += 1

    return model


def fine_tune():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = True

    num_classes = 1784  # len(extract_class_names())
    # print("Extracted class names and their number")

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate for fine-tuning
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    trained_model = batch_training(model, num_classes)

    return trained_model


def save_model(model, path):
    model.save(path)


if __name__ == "__main__":
    # start = time.time()
    # fine_tuned_model = fine_tune()
    # end = time.time()
    #
    # print(f'Fine-tuning took {end - start} seconds')
    #
    # save_model(fine_tuned_model, fine_tuned_cnn_path)

    feature_extraction_model = build_features_model(fine_tuned_cnn_path)

    img_path = '../data/SnakeCLEF2023-small_size/2003/Bitis_caudalis/12089782.jpeg'
    img_arr = load_and_preprocess_image(img_path)

    feature_vector = feature_extraction_model.predict(img_arr)

    print(f"Feature vector (shape {feature_vector.shape}):")
    print(feature_vector)

