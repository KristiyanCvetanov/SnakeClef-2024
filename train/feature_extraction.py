import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
import numpy as np
from PIL import Image

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def build_features_model():
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        pooling='avg',
    )

    image_input = Input(shape=(224, 224, 3), name='image_input')
    x = base_model(image_input)
    x = Flatten()(x) 
    x = Dense(4096, activation='relu')(x)
    feature_vector = x

    metadata_input = Input(shape=(2,), name='metadata_input')
    combined = Concatenate()([feature_vector, metadata_input])

    feature_extraction_model = Model(inputs=[image_input, metadata_input], outputs=combined)

    return feature_extraction_model

def extract_features(image_paths, metadata):
    feature_extraction_model = build_features_model()
    features = []

    for img_path, meta in zip(image_paths, metadata):
        img_array = load_and_preprocess_image(img_path)
        meta_array = np.array(meta).reshape(1, -1)
        feature_vector = feature_extraction_model.predict([img_array, meta_array])
        features.append(feature_vector.flatten())

    return np.array(features)

if __name__ == 'main':

    img_path = './image.jpg'

    img_array = load_and_preprocess_image(img_path)

    feature_extraction_model = build_features_model()
    feature_vector = feature_extraction_model.predict(img_array)

    print(f"Feature vector (shape {feature_vector.shape}):")
    print(feature_vector)
