import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
import pickle
import os

train_dir = '../data'

def custom_image_dataset_from_directory(directory, labels='inferred', label_mode='int', class_names=None,
                                        color_mode='rgb', batch_size=32, image_size=(256, 256), shuffle=True,
                                        seed=None, validation_split=None, subset=None, interpolation='bilinear'):
    # Load the dataset using the original function
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        interpolation=interpolation
    )

    # Extract class names from the directory structure
    if class_names is None:
        class_names = sorted([dir1 for dir1 in os.listdir(directory)
                              if os.path.isdir(os.path.join(directory, dir1))])

    # Define a dictionary to map directory names to class labels
    label_map = {class_name: i for i, class_name in enumerate(class_names)}

    # Custom function to map labels
    def map_labels(images, labels):
        new_labels = []
        for label in labels:
            # Convert integer label to string
            label_str = tf.strings.as_string(label)
            # Decode the label to string
            label_str = tf.strings.reduce_join(label_str, separator='/')
            # Extract the class label from the directory structure
            class_label = tf.strings.regex_replace(tf.strings.split(label_str, sep='/')[-2], pattern=r'\d+', rewrite='')
            # Map the class label to its corresponding index
            new_labels.append(label_map[class_label.numpy()])
        return images, new_labels

    # Apply the custom label mapping function to the dataset
    dataset = dataset.map(map_labels)

    return dataset, class_names


if __name__ == "__main__":

    # Load the dataset
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=6,
        image_size=(224, 224),
        shuffle=True,
        seed=123
    )
    print(dataset.cardinality().numpy())

    # Split the dataset into training and validation sets
    val_split = 0.1
    num_val_samples = int(val_split * dataset.cardinality().numpy())
    print(num_val_samples)
    num_train_samples = dataset.cardinality().numpy() - num_val_samples
    print(f'Train samples: {num_train_samples}')

    train_dataset = dataset.skip(num_val_samples)
    validation_dataset = dataset.take(num_val_samples)

    # Extract number of classes
    num_classes = 29

    # Load the pre-trained EfficientNetB0 model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add Global Average Pooling and Output Layers
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Unfreeze the base model's layers for fine-tuning
    for layer in base_model.layers:
        layer.trainable = True

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the new data
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=2  # Adjust the number of epochs based on your needs
    )

    for key in history.history:
        print(f'{key} = {history.history[key]}')

    model_json = model.to_json()

    # Save architecture and weights together using pickle
    with open('model.pkl', 'wb') as file:
        pickle.dump({'model_architecture': model_json, 'model_weights': "model_weights.h5"}, file)

    print('save model in pikle format in file model.pkl')
