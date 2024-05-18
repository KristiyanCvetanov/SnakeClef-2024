import csv

import albumentations as A
import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd
from tqdm import tqdm


def apply_augmentation_pipeline(image, num_augmentations):
    augmentation_pipeline = A.Compose([
        A.Rotate(limit=90, p=0.8),  # Rotate the image within the range of -90 to +90 degrees
        A.HorizontalFlip(p=0.65),  # 65% chance to flip horizontally
        A.VerticalFlip(p=0.65),  # 65% chance to flip vertically
        A.RandomBrightnessContrast(p=0.5),  # Randomly change brightness and contrast
        A.GaussNoise(p=0.5),  # Apply Gaussian noise
        A.CenterCrop(height=image.shape[0] - 20, width=image.shape[1] - 20, p=0.5),  # Center crop
        A.RGBShift(p=0.5),
        A.CLAHE(p=0.5)
    ])

    augmented_images = []

    for _ in range(num_augmentations):
        augmented_image = augmentation_pipeline(image=image)['image']
        augmented_images.append(augmented_image)

    return augmented_images


# !!! perform only once !!!
def perform_data_augmentation():
    # 1. create new metadata file for augmented data
    # 3. read images one by one
    # 4. apply augmentation pipeline
    # 5. for all new images
    # 6. (add to/create new) directory with name - the name of the species
    # 7. create unique id "augmented*number_of_augmented_pic*"
    # 8. create string with new file_path
    # 9. copy all other attributes from original metadata
    # 10. save image to new file_path
    # 11. append new line to new metadata file

    # with open('./augmented/augmented_metadata.csv', 'w', newline='') as new_metadata_file:
    #     writer = csv.writer(new_metadata_file)

    with open('../data/SnakeCLEF2022-TrainMetadata.csv', 'r') as original_metadata_file:
        columns = original_metadata_file.readline()
        # observation_id,endemic,binomial_name,country,code,class_id,file_path

        with open('./augmented_metadata.csv', 'a', newline='') as new_metadata_file:
            writer = csv.writer(new_metadata_file)
            writer.writerow(columns)

        snake_id = 0
        for line in original_metadata_file:
            tokens = line.split(',')

            new_observation_id = 'augmented' + str(snake_id)
            snake_id += 1
            if snake_id > 10:
                break

            new_file_name = new_observation_id + '.jpg'
            new_file_path = './augmented_images' + new_file_name

            original_file_path = '../data/SnakeCLEF2023-small_size/' + tokens[6]

            image = cv2.imread(original_file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            num_of_aug = get_number_of_augmentations(tokens[2])

            augmented_images = apply_augmentation_pipeline(image, num_of_aug)
            for aug_img in augmented_images:
                save_image(aug_img, new_file_name)

            tokens[0] = new_observation_id
            tokens[6] = new_file_path

            with open('./augmented_metadata.csv', 'a', newline='') as new_metadata_file:
                writer = csv.writer(new_metadata_file)
                writer.writerow(','.join(tokens))


def save_image(image, name):
    plt.imshow(image)
    plt.axis('off')
    plt.savefig('./augmented_images/' + name, bbox_inches='tight', pad_inches=0)


def get_number_of_augmentations(species):
    distribution = get_data_distribution()
    return int(max(50 - distribution[species]['num'], 0) / 3)


def get_data_distribution():
    data_dir = '../data/SnakeCLEF2023-small_size'
    years = os.listdir(data_dir)

    df = []
    for year in tqdm(years):
        species = os.listdir(data_dir + '/' + year)
        for s in species:
            current_s = os.listdir(data_dir + '/' + year + '/' + s)
            number = len(current_s)
            df.append({'year': year, 'species': s, 'num': number})

    data = pd.DataFrame(df)
    return data.groupby('species', as_index=False).num.sum().sort_values(by='num')


if __name__ == "__main__":
    perform_data_augmentation()
