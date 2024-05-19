import albumentations as A
import cv2
import os
import pandas as pd


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
    with open('../data/SnakeCLEF2022-TrainMetadata.csv', 'r') as original_metadata_file:
        columns = original_metadata_file.readline()
        distribution = get_data_distribution()

        with open('./augmented_metadata.csv', 'a', newline='') as new_metadata_file:
            new_metadata_file.write(columns)

        snake_id = 0
        for line in original_metadata_file:
            tokens = line.split(',')

            original_file_path = '../data/SnakeCLEF2023-small_size/' + tokens[6].strip()

            image = cv2.imread(original_file_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            species = original_file_path.split('/')[-2]
            num_of_aug = get_number_of_augmentations(species, distribution)

            augmented_images = apply_augmentation_pipeline(image, num_of_aug)
            for aug_img in augmented_images:
                new_observation_id = 'augmented' + str(snake_id)
                snake_id += 1
                print(str(snake_id + 1) + '/270252: ' + str(100*snake_id/270252))

                new_file_name = new_observation_id + '.jpg'
                new_file_path = 'data-augmentation/augmented_images/' + new_file_name
                save_image(aug_img, new_file_name)

                tokens[0] = new_observation_id
                tokens[6] = new_file_path + os.linesep

                with open('./augmented_metadata.csv', 'a', newline='') as new_metadata_file:
                    new_metadata_file.write(','.join(tokens))


def save_image(image, name):
    cv2.imwrite('./augmented_images/' + name, image)


def get_number_of_augmentations(species, distribution):
    return int(max(50 - distribution[distribution['species'] == species]['num'].iloc[0], 0) / 3)


def get_data_distribution():
    data_dir = '../data/SnakeCLEF2023-small_size'
    years = os.listdir(data_dir)

    df = []
    for year in years:
        species = os.listdir(data_dir + '/' + year)
        for s in species:
            current_s = os.listdir(data_dir + '/' + year + '/' + s)
            number = len(current_s)
            df.append({'year': year, 'species': s, 'num': number})

    data = pd.DataFrame(df)
    return data.groupby('species', as_index=False).num.sum().sort_values(by='num')


if __name__ == "__main__":
    perform_data_augmentation()
