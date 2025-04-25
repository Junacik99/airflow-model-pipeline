import os
import shutil
import random
import click

# Split dataset into train-valid-test
def split_dataset(data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, copy=True):
    """Splits a dataset into train, validation, and test sets.

    Args:
    data_path: The directory containing the dataset.
    train_ratio: The proportion of the dataset to use for training.
    val_ratio: The proportion of the dataset to use for validation.
    test_ratio: The proportion of the dataset to use for testing.
    """

    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    test_ratio = float(test_ratio)
    copy = True if copy.lower() == 'true' else False

    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    train_dir = os.path.join(data_path, 'train')
    valid_dir = os.path.join(data_path, 'valid')
    test_dir = os.path.join(data_path, 'test')


    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(data_path):
        if class_name in ['train', 'valid', 'test']:
            continue

        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Create subdirectories for each class in train, valid, and test
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        random.shuffle(images)  # Shuffle images randomly
        num_images = len(images)

        num_train = int(train_ratio * num_images)
        num_val = int(val_ratio * num_images)
        num_test = num_images - num_train - num_val

        print(f"Splitting {class_name}:")
        print(f'Training: {num_train}, Validation: {num_val}, Testing: {num_test}')

        train_images = images[:num_train]
        val_images = images[num_train: num_train + num_val]
        test_images = images[num_train + num_val:]

        for image in train_images:
            source_path = os.path.join(class_dir, image)
            destination_path = os.path.join(train_dir, class_name, image)
            shutil.copy(source_path, destination_path) if copy else shutil.move(source_path, destination_path)

        for image in val_images:
            source_path = os.path.join(class_dir, image)
            destination_path = os.path.join(valid_dir, class_name, image)
            shutil.copy(source_path, destination_path) if copy else shutil.move(source_path, destination_path)

        for image in test_images:
            source_path = os.path.join(class_dir, image)
            destination_path = os.path.join(test_dir, class_name, image)
            shutil.copy(source_path, destination_path) if copy else shutil.move(source_path, destination_path)


@click.command()
@click.option('-d', '--data_path', type=str, default='dataset', help='Path to the dataset')
@click.option('-t', '--train_ratio', type=float, default=0.7, help='Proportion of the dataset to use for training')
@click.option('-v', '--val_ratio', type=float, default=0.15, help='Proportion of the dataset to use for validation')
@click.option('-s', '--test_ratio', type=float, default=0.15, help='Proportion of the dataset to use for testing')
@click.option('-c', '--copy', is_flag=True, default=True, help='Copy files instead of moving them')
def main(
    data_path,
    train_ratio,
    val_ratio,
    test_ratio,
    copy,
    **kwargs
):
  split_dataset(data_path, train_ratio, val_ratio, test_ratio, copy)

if __name__ == '__main__':
  main()