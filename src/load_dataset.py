import click
import os

def check_dataset(data_path):
    """
    Load the dataset from the specified path.
    Load number of classes.
    """

    print(f"Checking dataset in {data_path}")

    num_classes = 0
    if os.path.exists(data_path) and os.path.isdir(data_path):
        subfolders = [entry for entry in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, entry))]
        for subfolder in subfolders:
            if subfolder in ['train', 'valid', 'test']:
                continue
            num_classes += 1
            print(f"Class: {subfolder}")
            folder_path = os.path.join(data_path, subfolder)
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
            print(f"Number of images: {len(images)}")
        print(f"Number of classes: {num_classes}")

    else:
        raise Exception(f"Path '{data_path}' does not exist or is not a directory.")
    
    return num_classes


@click.command()
@click.option('-d', '--data_path', type=str, default='dataset', help='Path to the dataset')
def main(
    data_path,
    **kwargs
    ):
    check_dataset(data_path)

if __name__ == '__main__':
    main()