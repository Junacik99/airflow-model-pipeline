from tensorflow.keras.models import load_model
import click
from src.data_augmentation import CardDataGenerator


def evaluate_model(
    model_path: str = 'model.keras',
    test_dir: str = 'dataset/test',
    img_width: int = 224,
    img_height: int = 224,
    batch_size: int = 32
):
    img_width = int(img_width)
    img_height = int(img_height)
    batch_size = int(batch_size)

    # Create a data generator for the test dataset
    test_datagen = CardDataGenerator()
    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

    # Load the model
    model = load_model(model_path)

    # Evaluate the model on the test dataset
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.4f}")


@click.command()
@click.option('--model_path', default='model.keras', help='Path to the trained model file.')
@click.option('--test_dir', default='dataset/test', help='Path to the test dataset directory.')
@click.option('--img_width', default=224, help='Width of the input images.')
@click.option('--img_height', default=224, help='Height of the input images.')
@click.option('--batch_size', default=32, help='Batch size for evaluation.')
def main(model_path, test_dir, img_width, img_height, batch_size):
    evaluate_model(model_path, test_dir, img_width, img_height, batch_size)

if __name__ == "__main__":
    main()