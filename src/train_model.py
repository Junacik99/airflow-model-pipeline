from tensorflow.keras.optimizers import Adam
import click

from src.data_augmentation import CardDataGenerator
from src.model_utils import plot_history
from src.model import CardModel

def train_model(
    img_width: int = 224,
    img_height: int = 224,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 10,
    num_classes: int = 10,
    train_dir: str = 'dataset/train',
    valid_dir: str = 'dataset/valid',
    test_dir: str = 'dataset/test',
    model_path: str = 'model.keras',
    fig_path: str = 'model_history.png'
):
    img_size = (int(img_width), int(img_height))
    num_classes = int(num_classes)
    batch_size = int(batch_size)
    epochs = int(epochs)
    learning_rate = float(learning_rate)

    print("Creating model")
    model = CardModel(num_classes=num_classes, img_size=img_size).model

    optimizer = Adam(
        learning_rate=learning_rate,
    )

    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
    )
    

    print("Loading dataset")
    # Data augmentation and preprocessing
    train_datagen = CardDataGenerator(
        rotation_range=40,  
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest',
        v_kernel_size=30,
        h_kernel_size=25
    )
    valid_datagen = CardDataGenerator()
    test_datagen = CardDataGenerator()

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print("Training model")
    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        batch_size=batch_size,
        epochs=epochs
    )
    
    plot_history(history, fig_path)
    
    model.save(model_path)
    print("Model saved to", model_path)


@click.command()
@click.option('--img_width', default=224, help='Image width')
@click.option('--img_height', default=224, help='Image height')
@click.option('--learning_rate', default=0.001, help='Learning rate')
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--epochs', default=10, help='Number of epochs')
@click.option('--num_classes', default=10, help='Number of classes')
@click.option('--train_dir', default='dataset/train', help='Training directory')
@click.option('--valid_dir', default='dataset/valid', help='Validation directory')
@click.option('--test_dir', default='dataset/test', help='Testing directory')
@click.option('--model_path', default='model.keras', help='Path to save the model')
@click.option('--fig_path', default='model_history.png', help='Path to save the training history plot')
def main(img_width, img_height, learning_rate, batch_size, epochs, num_classes, train_dir, valid_dir, test_dir, model_path, fig_path):
    train_model(
        img_width=img_width,
        img_height=img_height,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        num_classes=num_classes,
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        model_path=model_path,
        fig_path=fig_path
    )

if __name__ == '__main__':
    main()