import tensorflow as tf
import click


def deploy_model(
    model_path: str = 'model.keras',
    dest_path: str = 'model.tflite'
):   
    # Load a .keras model
    android_model = tf.keras.models.load_model(model_path)
    android_model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(android_model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the .tflite model
    with open(dest_path, "wb") as f:
        f.write(tflite_model)
        print(f"Model converted and saved to {dest_path}")


@click.command()
@click.option('--model_path', default='model.keras', help='Path to the trained model file.')
@click.option('--dest_path', default='model.tflite', help='Path to the destination .tflite model file.')
def main(model_path, dest_path):
    deploy_model(model_path, dest_path)

if __name__ == "__main__":
    main()