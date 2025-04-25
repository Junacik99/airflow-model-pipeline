import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization

from src.data_augmentation import CardDataGenerator
from src.model_utils import plot_history

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

    print("Obtraining base model")
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_size + (3,),
                                               include_top=False,
                                               weights='imagenet')
    print("Creating model")
    # Architecture
    avg = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(avg)
    x = BatchNormalization()(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    base_model.trainable = False

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        )
    model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    

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
        epochs=epochs)
    
    plot_history(history, fig_path)
    
    model.save(model_path)
    print("Model saved to", model_path)