import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, 
    GlobalAveragePooling2D, 
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
)

class CardModel(Model):
    def __init__(self, num_classes: int, img_size: tuple):
        ## Architecture ##
        # Base model  
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=img_size + (3,),
            include_top=False,
            weights='imagenet'
        )

        # Additional layers
        avg = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation='relu')(avg)
        x = BatchNormalization()(x)
        output = Dense(num_classes, activation='softmax')(x)

        # Create model
        model = Model(inputs=base_model.input, outputs=output)
        
        # Freeze base model layers
        base_model.trainable = False

        model.summary()

        self.model = model

class BinaryCardModel(Model):
    def __init__(self, img_size: tuple):
        ## Architecture ##
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=img_size + (3,)),
            Conv2D(64, (3,3),activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.1),

            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.1),

            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.1),

            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.25),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        model.summary()


        self.model = model