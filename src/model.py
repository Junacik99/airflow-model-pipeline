import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization

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

        self.model = model
