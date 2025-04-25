from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import random

# Motion blur code taken from https://medium.com/geekculture/custom-data-augmentation-using-keras-imagedatagenerator-7cfd58e54171
class CardDataGenerator(ImageDataGenerator):
    def __init__(self,
            h_kernel_size: int = None,
            v_kernel_size: int = None,
            **kwargs) -> None:

        super().__init__(
            preprocessing_function=self.actions,
            **kwargs)

        self.h_kernel_size = h_kernel_size
        self.v_kernel_size = v_kernel_size

    def actions(self, image: np.ndarray) -> np.ndarray:
        # Define Action List
        action_list = ['horizontal_motion_blur', 'vertical_motion_blur', 'lighting_effect', 'none']

        # Random values to select an operation
        operations = np.random.random(len(action_list)).tolist()
        maximum = operations.index(max(operations))
        op = action_list[maximum]

        # Normalize image
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        if op == 'horizontal_motion_blur':
            image = self.horizontal_motion_blur(image)
        elif op == 'vertical_motion_blur':
            image = self.vertical_motion_blur(image)
        elif op == 'lighting_effect':
            image = self.lighting_effect(image)
        else:
            return image

        return image

    def horizontal_motion_blur(self, image: np.ndarray) -> np.ndarray:
        if self.h_kernel_size == None:
            return image

        kernel_h = np.zeros((self.h_kernel_size, self.h_kernel_size))

        # Fill the middle row with ones.
        kernel_h[int((self.h_kernel_size - 1)/2), :] = np.ones(self.h_kernel_size)

        # Normalize.
        kernel_h /= self.h_kernel_size

        # Apply the horizontal kernel.
        horizonal_mb = cv2.filter2D(image, -1, kernel_h)
        horizonal_mb = np.reshape(horizonal_mb, image.shape)

        return horizonal_mb

    def vertical_motion_blur(self, image: np.ndarray) -> np.ndarray:
        if self.v_kernel_size == None:
            return image

        kernel_v = np.zeros((self.v_kernel_size, self.v_kernel_size))

        # Fill the middle row with ones.
        kernel_v[:, int((self.v_kernel_size - 1)/2)] = np.ones(self.v_kernel_size)

        # Normalize.
        kernel_v /= self.v_kernel_size

        # Apply the vertical kernel.
        vertical_mb = cv2.filter2D(image, -1, kernel_v)
        vertical_mb = np.reshape(vertical_mb, image.shape)

        return vertical_mb

    def lighting_effect(self, img: np.ndarray):
      # Randomly adjust brightness, contrast, and shadow intensity
      brightness = random.uniform(0.5, 1.5)
      contrast = random.uniform(0.5, 1.5)
      shadow_intensity = random.uniform(0.2, 0.5)

      # Adjust brightness and contrast
      img = cv2.convertScaleAbs(img, alpha=contrast, beta=int((brightness - 1) * 128))

      # Adding shadow effect
      shadow_overlay = np.zeros_like(img, dtype=np.uint8)
      rows, cols = img.shape[:2]
      shadow_center = (random.randint(0, cols), random.randint(0, rows))
      shadow_radius = random.randint(min(rows, cols) // 4, min(rows, cols) // 2)
      cv2.circle(shadow_overlay, shadow_center, shadow_radius, (0, 0, 0), -1)
      shadow_img = cv2.addWeighted(img, 1 - shadow_intensity, shadow_overlay, shadow_intensity, 0)

      return shadow_img