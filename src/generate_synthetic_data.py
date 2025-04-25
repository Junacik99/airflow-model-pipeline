import click
import os
import numpy as np
import cv2
import random

def lighting_effect(img, brightness=1.0, contrast=1.0, shadow_intensity=0.3):
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


def synthetizer(img_path, num_images=10):
	# Load image
	img = cv2.imread(img_path)

	for _ in range(num_images):
		# Randomly adjust brightness, contrast, and shadow intensity
		brightness_factor = random.uniform(0.5, 1.5)
		contrast_factor = random.uniform(0.5, 1.5)
		shadow_intensity = random.uniform(0.2, 0.5)

		# Add synthetic lighting effect
		synthetized_img = lighting_effect(img, brightness=brightness_factor, contrast=contrast_factor, shadow_intensity=shadow_intensity)

		# Yield sythetized image
		yield synthetized_img
		
def generate(data_path, num_images=10):
    """
    Generate synthetic data from the dataset.
    """
    print(f"Generating synthetic data in {data_path}")

    if os.path.exists(data_path) and os.path.isdir(data_path):
        # Iterate through subdirectories in data_path
        for subdir, _, files in os.walk(data_path):
            for file in files:
                # Check if the file is an image
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(subdir, file)

                    # Generate synthetic images using synthetizer
                    for i, synthetized_img in enumerate(synthetizer(img_path, int(num_images))):
                        # Define the output path for the synthesized image
                        output_path = os.path.join(subdir, f"{os.path.splitext(file)[0]}_synthetic_{i}.jpg")

                        cv2.imwrite(output_path, synthetized_img)
    else:
        raise Exception(f"Path '{data_path}' does not exist or is not a directory.")
    

@click.command()
@click.option('-d', '--data_path', type=str, default='dataset', help='Path to the dataset')
@click.option('-n', '--num_images', type=int, default=10, help='Number of synthetic images to generate')
def main(
    data_path,
    num_images,
    **kwargs
    ):
    generate(data_path, num_images)

if __name__ == '__main__':
    main()