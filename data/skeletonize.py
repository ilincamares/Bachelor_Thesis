import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from PIL import Image


def skeletonize_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    print(f"Skeletonizing {len(files)} images...")

    for i, filename in enumerate(files):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

        binary = thresh > 0

        skeleton = skeletonize(binary)

        skeleton_img = np.zeros_like(img)
        skeleton_img[skeleton] = 0  # Ink
        skeleton_img[~skeleton] = 255  # Background

        final_img = Image.fromarray(skeleton_img)
        final_img.save(os.path.join(output_folder, filename))

        if i % 500 == 0:
            print(f"Processed {i}...")

    print("Skeletonization Complete.")