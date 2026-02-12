import os
from PIL import Image, ImageDraw, ImageFont
import openpyxl
import pandas as pd

FONT_PATH = "fonts/NotoSansSC-Regular.ttf"
IMAGE_SIZE = 96
FONT_SIZE = 76


def generate_dataset(output_folder, character_file_path):
    df = pd.read_excel(character_file_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except OSError:
        print(f"Error: Could not find font file at {FONT_PATH}")
        return

    print(f"Generating images for {df.shape[0]} characters...")


    for i, row in df.iterrows():
        char = row['Character'][0]
        img = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), color=255)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (IMAGE_SIZE - text_width) / 2
        y = (IMAGE_SIZE - text_height) / 2 - bbox[1]

        if char == "一":
            y += bbox[1]/4

        draw.text((x, y), char, font=font, fill=0)

        unicode_name = f"U+{ord(char):X}"
        filename = f"{output_folder}/{unicode_name}.png"
        img.save(filename)

    print("Done!")