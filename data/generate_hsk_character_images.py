from data.generate_character_images import generate_dataset
from data.skeletonize import skeletonize_images

if __name__ == "__main__":
    generate_dataset(output_folder="hsk_dataset_images", character_file_path="character_files/HSK_Character_List.xlsx")
    skeletonize_images(input_folder="hsk_dataset_images", output_folder="hsk_dataset_skeletons")