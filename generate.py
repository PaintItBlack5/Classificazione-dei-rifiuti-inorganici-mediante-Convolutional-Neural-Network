from tqdm import tqdm
import os
import random
import argparse

'''
    Generate files following the CUB200 structure:
    - images/
    - images.txt
    - train_test_split.txt
    - image_class_labels.txt
    - classes.txt

Usage:
python generate.py <directory>

'''


def split_train_test_for_folders(args):
    folder_images = os.path.join(args.root_dir, "images")

    if not 0 < args.train_ratio < 1:
        raise ValueError("Il rapporto di addestramento deve essere compreso tra 0 e 1 esclusi.")
    
    _classes_file = open(args.classes_file, 'w')
    _image_class_labels_file = open(args.labels_file, 'w')
    _images_file = open(args.images_file, 'w')
    _train_test_split_file = open(args.split_file, 'w')

    subfolders = sorted([folder_name for folder_name in os.listdir(folder_images) if os.path.isdir(os.path.join(folder_images, folder_name))])
    idx_file = 1
    for idx_class, folder_name in enumerate(tqdm(subfolders, desc="Processing folders"), start=1):
        folder_path = os.path.join(folder_images, folder_name)
        if os.path.isdir(folder_path):
            class_id = str(int(folder_name.split(".")[0]))
            if idx_class == int(class_id):
                
                #print(class_id)
                files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
                num_train = int(len(files) * args.train_ratio)
                train_indices = random.sample(range(len(files)), num_train)

                classes_text = f"{class_id} {folder_name}\n"
                
                _classes_file.write(classes_text)

                for i, file in enumerate(tqdm(files, desc=f"Processing files in {folder_name}", leave=False)):
                    is_train_image = '1' if i in train_indices else '0'
                    split_text = f"{idx_file} {is_train_image}\n"
                    _train_test_split_file.write(split_text)

                    images_text = f"{idx_file} {os.path.join(folder_name, file)}\n"
                    _images_file.write(images_text)

                    image_class_text = f"{idx_file} {class_id}\n"
                    _image_class_labels_file.write(image_class_text)
                    
                    idx_file += 1
            else:
                raise ValueError("Classes directory are not starting with 001. 002. 003. etc!")

if __name__ == "__main__":
    folder_path = ""
    
    parser = argparse.ArgumentParser(description='Generate files for CUB200 structure')
    parser.add_argument('root_dir', type=str, help='Path to the root folder containing image folders.')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of images to be used for training (default: 0.8)')
    parser.add_argument('--classes_file', type=str, default="classes.txt", help='File containing classes')
    parser.add_argument('--labels_file', type=str, default="image_class_labels.txt", help='File containing image class labels')
    parser.add_argument('--images_file', type=str, default="images.txt", help='File containing images paths')
    parser.add_argument('--split_file', type=str, default="train_test_split.txt", help='File containing train and test split informations')

    args = parser.parse_args()

    
    print(f"Start generating files from {args.root_dir}/")
    split_train_test_for_folders(args)

    print("All done")