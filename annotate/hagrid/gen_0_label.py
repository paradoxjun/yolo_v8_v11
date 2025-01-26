import os
import shutil
import zipfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from annotate.process_image import get_img_files


def process_single_image(image_path, source_label_dir, target_label_dir):
    label_path = image_path.replace('images', 'labels')[:-3] + 'txt'

    with open(label_path, 'r') as f:
        lines = f.readlines()

    save_label_path = image_path.replace(source_label_dir, target_label_dir).replace('images', 'labels')[:-3] + 'txt'
    if not os.path.exists(os.path.dirname(save_label_path)):
        os.makedirs(os.path.dirname(save_label_path))

    # Modify labels if more than 2 lines
    modified_lines = []
    for line in lines:
        values = line.strip().split()
        if len(values) >= 5:
            values[0] = '0'  # Modify class to 0
            modified_lines.append(' '.join(values) + '\n')

    with open(save_label_path, 'w') as f:
        f.writelines(modified_lines)


def process_yolo_labels(file_list, source_label_dir, target_label_dir):
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for image_path in file_list:
            futures.append(executor.submit(process_single_image, image_path, source_label_dir, target_label_dir))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            future.result()  # This will raise an exception if the task failed


# F:\datasets\hagrid\yolo_det\val\images\call
# F:\datasets\hagrid\yolo_det_label\val\labels

mode = 'train'
file_list = get_img_files(fr'F:\datasets\hagrid\yolo_det\{mode}\images')

process_yolo_labels(file_list, source_label_dir='yolo_det', target_label_dir='yolo_det_label')


def zip_folder(folder_path, output_zip_path):
    # Create a ZipFile object
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the folder
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Create a relative path to store in the zip file
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, relative_path)


folder_path = r'F:\datasets\hagrid\yolo_det_label'  # Replace with the path to the folder you want to zip
output_zip_path = r'F:\datasets\hagrid\yolo_det_label.zip'  # Replace with the desired name of the zip file

zip_folder(folder_path, output_zip_path)
