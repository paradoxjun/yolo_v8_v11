import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from annotate.process_image import get_img_files


def process_single_image(image_path, target_image_dir, target_label_dir):
    label_path = image_path.replace('images', 'labels')[:-3] + 'txt'
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Skip if only one line or empty
    if len(lines) <= 1:
        return

    image_name = os.path.basename(image_path)
    label_name = image_name[:-3] + 'txt'

    # Modify labels if more than 2 lines
    modified_lines = []
    for line in lines:
        values = line.strip().split()
        if len(values) >= 5:
            values[0] = '0'  # Modify class to 0
            modified_lines.append(' '.join(values) + '\n')

    # Save modified label file
    target_label_file = os.path.join(target_label_dir, label_name)
    with open(target_label_file, 'w') as f:
        f.writelines(modified_lines)

    # Copy image file to target directory
    shutil.copy(image_path, os.path.join(target_image_dir, image_name))


def process_yolo_labels(file_list, target_image_dir, target_label_dir):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for image_path in file_list:
            futures.append(executor.submit(process_single_image, image_path, target_image_dir, target_label_dir))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            future.result()  # This will raise an exception if the task failed


mode = 'test'
file_list = get_img_files(fr'F:\datasets\hagrid\yolo_det\{mode}\images')
target_image_dir = fr'F:\datasets\hagrid\2hand_image\{mode}\images'
target_label_dir = fr'F:\datasets\hagrid\2hand_image\{mode}\labels'

process_yolo_labels(file_list, target_image_dir, target_label_dir)
