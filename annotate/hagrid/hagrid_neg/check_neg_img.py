import annotate.process_image as img_ops
from load_model import predictor_ren, predictor_shou
from add_mask import apply_mask_hand


def save_negative_images(img_root_list):
    for img_root in img_root_list:
        img_path_list = img_ops.get_img_files(img_root)

        for img_path in img_path_list:
            is_modify = False
            img = img_ops.image_read_cv2(img_path)
            ren = predictor_ren(img)[0]

            if ren.keypoints is None:
                continue

            hands = predictor_shou(img)[0]
            is_modify += apply_mask_hand(img, hands.boxes.xyxy)

            for i, bbox in enumerate(ren.boxes.xyxy):
                x1, y1, x2, y2 = list(map(int, bbox))
                img_ren = img[y1:y2, x1:x2]
                person_hands = predictor_shou(img_ren)[0]
                is_modify += apply_mask_hand(img_ren, person_hands.boxes.xyxy)

            if is_modify:
                img_ops.image_save_cv2(img, img_path, True)


if __name__ == '__main__':
    img_root_list = [r'datasets/neg_hand/train',
                     r'datasets/neg_hand/val']

    save_negative_images(img_root_list)
