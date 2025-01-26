import annotate.process_image as img_ops
from load_model import predictor_ren, predictor_shou
from get_hand_area import get_arm_keypoint
from add_mask import apply_mask_body, apply_mask_hand
from os.path import isfile, basename
from os.path import join as path_join


def save_negative_images(img_root_list, train_path, val_path, ratio=0.9):
    num_train, num_val = 0, 0   # 训练集和测试集图片数量

    for img_root in img_root_list:
        img_path_list = img_ops.get_img_files(img_root)

        for img_path in img_path_list:
            img = img_ops.image_read_cv2(img_path)
            img_name = basename(img_path)

            train_save_path = path_join(train_path, img_name)
            val_save_path = path_join(val_path, img_name)

            if isfile(train_save_path):
                print(f"图片已经存在：{train_save_path}")
                continue

            if isfile(val_save_path):
                print(f"图片已经存在：{val_save_path}")
                continue

            if num_train * (1 - ratio) <= num_val * ratio:
                save_path = train_save_path
                num_train += 1
            else:
                save_path = val_save_path
                num_val += 1

            ren = predictor_ren(img)[0]
            hands = predictor_shou(img)[0]
            data = get_arm_keypoint(ren.keypoints.xy)

            apply_mask_body(img, ren.boxes.xyxy, data)
            apply_mask_hand(img, hands.boxes.xyxy)

            img_ops.image_save_cv2(img, save_path)


if __name__ == '__main__':
    img_root_list = [r'datasets/bank_monitor/seed_img/finish',
                     'datasets/COCO2017/det_ren/images',
                     'datasets/VOC/VOCtrainval_11-May-2012/VOCdevkit/VOC2012',
                     'datasets/bank_monitor/data_for_train_and_val/2024-04-25',
                     'datasets/bank_monitor/data_for_train_and_val/2024-05-10',
                     'datasets/bank_monitor/data_for_train_and_val/2024-05-29',
                     'datasets/bank_monitor/data_for_train_and_val/2024-06-22-all',
                     'datasets/bank_monitor/data_for_train_and_val/2024-07-01-all',
                     'datasets/bank_monitor/2024-05-28-柜台内结账场景-collect'
                     ]

    train_path = r'datasets/neg_hand/train'
    val_path = r'datasets/neg_hand/val'

    save_negative_images(img_root_list, train_path, val_path)
