import glob
import cv2
import os
import numpy as np
from pathlib import Path

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}\n"


def get_img_files(img_path):
    """
    Read image files. 获取一个目标下所有图片。
    修改自: ultralytics/data/base.py/BaseDataset.get_img_files()
    """
    try:
        f = []  # image files
        for p in img_path if isinstance(img_path, list) else [img_path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                # F = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                    # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            else:
                raise FileNotFoundError(f"Data path: {p} does not exist")
        im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
        # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
        assert im_files, f"No images found in {img_path}. {FORMATS_HELP_MSG}"
    except Exception as e:
        raise FileNotFoundError(f"Error loading data from {img_path}\n") from e

    return im_files


def image_read_cv2(image_path):
    # 使用 OpenCV 读取图片
    image = cv2.imread(image_path)

    # 检查图片是否成功读取
    if image is None:
        print("Error: Could not read image.")
        return None

    return image


def image_show_cv2(image, window_name="Image"):
    """
    显示指定的图片
    :param image: 要显示的图片
    :param window_name: 显示窗口的名称
    """
    if isinstance(image, str):
        image = image_read_cv2(image)

    # 使用 OpenCV 显示图片
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_save_cv2(image, save_path, override=False):
    """
    保存指定的图片
    :param image: 要保存的图片
    :param save_path: 保存图片的文件路径
    :param override: 存在则覆盖
    """
    if not override and os.path.isfile(save_path):
        raise ValueError('save_path is exist. ')

    cv2.imwrite(save_path, image)
    print(f"Image saved at {save_path}")


def apply_gaussian_blur(image, ksize=(15, 15), sigmaX=0):
    """
    对指定的图片应用高斯模糊
    :param image: 要处理的图片
    :param ksize: 高斯核的大小
    :param sigmaX: 高斯核在 X 方向的标准差
    :return: 处理后的图片
    """
    if isinstance(image, str):
        image = image_read_cv2(image)

    blurred_image = cv2.GaussianBlur(image, ksize, sigmaX)

    return blurred_image


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scalefill=False, scaleup=True, stride=32):
    """
    调整图片大小并填充以适应目标尺寸。
    :param im:输入图片。
    :param new_shape:目标形状，默认 (640, 640)。
    :param color:填充颜色，默认 (114, 114, 114)。
    :param auto:自动调整填充，保持最小矩形。True会让图片宽高是stride的最小整数倍，比如32，可以方便卷积。
    :param scalefill:是否拉伸填充。在auto是False时，True会让图片拉伸变形。
    :param scaleup:是否允许放大。False让图片只能缩小。
    :param stride:步幅大小，默认 32。
    :return:返回调整后的图片，缩放比例(宽，高)和填充值。
    """
    shape = im.shape[:2]    # 获取当前图片的形状 [高度, 宽度]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])   # 缩放比例 (新尺寸 / 旧尺寸)
    if not scaleup:         # 如果不允许放大，只进行缩小 (提高验证的 mAP)
        r = min(r, 1.0)

    ratio = r, r    # 计算填充宽度和高度的缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))      # 新的未填充尺寸 (宽度, 高度)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]   # 计算宽高方向的填充值
    if auto:        # 如果设置为自动，保持最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)     # 使填充值是步幅的倍数
    elif scalefill:     # 如果拉伸填充，完全填充
        dw, dh = 0.0, 0.0   # 不进行填充
        new_unpad = (new_shape[1], new_shape[0])    # 未填充的尺寸就是目标尺寸
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]    # 计算宽高的缩放比例

    dw /= 2     # 将填充值均分到两侧
    dh /= 2     # 将填充值均分到上下

    if shape[::-1] != new_unpad:    # 如果当前形状和新的未填充形状不同，则调整大小
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))    # 计算上下填充的像素数
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))    # 计算左右填充的像素数
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加填充边框，填充值为指定颜色

    return im, ratio, (dw, dh)      # 返回调整后的图片，缩放比例和填充值


def letter_reverse(im, ratio, padding):
    """
    将经过 letterbox 函数处理后的图片复原
    :param im: 调整后的图片
    :param ratio: 缩放比例 (宽, 高)
    :param padding: 填充值 (dw, dh)
    :return: 复原后的图片
    """
    # 移除填充
    h, w = im.shape[:2]
    dw, dh = padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = im[top: h-bottom, left: w-right]

    # 获取实际图片的高和宽
    h = int(im.shape[1] / ratio[1])
    w = int(im.shape[0] / ratio[0])

    # 调整大小至原始尺寸
    im = cv2.resize(im, (h, w), interpolation=cv2.INTER_LINEAR)

    return im


def point_reverse(x, y, ratio, padding):
    """
    复原经过 letterbox 处理后的点的坐标
    :param x: 调整后的点的 x 坐标
    :param y: 调整后的点的 y 坐标
    :param ratio: 缩放比例 (宽, 高)
    :param padding: 填充值 (dw, dh)
    :return: 复原后的点的坐标 (原始图片上的坐标)
    """
    # 移除填充
    dw, dh = padding
    x = x - dw
    y = y - dh

    # 逆向缩放
    x = int(x / ratio[0])
    y = int(y / ratio[1])

    return x, y


if __name__ == '__main__':
    img_path = 'datasets/bank_monitor/seed_img/finish/0cf015.jpg'
    img_save = 'datasets/bank_monitor/kx_1.jpg'

    image = image_read_cv2(image_path=img_path)
    print(image.shape)
    image_show_cv2(image)
    image_letter, ratio, dwh = letterbox(image, auto=False)
    print(image_letter.shape)
    image_show_cv2(image_letter)
    image_letter_reverse = letter_reverse(image_letter, ratio, dwh)
    print(image_letter_reverse.shape)
    image_show_cv2(image_letter_reverse)
