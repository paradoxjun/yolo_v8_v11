import cv2
import numpy as np
import onnxruntime as ort


class YOLOv7TinyDetector:
    def __init__(self, model_path, input_size=(224, 224), conf_threshold=0.25, nms_threshold=0.8):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.session = ort.InferenceSession(model_path)

    def preprocess_image(self, img):
        h, w, _ = img.shape
        scale = min(self.input_size[1] / h, self.input_size[0] / w)
        nh, nw = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        new_img = np.full((self.input_size[1], self.input_size[0], 3), 114, dtype=np.uint8)
        new_img[(self.input_size[1] - nh) // 2: (self.input_size[1] - nh) // 2 + nh,
        (self.input_size[0] - nw) // 2: (self.input_size[0] - nw) // 2 + nw, :] = img_resized

        img = new_img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img, scale, (self.input_size[1] - nh) // 2, (self.input_size[0] - nw) // 2

    def postprocess(self, output, img_shape, scale, pad_y, pad_x):
        boxes = output[0][:, :4]
        scores = output[0][:, 4]
        class_ids = output[0][:, 5]

        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= scale

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.nms_threshold)

        result_boxes = []
        result_scores = []
        result_class_ids = []

        for i in indices:
            i = i[0]
            box = boxes[i].astype(int)
            result_boxes.append(box)
            result_scores.append(scores[i])
            result_class_ids.append(int(class_ids[i]))

        return result_boxes, result_scores, result_class_ids

    def detect(self, img):
        input_tensor, scale, pad_y, pad_x = self.preprocess_image(img)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        print(outputs)
        result_boxes, result_scores, result_class_ids = self.postprocess(outputs, img.shape, scale, pad_y, pad_x)
        return result_boxes, result_scores, result_class_ids

    def __call__(self, img):
        if isinstance(img, str):
            img = cv2.imread(img)
            boxes, scores, class_ids = self.detect(img)
            results = [boxes, scores, class_ids]
            return results
        elif isinstance(img, list):
            results = []
            for im in img:
                if isinstance(im, str):
                    im = cv2.imread(im)
                boxes, scores, class_ids = self.detect(im)
                results.append((boxes, scores, class_ids))
            return results
        elif isinstance(img, np.ndarray):
            return self.detect(img)
        else:
            raise ValueError("Unsupported input type")


# 示例使用
if __name__ == '__main__':
    # model_path = '/home/chenjun/code/ultralytics_YOLOv8/weights/general_PPLCNetV2_base_pretrained_v1.0_infer/inference.onnx'
    model_path = '../../../weights/hand_gesture/YoloV7Tiny.onnx'
    detector = YOLOv7TinyDetector(model_path)

    # 单张图片
    img_path = r'../../assets/girl.png'
    res = detector(img_path)
    print(res)
    boxes, scores, class_ids = detector(img_path)
    print(f"Boxes: {boxes}")
    print(f"Scores: {scores}")
    print(f"Class IDs: {class_ids}")
