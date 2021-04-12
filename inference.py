from __future__ import print_function
import os
import cv2
from mxnet import gluon
import mxnet as mx
import numpy as np
import json
import time

model_root_dir = '.'
object_detection_model_name = os.environ.get('OBJECT_DETECTION_MODEL_NAME', 'yolo3_mobilenet1.0_coco')


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ObjectDetectionService(object):
    # class attributes
    detector = None
    ctx = mx.cpu() if mx.context.num_gpus() == 0 else mx.gpu()

    @classmethod
    def get_model(cls):
        """
        Get the model object for this instance, loading it if it's not already loaded.

        :return:
        """
        if cls.detector is None:
            # 0.68s in mac pro
            if object_detection_model_name == 'ssd_512_resnet50_v1_coco':
                cls.detector = gluon.nn.SymbolBlock.imports(
                    symbol_file=os.path.join(model_root_dir, 'face_detector_ssd_512_resnet50_v1_coco-symbol.json'),
                    input_names=['data'],
                    param_file=os.path.join(model_root_dir, 'face_detector_ssd_512_resnet50_v1_coco-0000.params'),
                    ctx=cls.ctx
                )
            # 1.3s in mac pro
            elif object_detection_model_name == 'yolo3_darknet53_coco':
                cls.detector = gluon.nn.SymbolBlock.imports(
                    symbol_file=os.path.join(model_root_dir, 'face_detector_yolo3_darknet53_coco-symbol.json'),
                    input_names=['data'],
                    param_file=os.path.join(model_root_dir, 'face_detector_yolo3_darknet53_coco-0000.params'),
                    ctx=cls.ctx
                )
            # 0.58s in mac pro
            elif object_detection_model_name == 'yolo3_mobilenet1.0_coco':
                cls.detector = gluon.nn.SymbolBlock.imports(
                    symbol_file=os.path.join(model_root_dir, 'face_detector_yolo3_mobilenet1.0_coco-symbol.json'),
                    input_names=['data'],
                    param_file=os.path.join(model_root_dir, 'face_detector_yolo3_mobilenet1.0_coco-0000.params'),
                    ctx=cls.ctx
                )
            
            elif object_detection_model_name == 'faster_rcnn_fpn_resnet101_v1d_coco':
                cls.detector = gluon.nn.SymbolBlock.imports(
                    symbol_file=os.path.join(model_root_dir, 'face_detector_faster_rcnn_fpn_resnet101_v1d_coco-symbol.json'),
                    input_names=['data'],
                    param_file=os.path.join(model_root_dir, 'face_detector_faster_rcnn_fpn_resnet101_v1d_coco-0000.params'),
                    ctx=cls.ctx
                )
            else:
                return None
        return cls.detector

    @classmethod
    def predict(cls, resized_rescaled_normalized_img):
        handler = cls.get_model()
        class_ids, mx_scores, mx_bounding_boxes = handler(resized_rescaled_normalized_img)
        return class_ids, mx_scores, mx_bounding_boxes

def test():
    cap = cv2.VideoCapture(0)
    print("capture finished")
    while (1):
        ret, frame = cap.read()
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite("/Users/aaron/01_code/01_industryCV/edgeInference/temp.png", frame)
        cv2.imshow("capture", frame)
    cap.release()
    cv2.destroyAllWindows()

def transformation():

    # cv2.imshow("capture", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.imwrite("/Users/aaron/01_code/01_industryCV/edgeInference/temp.png", frame)
    # cap.release()
    # cv2.destroyAllWindows()

    short_size = 360

    while (1):

        print("start capturing image from camera!")

        t_start = time.time()
        cap = cv2.VideoCapture(0)        
        ret, frame = cap.read()
        cv2.imwrite("/Users/aaron/01_code/01_industryCV/edgeInference/temp.png", frame)

        # pre-process
        img = mx.img.imread("/Users/aaron/01_code/01_industryCV/edgeInference/temp.png")
        height, width, channels = img.shape[0], img.shape[1], img.shape[2]

        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        resized_img = mx.image.resize_short(img, size=short_size)
        resized_rescaled_img = mx.nd.image.to_tensor(resized_img)
        resized_rescaled_normalized_img = mx.nd.image.normalize(resized_rescaled_img, mean=mean, std=std)
        resized_rescaled_normalized_img = resized_rescaled_normalized_img.expand_dims(0)
        if mx.context.num_gpus() != 0:
            resized_rescaled_normalized_img = resized_rescaled_normalized_img.copyto(mx.gpu())

        # inference
        class_ids, mx_scores, mx_bounding_boxes = ObjectDetectionService.predict(resized_rescaled_normalized_img)

        # post-process
        class_ids = class_ids.asnumpy()
        mx_scores = mx_scores.asnumpy()
        mx_bounding_boxes = mx_bounding_boxes.asnumpy()

        # resize detection results back to original image size
        scale_ratio = short_size / height if height < width else short_size / width
        bbox_coords, bbox_scores = list(), list()
        for index, bbox in enumerate(mx_bounding_boxes[0]):
            prob = float(mx_scores[0][index][0])
            if prob < 0.0:
                continue

            [x_min, y_min, x_max, y_max] = bbox
            x_min = int(x_min / scale_ratio)
            y_min = int(y_min / scale_ratio)
            x_max = int(x_max / scale_ratio)
            y_max = int(y_max / scale_ratio)
            bbox_coords.append([x_min, y_min, x_max, y_max])
            bbox_scores.append([prob])

        body = {
            'width': width,
            'height': height,
            'channels': channels,
            'bbox_scores': bbox_scores,  # shape = (N, 1)
            'bbox_coords': bbox_coords,  # shape = (N, 4)
        }
        color = (0, 255, 0)

        for i, val in enumerate(bbox_coords):
            x = val[0]
            y = val[1]
            w = val[2]
            h = val[3]
            # bounding box
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

        for j, val in enumerate(bbox_scores):
            # score text
            text = "score: " + str(val[0])
            AddText = frame.copy()
            cv2.putText(AddText, text, (200, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
        
        # stitch original and texted images
        final = np.hstack([frame, AddText])

        cv2.imwrite("/Users/aaron/01_code/01_industryCV/edgeInference/tempBound.png", final)
        t_end = time.time()

        print('Time consumption = {} second'.format(t_end - t_start))
        print('Response = {}'.format(body))

if __name__ == "__main__":
    # execute only if run as a script
    transformation()
    # test()
