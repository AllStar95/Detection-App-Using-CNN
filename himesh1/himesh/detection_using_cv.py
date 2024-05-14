import cv2
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent

def detect_image(path: str,  model, class_labels, integrated=False):
    image = cv2.imread(path)

    # model.setInputSize(image.shape[1], image.shape[0])
    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    class_index, confidence, bbox = model.detect(image, confThreshold=0.5)
    print(class_index)

    font_scale = 1
    font_face = cv2.FONT_HERSHEY_COMPLEX

    for index, conf, box in zip(class_index.flatten(), confidence.flatten(), bbox):
        cv2.rectangle(image, box, (255, 0, 0), 2)
        cv2.putText(image, class_labels[index - 1], (box[0] + 10, box[1] + 30), font_face, fontScale=font_scale,
                    color=(0, 255, 0), thickness=2)
        cv2.putText(image, str(conf), (box[0] + 150, box[1] + 30), font_face, font_scale, (0, 255, 0), thickness=2)

    if not integrated:
        cv2.imshow("Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image

def detect_video(path: str, model, class_labels, use_cam=False, capture_code=0):
    if not use_cam:
        cap = cv2.VideoCapture(path)
    else:
        cap = cv2.VideoCapture(capture_code)
        cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        raise IOError('Failed to open the video file!')

    cap.set(3, 640 * 2)  # set width as 640
    cap.set(4, 480 * 2)  # set height as 480

    font_scale = 1
    font_face = cv2.FONT_HERSHEY_COMPLEX

    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    while cap.isOpened():
        ret, frame = cap.read()
        class_index, confidence, bbox = model.detect(frame, confThreshold=0.5)

        if len(class_index) and len(confidence):
            for index, conf, box in zip(class_index.flatten(), confidence.flatten(), bbox):
                if index > len(class_labels):
                    continue
                cv2.rectangle(frame, box, (255, 0, 0), 2)
                cv2.putText(frame, class_labels[index - 1], (box[0] + 10, box[1] + 30), font_face, fontScale=font_scale, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, str(conf), (box[0] + 150, box[1] + 30), font_face, font_scale, (0, 255, 0), thickness=2)

        cv2.imshow('Detection', frame)

        if cv2.waitKey(2) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    config_file = ROOT_DIR / 'model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = ROOT_DIR / 'model/frozen_inference_graph.pb'

    model = cv2.dnn_DetectionModel(str(frozen_model), str(config_file))

    class_labels = []
    label_file = ROOT_DIR / 'model/labels.txt'
    with open(label_file, 'rt') as fpt:
        class_labels = fpt.read().rstrip('\n').split('\n')

    print(class_labels)
    print(len(class_labels))

    # detect_image('/Users/flameberry/Downloads/IMG_1225.jpg', model, class_labels)
    # detect_image(str(ROOT_DIR) + '/vendor/Mask_RCNN-TF2/images/25691390_f9944f61b5_z.jpg', model, class_labels)
    detect_video('/Users/flameberry/Downloads/IMG_0876.MOV', model, class_labels)
