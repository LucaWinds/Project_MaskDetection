import os
import numpy as np
import cv2


def main():
    capture = cv2.VideoCapture(0)  # カメラ
    if not capture.isOpened():
        exit()

    dir = "D:\Project\Project_MaskDetection\detector\yunet"+'\\'+"face_detection_yunet_2022mar.onnx"
    face_detector = cv2.FaceDetectorYN.create(dir, "", (150, 150))

    while True:
        result, image = capture.read()
        if result is False:
            cv2.waitKey(0)
            break

        height, width, _ = image.shape
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(image)
        faces = faces if faces is not None else []

        for face in faces:
            box = list(map(int, face[:4]))
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)

            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 2
            cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)

        cv2.imshow("face detection", image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()