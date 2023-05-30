import tensorflow as tf
import numpy as np
import cv2
import winsound

# Preprocess image
def reimg(img):
    img_resized = cv2.resize(img, (150, 150), interpolation=cv2.INTER_NEAREST)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_reshaped = img_rgb.reshape(-1, 150, 150, 3)
    re_img = img_reshaped.astype(np.float32) / 255.0

    return re_img


# Load the model
model = tf.keras.models.load_model('../model/maskmodel.h5')

# Setting The Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 176)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 144)

# Not using now
cnt = 0

# Load Yunet face detector
detector = cv2.FaceDetectorYN.create("./yunet/face_detection_yunet_2022mar.onnx", "", (176, 144))

while cap.isOpened():
    cnt += 1
    # get the frame
    ret, frame = cap.read()

    detector.setInputSize((176, 144))

    # detect face
    _, faces = detector.detect(frame)
    faces = faces if faces is not None else []

    for face in faces:
        # Setting the Output
        box = list(map(int, face[:4]))
        color = (0, 0, 255)
        color2 = (255, 0, 0)
        thickness = 1

        # Preprocessing the image
        img = reimg(frame)

        # Predict the result
        mask = model.predict(img)

        # Detect the result
        if mask > 0.6:
            print('WITHOUT MASKED', mask)
            cv2.rectangle(frame, box, color, thickness, cv2.LINE_AA)
            cv2.putText(frame, 'WITHOUT MASKED', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1, color)

        else:
            print('MASKED', mask)
            cv2.rectangle(frame, box, color2, thickness, cv2.LINE_AA)
            cv2.putText(frame, 'MASKED', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1, color2)

    # show the result
    cv2.imshow('mask detect', frame)

    key = cv2.waitKey(10)

    # Press Q to Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
