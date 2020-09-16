"""Use openCV to detect faces in an image."""

import argparse
from configparser import ConfigParser

import cv2


class FaceDetector:
    def __init__(self, face_cascade_path):
        # load the face detector
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def detect(self, image):
        faces = self.face_cascade.detectMultiScale(image)
        # if len(faces) > 1:
        #     raise RuntimeError("Detect multiple faces in the input video!")

        img_h = image.shape[0]
        img_w = image.shape[1]
        for (x, y, w, h) in faces:
            x = max(x - w // 3, 0)
            y = max(y - h // 3, 0)
            w = min(int(w * 1.7), img_w - 1 - x)
            h = min(int(h * 1.7), img_h - 1 - y)
            # image = cv2.rectangle(image, (x, y), (x + int(w), y + int(h)), color=(255, 0, 255))
            return x, y, min(w, h), min(w, h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="the image to be detected")
    parser.add_argument("-o", "--output", type=str, help="the path to save detection results")
    args = parser.parse_args()

    # construct the face detector
    face_detector = FaceDetector("haarcascade_frontalface_alt.xml")
    x, y, w, h = face_detector.detect(cv2.imread(args.image))

    # write into ../.cropsource or ../.croptarget
    config = ConfigParser()
    config.add_section("Credentials")
    config["Credentials"]["x"] = str(x)
    config["Credentials"]["y"] = str(y)
    config["Credentials"]["w"] = str(w)
    config["Credentials"]["h"] = str(h)
    with open(args.output, "w") as file:
        config.write(file)
