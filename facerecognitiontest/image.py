import face_recognition as FaceRecognition
import cv2
import numpy
import os


class ImageLoader:
    def __init__(self) -> None:
        self.images: dict[str, numpy.ndarray] = {}

    def LoadImageFile(self, name: str, path: str) -> None:
        img = FaceRecognition.load_image_file(path)
        if len(FaceRecognition.face_encodings(img)) == 0:
            raise ValueError("No face found in the image")

        if name in self.images.keys():
            raise ValueError("Name already exists in the image set")

        self.images[name] = FaceRecognition.face_encodings(img)[0]

    def GetFiles(self, directory: str) -> list[str]:
        return [f"{directory}/{path}" for path in os.listdir(directory)]

    def GenerateFaceSetFromDirectory(self, directory: str) -> None:
        """This function is used for mocking..."""
        for path in self.GetFiles(directory):
            name = path.split("/")[-1].split(".")[0]
            self.LoadImageFile(name, path)
