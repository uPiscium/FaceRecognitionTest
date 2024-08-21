import face_recognition as FaceRecognition
import numpy


class Recognizer:
    def __init__(self, knownFaces: dict[str, numpy.ndarray], interval: int = 5) -> None:
        self.knownFaces: dict[str, numpy.ndarray] = knownFaces
        self.names: list[str] = []
        self.locations: list[tuple[int, int, int, int]] = []
        self.clock: int = 0
        self.interval: int = interval

    def FindFaces(
        self, currentFrame: numpy.ndarray
    ) -> tuple[list[numpy.ndarray], list[tuple[int, int, int, int]]]:
        locations = FaceRecognition.face_locations(currentFrame)
        return FaceRecognition.face_encodings(currentFrame, locations), locations

    def CompareFaces(
        self,
        target: numpy.ndarray,
        threshould: float = 0.35,
    ) -> str:
        names = list(self.knownFaces.keys())
        faces = list(self.knownFaces.values())
        matches = FaceRecognition.compare_faces(faces, target, threshould)
        distances = FaceRecognition.face_distance(faces, target)
        index = numpy.argmin(distances)
        if matches[index]:
            return names[index]
        return "Unknown"

    def ProcessFrame(self, frame: numpy.ndarray) -> None:
        if self.clock % self.interval == 0:
            faces, locations = self.FindFaces(frame)
            self.names = [self.CompareFaces(face) for face in faces]
            self.locations = locations
            self.clock = 0
        self.clock += 1
