import face_recognition as FaceRecognition
import numpy


class Recognizer:
    """
    face recognition を用いて顔認識を行うクラス
    """

    def __init__(self, knownFaces: dict[str, numpy.ndarray], interval: int = 5) -> None:
        """
        knownFaces: 既知の顔の特徴量を格納した辞書
        interval: 何フレームごとに顔認識を行うか
        """
        self.knownFaces: dict[str, numpy.ndarray] = knownFaces
        self.names: list[str] = []
        self.locations: list[tuple[int, int, int, int]] = []
        self.clock: int = 0
        self.interval: int = interval

    def FindFaces(
        self, currentFrame: numpy.ndarray
    ) -> tuple[list[numpy.ndarray], list[tuple[int, int, int, int]]]:
        """
        currentFrame: 現在のフレーム
        return: 顔の特徴量と位置

        顔の特徴量と位置を返す
        """
        locations = FaceRecognition.face_locations(currentFrame)
        return FaceRecognition.face_encodings(currentFrame, locations), locations

    def CompareFaces(
        self,
        target: numpy.ndarray,
        threshould: float = 0.35,
    ) -> str:
        """
        target: 比較する顔の特徴量
        threshould: 顔の特徴量の閾値
        return: 顔の名前

        顔の特徴量を比較して名前を返す
        """
        names = list(self.knownFaces.keys())
        faces = list(self.knownFaces.values())
        matches = FaceRecognition.compare_faces(faces, target, threshould)
        distances = FaceRecognition.face_distance(faces, target)
        index = numpy.argmin(distances)
        if matches[index]:
            return names[index]
        return "Unknown"

    def ProcessFrame(self, frame: numpy.ndarray) -> None:
        """
        frame: 現在のフレーム

        顔認識を行い、名前と位置をself.namesとself.locationsに保存する
        """
        if self.clock % self.interval == 0:
            faces, locations = self.FindFaces(frame)
            self.names = [self.CompareFaces(face) for face in faces]
            self.locations = locations
            self.clock = 0
        self.clock += 1
