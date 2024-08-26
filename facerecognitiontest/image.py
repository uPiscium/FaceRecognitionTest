import face_recognition as FaceRecognition
import numpy
import os


class ImageLoader:
    """
    画像を読み込み、顔のエンコーディングを取得し、データを保存するクラス
    """

    def __init__(self) -> None:
        self.images: dict[str, numpy.ndarray] = {}  # 画像の名前とエンコーディングの辞書

    def LoadImageFile(self, name: str, path: str) -> None:
        """
        name: 写真に対応する名前
        path: 画像のパス

        画像を読み込み、顔のエンコーディングを取得し、データをself.imagesに保存する
        """
        img = FaceRecognition.load_image_file(path)
        if len(FaceRecognition.face_encodings(img)) == 0:
            raise ValueError("No face found in the image")

        if name in self.images.keys():
            raise ValueError("Name already exists in the image set")

        self.images[name] = FaceRecognition.face_encodings(img)[0]

    def GetFiles(self, directory: str) -> list[str]:
        """
        directory: ディレクトリのパス

        ディレクトリ内のファイルのパスを取得する
        この関数はローカル環境でのテスト実行時にモックするために使用されます
        """
        return [f"{directory}/{path}" for path in os.listdir(directory)]

    def GenerateFaceSetFromDirectory(self, directory: str) -> None:
        """
        directory: 画像が保存されているディレクトリのパス

        ディレクトリ内の画像を読み込み、顔のエンコーディングを取得し、データを保存する
        この関数はローカル環境でのテスト実行時にモックするために使用されます
        """
        """This function is used for mocking..."""
        for path in self.GetFiles(directory):
            name = path.split("/")[-1].split(".")[0]
            self.LoadImageFile(name, path)
