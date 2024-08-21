import cv2
import numpy

from recognizer import Recognizer


class Window:
    def __init__(self, title: str) -> None:
        self.captureDevice: cv2.VideoCapture = cv2.VideoCapture(0)
        self.title = title
        self.window = cv2.namedWindow(title)
        self.frame: numpy.ndarray | None = None

    def GetShrinkFrame(self, scale: float = 0.25) -> numpy.ndarray:
        if self.frame is None:
            return numpy.zeros((0, 0))
        return cv2.resize(self.frame, (0, 0), fx=scale, fy=scale)

    def DrawInfo(self, recognizer: Recognizer) -> None:
        if self.frame is None:
            return

        for (top, right, bottom, left), name in zip(
            recognizer.locations, recognizer.names
        ):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(
                self.frame,
                (left, bottom - 35),
                (right, bottom),
                (0, 0, 255),
                cv2.FILLED,
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                self.frame,
                name,
                (left + 6, bottom - 6),
                font,
                1.0,
                (255, 255, 255),
                1,
            )

    def WaitKey(self, key: str) -> int:
        return cv2.waitKey(1) & 0xFF == ord(key)

    def ShowFrame(self) -> None:
        if self.frame is None:
            return
        cv2.imshow(self.title, self.frame)

    def Update(self) -> None:
        self.frame = self.captureDevice.read()[1]

    def Close(self) -> None:
        self.captureDevice.release()
        cv2.destroyAllWindows()
