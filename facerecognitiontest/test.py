from image import ImageLoader
from recognizer import Recognizer
from window import Window

# Load the image
loader = ImageLoader()
loader.GenerateFaceSetFromDirectory("facerecognitiontest/images")

# Create the recognizer
recognizer = Recognizer(loader.images)

# Create the window
window = Window("Face Recognition Test")

while True:
    window.Update()
    frame = window.GetShrinkFrame()
    recognizer.ProcessFrame(frame)
    window.DrawInfo(recognizer)
    window.ShowFrame()

    if window.WaitKey("q"):
        break

window.Close()
