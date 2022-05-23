# from cv2 import *  # Import functions from OpenCV
from PIL import Image
from imageio import imread, imsave
import cv2
# source = Image.open('medianFiltered.jpg')


def medFiltSeq():
    source = cv2.imread("medianFiltered.jpg")
    final = cv2.medianBlur(source, 3)
    imsave("medianFilteredSeq.jpg", final)
    print("saved")


if __name__ == '__main__':
    medFiltSeq()
