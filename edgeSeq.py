import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image


def sobelOperator(img):
    container = np.copy(img)
    size = container.shape
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - \
                (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
            gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - \
                (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
            container[i][j] = min(255, np.sqrt(gx**2 + gy**2))
    return container
    pass


def edgeDetectSeq():
    img = cv2.cvtColor(cv2.imread("medianFiltered.jpg"), cv2.COLOR_BGR2GRAY)
    img = sobelOperator(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    imgSave = Image.fromarray(img)
    imgSave.save('edgeDetectSeq.png')
    print("!")
    # plt.imshow(img)
    # plt.show()


if __name__ == '__main__':
    edgeDetectSeq()
