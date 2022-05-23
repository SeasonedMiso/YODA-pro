import numpy
from PIL import Image
from time import time
import pyopencl


def findedges(plt, dvc, image):
    platform = pyopencl.get_platforms()[plt]
    device = platform.get_devices()[dvc]
    data = numpy.asarray(image).astype(numpy.int32)
    memFlg = pyopencl.mem_flags
    cont = pyopencl.Context([device])
    queue = pyopencl.CommandQueue(cont)

    ima = pyopencl.Buffer(cont, memFlg.READ_ONLY |
                          memFlg.COPY_HOST_PTR, hostbuf=data)
    out = pyopencl.Buffer(cont, memFlg.WRITE_ONLY, data.nbytes)
    prog = pyopencl.Program(cont, getKernel('edgeKern.c') %
                            (data.shape[1], data.shape[0])).build()
    prog.detectedge(queue, data.shape, None, ima, out)
    result = numpy.empty_like(data)
    pyopencl.enqueue_copy(queue, result, out)
    result = result.astype(numpy.uint8)
    print(result)
    img = Image.fromarray(result)
    # img.show()
    img.save('edgeDetect.png')


def getKernel(kern):
    kernel = open(kern).read()
    return kernel


def edgeDetectPara():
    image = Image.open('medianFiltered.jpg')
    findedges(0, 0, image)
    print("saved")


if __name__ == '__main__':
    edgeDetectPara()
