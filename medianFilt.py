import pyopencl
from imageio import imread, imsave
import numpy

image = imread("grainy.jpg").astype(numpy.float32)
print(image.shape)
image = numpy.mean(image, axis=2)
print(image.shape)

cont = pyopencl.create_some_context()
queue = pyopencl.CommandQueue(cont)
memFlg = pyopencl.mem_flags

src = """
void sort(int *a, int *b, int *c) {
   int swap;
   if(*a > *b) {
      swap = *a;
      *a = *b;
      *b = swap;
   }
   if(*a > *c) {
      swap = *a;
      *a = *c;
      *c = swap;
   }
   if(*b > *c) {
      swap = *b;
      *b = *c;
      *c = swap;
   }
}
__kernel void medianFilter(
    __global float *image, __global float *result, __global int *width, __global
    int *height)
{
    int w = *width;
    int h = *height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w*posy + posx;
    if( posx == 0 || posy == 0 || posx == w-1 || posy == h-1 )
    {
        result[i] = image[i];
    }
    else
    {
        int pixel00, pixel01, pixel02, pixel10, pixel11, pixel12, pixel20,
            pixel21, pixel22;
        pixel00 = image[i - 1 - w];
        pixel01 = image[i- w];
        pixel02 = image[i + 1 - w];
        pixel10 = image[i - 1];
        pixel11 = image[i];
        pixel12 = image[i + 1];
        pixel20 = image[i - 1 + w];
        pixel21 = image[i + w];
        pixel22 = image[i + 1 + w];
        //sort the rows
        sort( &(pixel00), &(pixel01), &(pixel02) );
        sort( &(pixel10), &(pixel11), &(pixel12) );
        sort( &(pixel20), &(pixel21), &(pixel22) );
        //sort the columns
        sort( &(pixel00), &(pixel10), &(pixel20) );
        sort( &(pixel01), &(pixel11), &(pixel21) );
        sort( &(pixel02), &(pixel12), &(pixel22) );
        //sort the diagonal
        sort( &(pixel00), &(pixel11), &(pixel22) );
        // median is the the middle value of the diagonal
        result[i] = pixel11;
    }
}
"""


def medFiltPara():
    prog = pyopencl.Program(cont, src).build()
    img = pyopencl.Buffer(cont, memFlg.READ_ONLY |
                          memFlg.COPY_HOST_PTR, hostbuf=image)
    result = pyopencl.Buffer(cont, memFlg.WRITE_ONLY, image.nbytes)
    width = pyopencl.Buffer(
        cont, memFlg.READ_ONLY | memFlg.COPY_HOST_PTR, hostbuf=numpy.int32(image.shape[1])
    )
    height = pyopencl.Buffer(
        cont, memFlg.READ_ONLY | memFlg.COPY_HOST_PTR, hostbuf=numpy.int32(image.shape[0])
    )
    prog.medianFilter(queue, image.shape, None, img, result, width, height)
    out = numpy.empty_like(image)
    pyopencl.enqueue_copy(queue, out, result)
    imsave("medianFiltered.jpg", out)


if __name__ == '__main__':
    medFiltPara()
