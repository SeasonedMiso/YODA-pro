__kernel void detectedge(__global int *im,__global int *out){
      int a = get_global_id(1);
      int b = get_global_id(0);
      int width = %d;
      int rown = %d;
      int value;


              value = -im[(b)*width + a] -  0* im[(b)*width + a+1] + im[(b)*width + a+2]
                      -2*im[(b+1)*width + a] +  0*im[(b+1)*width + a+1] + 2*im[(b+1)*width + a+2]
                      -im[(b+2)*width + a] -  0*im[(b+2)*width + a+1] + im[(b+2)*width + a+2];

              value = (value < 0   ? 0   : value);
              value = (value > 255 ? 255 : value);
              out[b*width + a] = value;

  }