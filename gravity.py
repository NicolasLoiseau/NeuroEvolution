import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

gravity_kernel_template = """
       __global__ void gravity(float *a)
       {  
           int idb = blockIdx.x * %(l)s * blockDim.y;

           int ind = 0;

           while (ind < %(l)s)
           { 
               if (a[idb + ((%(l)s - 1) - (ind))* %(c)s + threadIdx.y] == 0)
               {
                   ind++;
               }

               else
               {
                   if (ind == 0)
                   {
                       ind++;
                   }

                   else
                   {  
                      if (a[idb + ((%(l)s - 1) - (ind)+1)* %(c)s + threadIdx.y] != 0)
                      {
                           ind++;
                      }
                      else
                      {
                          int temp = a[idb + ((%(l)s - 1) - (ind))* %(c)s + threadIdx.y];

                          a[idb + ((%(l)s - 1) - (ind)+1)* %(c)s + threadIdx.y] = temp;

                          a[idb + ((%(l)s - 1) - (ind))* %(c)s + threadIdx.y] = 0;

                          ind = 0;
                      }
                   }
               }
           }   
       }
       """


def gravity_gpu(f):
    # Constants
    g = f.shape[0]
    l = f.shape[1]
    c = f.shape[2]

    # Define type
    f = f.astype(np.float32)

    # Create memory space on device
    f_gpu = cuda.mem_alloc(f.nbytes)

    # copy items on device
    cuda.memcpy_htod(f_gpu, f)

    # Define cuda kernel
    gravity_kernel = gravity_kernel_template % {'l': l, 'c': c}
    mod = SourceModule(gravity_kernel)

    # Define function form kernel (on gpu)
    func = mod.get_function("gravity")

    # Apply function (on gpu)
    func(f_gpu, grid=(g, 1), block=(1, c, 1))

    # Empty array to store result
    f_new = np.empty_like(f)

    # copy results back from device to host
    cuda.memcpy_dtoh(f_new, f_gpu)

    return f_new
