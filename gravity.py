import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
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


if __name__ == '__main__':
    print('gravity test run')
    # Input definition
    a = np.zeros((6, 3)).astype(int)
    b = a.copy()
    b[0, :] = 1
    c = np.ones((6, 3)).astype(int)
    c[1, :] = 1
    d = a.copy()
    d[0:4, :] = 1
    e = a.copy()
    e[1, [0, 2]] = 1
    e[2, 1] = 1
    e[0, 1] = 1
    f = np.asarray((a, b, c, d, e))
    print("-" * 80)
    print(f)
    print("-" * 80)
    # Apply function
    f_test = gravity_gpu(f)
    print('Result of distributed gravity')
    print(f_test)