import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Cuda kernel
remodelling_kernel_template = """
    __global__ void Remodeling(float *a, int *moves, int *scores, int *caps, int *vect_play)
    {
        if (vect_play[threadIdx.x] == 0)
        {
            int temp = 0;
            
            int spt = moves[4 * threadIdx.x] * %(c)s + moves[4 * threadIdx.x + 1];
            
            int ept = moves[4 * threadIdx.x + 2] * %(c)s + moves[4 * threadIdx.x + 3];

            int ids = threadIdx.x * %(l)s * %(c)s +   spt;

            int ide = threadIdx.x * %(l)s * %(c)s +  ept;

            temp = a[ids];

            a[ids] = 0;

            if(a[ide] != temp)
            {
                a[ide] += temp;

                scores[threadIdx.x] += a[ide];
            }
            else
            {
                scores[threadIdx.x] += 2 * temp;
            }
            caps[threadIdx.x] = %(d)s + scores[threadIdx.x]/100;
        }
    }
    """


def remodeling_gpu(a, moves, scores, caps, play_vect, cap0=5):
    """move (i,j) to (k,l) for every individual using gpu"""
    # Constants
    g = a.shape[0]
    l = a.shape[1]
    c = a.shape[2]

    # Define type for gpu computation
    a = a.astype(np.float32)
    scores = scores.astype(np.int32)
    moves = moves.astype(np.int32)
    play_vect = play_vect.astype(np.int32)
    caps = caps.astype(np.int32)

    # Create memory space on device
    a_gpu = cuda.mem_alloc(a.nbytes)
    scores_gpu = cuda.mem_alloc(scores.nbytes)
    moves_gpu = cuda.mem_alloc(moves.nbytes)
    play_vect_gpu = cuda.mem_alloc(play_vect.nbytes)
    caps_gpu = cuda.mem_alloc(caps.nbytes)

    # Copy items on device
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(scores_gpu, scores)
    cuda.memcpy_htod(moves_gpu, moves)
    cuda.memcpy_htod(play_vect_gpu, play_vect)
    cuda.memcpy_htod(caps_gpu, caps)

    # Empty arrays to store results
    a_new = np.empty_like(a)
    scores_new = np.empty_like(scores)
    caps_new = np.empty_like(caps)

    # Define remodelling kernel
    remodelling_kernel = remodelling_kernel_template % {'l': l, 'c': c, 'd': cap0}
    mod = SourceModule(remodelling_kernel)

    # Define function form kernel (on GPU)
    func = mod.get_function("Remodeling")

    # Apply function (on GPU)
    func(a_gpu, moves_gpu, scores_gpu, caps_gpu, play_vect_gpu, grid=(1, 1), block=(g, 1, 1))

    # Copy results back from device
    cuda.memcpy_dtoh(a_new, a_gpu)
    cuda.memcpy_dtoh(scores_new, scores_gpu)
    cuda.memcpy_dtoh(caps_new, caps_gpu)

    return a_new, scores_new, caps_new
