import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit


kernel_code_template = """
__global__ void MatrixMulKernel(float *a, float *b, float *c)
{
    float Pvalue = 0;

    for (int k = 0; k < %(p)s; ++k) {
        float Aelement = a[blockIdx.x * %(n)s * %(p)s + threadIdx.x * %(p)s + k];
        float Belement = b[blockIdx.x * %(p)s * %(l)s + k * %(l)s + threadIdx.y];
        Pvalue += Aelement * Belement;
    }

    c[blockIdx.x * %(n)s * %(l)s + threadIdx.x * %(l)s + threadIdx.y] = Pvalue;
}
"""


class NeuralNetworkGPU(object):

    def __init__(self, generation_size, row_nb, column_nb, skeletons_gpu, syn0=None, syn1=None, syn2=None):
        self.generation_size = generation_size
        self.column_nb = column_nb
        self.row_nb = row_nb
        self.input_nb = self.column_nb * self.row_nb
        self.output_nb = (self.row_nb - 1) * (3 * self.column_nb - 2)
        self.syn0 = syn0 or self.syn_init(self.input_nb, self.output_nb)
        self.syn1 = syn1 or self.syn_init(self.output_nb, self.output_nb)
        self.syn2 = syn2 or self.syn_init(self.output_nb, self.output_nb)
        self.skeletons_gpu = skeletons_gpu
        self.output_gpu = driver.mem_alloc(np.ones((self.generation_size, 1, self.output_nb)).astype(np.float32).nbytes)
        self.syn0_gpu = driver.mem_alloc(np.ones((self.generation_size, self.input_nb, self.output_nb)).astype(np.float32).nbytes)
        self.syn1_gpu = driver.mem_alloc(np.ones((self.generation_size, self.output_nb, self.output_nb)).astype(np.float32).nbytes)
        self.syn2_gpu = driver.mem_alloc(np.ones((self.generation_size, self.output_nb, self.output_nb)).astype(np.float32).nbytes)

    def syn_init(self, x, y):
        return np.array([2 * np.random.random((x, y)) - 1 for _ in range(self.generation_size)]).astype(np.float32)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, input):
        l1 = self.sigmoid(self.gpu_mult(input, self.syn0, self.skeletons_gpu, self.syn0_gpu))
        l2 = self.sigmoid(self.gpu_mult(l1, self.syn1, self.output_gpu, self.syn1_gpu))
        return self.sigmoid(self.gpu_mult(l2, self.syn2, self.output_gpu, self.syn2_gpu))

    def mutate(self, index):
        l = len(index)

        self.syn0[:l] = self.syn0[index]
        self.syn0[l:] = self.syn0[index]
        self.syn0[l:] += np.array([(2 * np.random.random((self.input_nb, self.output_nb)) - 1) / 5 for i in range(l)]).astype(np.float32)

        self.syn1[:l] = self.syn1[index]
        self.syn1[l:] = self.syn1[index]
        self.syn1[l:] += np.array([(2 * np.random.random((self.output_nb, self.output_nb)) - 1) / 5 for i in range(l)]).astype(np.float32)

        self.syn2[:l] = self.syn2[index]
        self.syn2[l:] = self.syn2[index]
        self.syn2[l:] += np.array([(2 * np.random.random((self.output_nb, self.output_nb)) - 1) / 5 for i in range(l)]).astype(np.float32)

    def export(self):
        return {
            'column_nb': self.column_nb,
            'row_nb': self.row_nb,
            'syn0': self.syn0.tolist(),
            'syn1': self.syn1.tolist(),
            'syn2': self.syn2.tolist()
        }

    def gpu_mult(self, a_cpu, b_cpu, a_gpu, b_gpu):

        n = a_cpu.shape[1]
        p = a_cpu.shape[2]
        l = b_cpu.shape[2]

        # transfer host (CPU) memory to device (GPU) memory
        driver.memcpy_htod(a_gpu, a_cpu.astype(np.float32))
        driver.memcpy_htod(b_gpu, b_cpu.astype(np.float32))

        # create empty gpu array for the result (C = A * B)
        c_gpu = gpuarray.empty((self.generation_size, n, l), np.float32)

        kernel_code = kernel_code_template % {
            'n': n,
            'p': p,
            'l': l
        }

        # compile the kernel code
        mod = compiler.SourceModule(kernel_code)

        # get the kernel function from the compiled module
        matrixmul = mod.get_function("MatrixMulKernel")

        # call the kernel on the card
        matrixmul(
            # inputs
            a_gpu, b_gpu,
            # output
            c_gpu,
            block=(n, l, 1),
            grid=(self.generation_size, 1)
        )

        return c_gpu.get()
