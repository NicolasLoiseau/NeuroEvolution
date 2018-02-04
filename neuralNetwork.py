import numpy as np
from pycuda import driver, compiler, gpuarray, tools

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
_global_ void MatrixMulKernel(float *a, float *b, float *c)
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


class NeuralNetwork(object):

    def __init__(self, generation_size, row_nb, column_nb, syn0=None, syn1=None, syn2=None):
        self.generation_size = generation_size
        self.column_nb = column_nb
        self.row_nb = row_nb
        self.input_nb = self.column_nb * self.row_nb
        self.output_nb = (self.row_nb - 1) * (3 * self.column_nb - 2)
        self.syn0 = syn0 or [2 * np.random.random((self.input_nb, self.output_nb)) - 1 for _ in range(self.generation_size)]
        self.syn1 = syn1 or [2 * np.random.random((self.output_nb, self.output_nb)) - 1 for _ in range(self.generation_size)]
        self.syn2 = syn2 or [2 * np.random.random((self.output_nb, self.output_nb)) - 1 for _ in range(self.generation_size)]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, input):

        input = np.concatenate(input, axis=0)
        syn0 = np.concatenate(self.syn0, axis=0)
        syn1 = np.concatenate(self.syn1, axis=0)
        syn2 = np.concatenate(self.syn2, axis=0)

        l1 = self.sigmoid(self.gpu_mult(input, syn0))
        l2 = self.sigmoid(self.gpu_mult(l1, syn1))
        return self.sigmoid(self.gpu_mult(l2, syn2))

    def mutate(self, index):
        for i in index:
            self.syn0[i] += (2 * np.random.random((self.input_nb, self.output_nb)) - 1) / 5
            self.syn1[i] += (2 * np.random.random((self.output_nb, self.output_nb)) - 1) / 5
            self.syn2[i] += (2 * np.random.random((self.output_nb, self.output_nb)) - 1) / 5

    def export(self):
        return {
            'column_nb': self.column_nb,
            'row_nb': self.row_nb,
            'syn0': self.syn0.tolist(),
            'syn1': self.syn1.tolist(),
            'syn2': self.syn2.tolist()
        }

    def gpu_mult(self, a_cpu, b_cpu):

        n = a_cpu.shape[0] // self.generation_size
        p = a_cpu.shape[1]
        l = b_cpu.shape[1]

        # transfer host (CPU) memory to device (GPU) memory
        a_gpu = gpuarray.to_gpu(a_cpu)
        b_gpu = gpuarray.to_gpu(b_cpu)

        # create empty gpu array for the result (C = A * B)
        c_gpu = gpuarray.empty((n * self.generation_size, l), np.float32)

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
