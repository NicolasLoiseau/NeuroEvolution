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


def gpu_mult(self, a_list, b_list):

    n, p = a_list[0].shape
    l = b_list[0].shape[1]

    a_cpu = np.concatenate(a_list, axis=0)
    b_cpu = np.concatenate(b_list, axis=0)

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


class NeuralNetwork(object):

    def __init__(self, row_nb, column_nb, syn0=None, syn1=None, syn2=None):
        self.column_nb = column_nb
        self.row_nb = row_nb
        self.input_nb = self.column_nb * self.row_nb
        self.output_nb = self.column_nb * self.row_nb + 4
        self.syn0 = syn0 or 2 * np.random.random((self.input_nb, self.output_nb)) - 1
        self.syn1 = syn1 or 2 * np.random.random((self.output_nb, self.output_nb)) - 1
        self.syn2 = syn2 or 2 * np.random.random((self.output_nb, self.output_nb)) - 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, input):
        l1 = self.sigmoid(np.dot(input, self.syn0))
        l2 = self.sigmoid(np.dot(l1, self.syn1))
        return self.sigmoid(np.dot(l2, self.syn2))

    def mutate(self):
        self.syn0 += (2 * np.random.random((self.input_nb, self.output_nb)) - 1) / 10
        self.syn1 += (2 * np.random.random((self.output_nb, self.output_nb)) - 1) / 10
        self.syn2 += (2 * np.random.random((self.output_nb, self.output_nb)) - 1) / 10

    def export(self):
        return {
            'column_nb': self.column_nb,
            'row_nb': self.row_nb,
            'syn0': self.syn0.tolist(),
            'syn1': self.syn1.tolist(),
            'syn2': self.syn2.tolist()
        }
