
# cuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda import characterize

# numpy
import numpy as np

# under the line
from individual import Kernel
from neuralNetwork import NeuralNetwork
from gravity import gravity_gpu
from remodelling import remodelling_gpu

# Cuda kernel for gravity function
gravity_kernel_template = """
       __global__ void gravity(float *a)
       {  
           int idb = blockIdx.x * blockDim.x * blockDim.y;

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
    func(f_gpu, grid=(g, 1), block=(l, c, 1))

    # Empty array to store result
    f_new = np.empty_like(f)

    # copy results back from device to host
    cuda.memcpy_dtoh(f_new, f_gpu)

    return f_new

class Generation:
    def __init__(self, generation_size, row_nb, column_nb, cap):
        self.generation_size = generation_size
        self.row_nb = row_nb
        self.column_nb = column_nb
        self.cap0 = cap
        self.skeletons = np.zeros((self.generation_size,self.row_nb, self.column_nb)).astype(int)
        self.scores = np.zeros(self.generation_size)
        self.caps = np.ones(self.generation_size)*self.cap0
        self.move_mapper = self.build_move_mapper()
        self.intelligence = NeuralNetwork(self.generation_size, self.row_nb, self.column_nb)

    def reset(self):
        self.skeletons = np.zeros((self.generation_size,self.row_nb, self.column_nb)).astype(int)
        self.scores = np.zeros(self.generation_size)
        self.caps = np.ones(self.generation_size)*self.cap0

    def fill(self):
        height = self.row_nb // 2
        sub_skeleton = np.random.randint(self.cap0, size=(self.generation_size, height, self.column_nb)) + 1
        self.skeletons[:, self.row_nb - height:, :] = sub_skeleton

    @property
    def game_over(self):
        return self.skeletons.sum(axis=2)[:, 0] != 0

    def remodeling(self, start_point, end_point, play_range):
        for i in play_range:
            self.individuals[i].remodelling

    def refill(self, not_equal, play_range):
        for i in play_range:
            if not_equal[i]:
                column = np.random.randint(0, self.column_nb - 1)
                values = [1, 1, 2, 2, self.caps[i], self.caps[i] - 1]
                uniform = np.random.randint(0, 5)
                value = values[uniform]
                self.skeletons[i][0, column] = value

    def gravity(self):
        gravity_gpu(self.skeletons)


    def one_play(self):
        start_pt, end_pt = self.get_move()
        not_equal = [self.skeletons[i][start_pt[i][0]][start_pt[i][1]] != self.skeletons[i][end_pt[i][0]][end_pt[i][1]] for i in range(self.generation_size)]
        play_range = np.where(np.invert(self.game_over))[0]
        play_vec = self.game_over.astype(int)
        self.remodeling(start_pt, end_pt, play_vec)
        self.refill(not_equal, play_range)
        self.gravity()

    def play(self):
        self.reset()
        self.fill()
        while not all(self.game_over):
            self.one_play()
        return self.scores

    def old_get_move(self):
        start_points = np.zeros((self.generation_size, 2))
        end_points = np.zeros((self.generation_size, 2))
        for i in range(self.generation_size):
            start_point, end_point = self.individuals[i].get_move()
            start_points[i] = start_point
            end_points[i] = end_point
        return start_points.astype(int), end_points.astype(int)

    def build_move_mapper(self):
        """Construct the dictionary to map the neural network output with moves."""
        move_mapper = dict()

        # up movement
        for i in range(1, self.row_nb):
            for j in range(self.column_nb):
                move_mapper[(i - 1) * self.column_nb + j] = [(i, j), (i - 1, j)]

        N = (self.row_nb - 1) * self.column_nb
        # right movement
        for i in range(1, self.row_nb):
            for j in range(self.column_nb - 1):
                move_mapper[N + (i - 1) * (self.column_nb - 1) + j] = [(i, j), (i, j + 1)]

        M = N + (self.row_nb - 1) * (self.column_nb - 1)
        # left movement
        for i in range(1, self.row_nb):
            for j in range(1, self.column_nb):
                move_mapper[M + (i - 1) * (self.column_nb - 1) + j - 1] = [(i, j), (i, j - 1)]
        return move_mapper

    def get_move(self):
        """Return the moves choosen by the intelligence for each individual."""
        sorted_index = np.flip(np.apply_along_axis(np.argsort, 2, self.neural()), 2)
        u, v, w = sorted_index.shape
        return np.array([self.best_move(i, index) for i, index in zip(range(u), sorted_index.reshape(u, w))])

    def best_move(self, i, sorted_index):
        """Find the first possible move according to the neural network output."""
        for index in sorted_index:
            pts = self.move_mapper[index]
            if self.fusible(i, pts[0], pts[1]):
                return pts[0], pts[1]
        raise Exception

    def fusible(self, i, start_point, end_point):
        """Check if the cases (i, j) and (k, l) are fusible and (i, j) is not empty."""
        condition1 = self.skeletons[i][start_point]
        condition2 = self.skeletons[i][start_point] + self.skeletons[i][end_point] <= self.caps[i]
        condition3 = self.skeletons[i][start_point] == self.skeletons[i][end_point]
        return condition1 and (condition2 or condition3)

    def neural(self):
        """The neural network output"""
        nn_input = self.skeletons.reshape((self.generation_size, 1, self.row_nb * self.column_nb))
        return self.intelligence(nn_input)

if __name__ == '__main__':
    for i in range(0,1):
        gen = Generation(1, 6, 3, 7)
        #print([i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i])
        gen.fill()
        gravity_gpu(gen.skeletons)



