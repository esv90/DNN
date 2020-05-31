import numpy as np
from scipy import signal

x1 = np.array([[100, 100, 100, 100, 0, 0, 0, 0],
               [100, 100, 100, 100, 0, 0, 0, 0],
               [100, 100, 100, 100, 0, 0, 0, 0],
               [100, 100, 100, 100, 0, 0, 0, 0],
               [100, 100, 100, 100, 0, 0, 0, 0],
               [100, 100, 100, 100, 0, 0, 0, 0],
               [100, 100, 100, 100, 0, 0, 0, 0],
               [100, 100, 100, 100, 0, 0, 0, 0]])

w1 = np.array([[1, 0, -1],
               [1, 0, -1],
               [1, 0, -1]])


arr_valid = np.array([[  0,   0, 300, 300,   0,   0],
                      [  0,   0, 300, 300,   0,   0],
                      [  0,   0, 300, 300,   0,   0],
                      [  0,   0, 300, 300,   0,   0],
                      [  0,   0, 300, 300,   0,   0],
                      [  0,   0, 300, 300,   0,   0]])

arr_same = np.array([[-200,    0,    0,  200,  200,    0,    0,    0],
                     [-300,    0,    0,  300,  300,    0,    0,    0],
                     [-300,    0,    0,  300,  300,    0,    0,    0],
                     [-300,    0,    0,  300,  300,    0,    0,    0],
                     [-300,    0,    0,  300,  300,    0,    0,    0],
                     [-300,    0,    0,  300,  300,    0,    0,    0],
                     [-300,    0,    0,  300,  300,    0,    0,    0],
                     [-200,    0,    0,  200,  200,    0,    0,    0]])

arr_stride1 = arr_valid # default is with stride length equal to 1

arr_stride2 = np.array([[  0, 300,   0],
                        [  0, 300,   0],
                        [  0, 300,   0]])

np.testing.assert_allclose(signal.convolve2d(x1, w1[::-1, ::-1], 'valid'), arr_valid)
np.testing.assert_allclose(signal.convolve2d(x1, w1[::-1, ::-1], 'same'), arr_same)

stride1 = 1
np.testing.assert_allclose(signal.convolve2d(x1, w1[::-1, ::-1], 'valid')[::stride1, ::stride1], arr_stride1)

stride2 = 2
np.testing.assert_allclose(signal.convolve2d(x1, w1[::-1, ::-1], 'valid')[::stride2, ::stride2], arr_stride2)
