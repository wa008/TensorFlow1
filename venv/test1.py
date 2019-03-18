import tensorflow as tf
import numpy as np
import tkinter
x = np.ones((8, 1))

x = x.reshape(2,4)
print(x)
x = x.reshape(-1, 1)
print(x)
