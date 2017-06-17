import numpy as np
import tensorflow as tf
import gensim as gs
from sumfunc import sumthese

a = np.array([1, 2, 3])
b = a * 2
print(b)

print(sumthese(a,b))