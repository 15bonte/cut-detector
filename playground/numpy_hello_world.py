"""Simple module to try numpy functions in it.

"""

import numpy as np

def hello_numpy():
    a = np.arange(15).reshape(3, 5)
    print("a\n", a)
    print(a.shape, a.ndim)



if __name__ == "__main__":
    hello_numpy()