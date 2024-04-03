""" Current are the "best" blob detection functions of every type
"""

from .detection_impl import lapgau, diffgau, hessian

cur_dog = lapgau
cur_dog = diffgau
cur_doh = hessian


