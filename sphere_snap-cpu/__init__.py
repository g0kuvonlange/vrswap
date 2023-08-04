"""
sphere_snap.

A quick and easy to use library for reprojecting various image types.
"""
from __future__ import absolute_import
import logging
import numpy as np


__version__ = "1.0.2"
__author__ = 'Andrei Georgescu'
__credits__ = ''


is_cupy_available = None

def __init_cupy():
    try:
        # Try to import cupy
        import cupy as cp
        import cupyx.scipy.linalg as cpxl
        # Try to access a device
        cp.cuda.Device(0).compute_capability
        # Flag indicating successful import
        logging.info("CuPy library is available!")
        return True
    except ImportError:
        logging.info("CuPy library is not available, rolling back on Numpy")
    except RuntimeError:
        logging.info("CuPy available but access to device 0 is not possible -> Rolling back on Numpy")
    return False



def cupy_available():
    """
    Return ``True`` if CuPy is installed and a GPU device is available,
    otherwise return ``False``.
    """
    return is_cupy_available


def to_cp(arr):
    """
    If CuPy available, convert numpy array to cupy array
    """
    if is_cupy_available and type(arr) == np.ndarray:
        arr = cp.asarray(arr)
    return arr


def to_np(arr):
    # return numpy array on host device (CPU)
    if is_cupy_available:
        cupy_v = int(cp.__version__.split('.')[0])
        # since release 9.0.0, core was renamed to _core
        cp_arr_type = cp.core.core.ndarray if cupy_v < 9 else cp.ndarray
        if type(arr) == cp_arr_type:
            arr = arr.get()
    return arr


MIN_ELEMENTS_FOR_CUPY = 500000


# decorator to change the numpy package to cupy
def cupy_wrapper(func):
    def wrap(*args, **kwargs):
        if is_cupy_available:
            kwargs["np"] = cp 
        return func(*args, **kwargs)
    return wrap


def custom_cupy_wrap(pre, post):
    def decorate(func):
        def call(*args, **kwargs):
            #only use cupy for big arrays
            if is_cupy_available:
                size_list = [arg.size for arg in args if type(arg) == np.ndarray]
                if len(size_list) > 0 and np.max(np.array(size_list)) > MIN_ELEMENTS_FOR_CUPY:
                    args = pre(func, *args)
                    kwargs["np"] = cp
                    result = func(*args, **kwargs)
                    if result is tuple:
                        result = post(func, *result)
                        return result
                    else:
                        result = post(func, (result))
                        return result[0]
            return func(*args, **kwargs)
        call._original = func
        call._pipe = cupy_wrapper(func)
        return call
    return decorate


def convert_to_cupy(func, *args):
    return [to_cp(arg) for arg in args]


def convert_to_numpy(func, *results):
    return tuple(to_np(res) for res in results)

def init_package():
    global is_cupy_available    
    
    is_cupy_available = __init_cupy()

    if is_cupy_available:
        logging.info("SphereSnap will use CuPy for image reprojections!")
    else:
        logging.info("SphereSnap will use Numpy for image reprojections!")

