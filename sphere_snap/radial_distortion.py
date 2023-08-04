import logging
import numpy as np
import cv2
import threading
import warnings

from threading import Lock
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.utils.exceptions import AstropyWarning

import sphere_snap.utils as snap_utils
import sphere_snap.sphere_coor_projections as sphere_proj
from . import custom_cupy_wrap, convert_to_cupy, convert_to_numpy, to_np, to_cp


warnings.simplefilter('ignore', category=AstropyWarning)

import numpy as npy



class RadialDistorter:
    __mutex = Lock()
    __scale_factor = 2
    __thread_instances_dict = dict()

    @staticmethod 
    def get():
        """ Static access method"""
        thread_id = threading.get_ident()
        if thread_id not in RadialDistorter.__thread_instances_dict:
            RadialDistorter()
        return RadialDistorter.__thread_instances_dict[thread_id]
    
    @staticmethod
    def __map_id(res_hw, dist_coeff):
        return f"{int(res_hw[0])}_{int(res_hw[1])}_{dist_coeff[0]:0.8f}_{dist_coeff[1]:0.4f}_{dist_coeff[2]:0.4f}"
    
    def __init__(self):
        thread_id = threading.get_ident()
        if  thread_id not in RadialDistorter.__thread_instances_dict:
            RadialDistorter.__thread_instances_dict[thread_id] = self
            self.__dist_mapping_dict = dict()
        else:
            raise Exception("This class is a Singleton!")

    @staticmethod
    @custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
    def __construct_map(distorted_coor, undistorted_coor, res_hw, np=npy):
        # creates a map between distorted and undistorted coordinates
        # this map might contain NaN values where there are no distortion calculated
        
        new_hw = (res_hw*undistorted_coor.max(axis=0)).astype(int)
        distorted_uv = np.array(undistorted_coor*(res_hw-1)).astype(int)

        # we use unique for selecting a coordinate omly once
        # use numpy method as unique is not fully suported in cupy 
        distorted_uv = to_np(distorted_uv)
        distorted_uv, unique_indices = npy.unique(distorted_uv, return_index=True, axis=0)
        dv, du = npy.split(distorted_uv, 2, axis=-1)

        map_undist2coef = np.ones((int(new_hw[0]), int(new_hw[1]), 2)).astype(np.float32) * np.nan
        map_undist2coef[dv, du] = distorted_coor[unique_indices].reshape(-1, 1, 2)/res_hw
        return map_undist2coef, new_hw

    def coor_mapping(self, res_hw, dist_coeff, ignore_cache=False, same_resolution = True):
        dist_coeff = np.array(dist_coeff)
        map_id = RadialDistorter.__map_id(res_hw, dist_coeff)
        RadialDistorter.__mutex.acquire()
        if ignore_cache is True or map_id not in self.__dist_mapping_dict:
            logging.debug(f"{map_id} was constructed")
            map_hw = (np.array(res_hw)/RadialDistorter.__scale_factor).astype(int)
            #use _pipe so that all are done with cupy if available and only convert it tu cpu at the end
            distorted_coor = sphere_proj.get_2D_coor_grid._pipe(map_hw)
            distorted_coor[...,[0,1]] = distorted_coor[...,[1,0]]
            undistorted_coor = sphere_proj.undistort._pipe(distorted_coor,
                     to_cp(dist_coeff),
                     to_cp(np.array(map_hw)),
                     normalize_values = same_resolution
                    )

            mapping, new_map_hw = RadialDistorter.__construct_map._pipe(distorted_coor, undistorted_coor, to_cp(np.array(map_hw))) 
            mapping = to_np(mapping)
            new_map_hw = to_np(new_map_hw)

            # we need to do this in in order to ensure there are no NaN values in the map
            kernel = Gaussian2DKernel(x_stddev=1)
            mapping[..., 0] = interpolate_replace_nans(mapping[..., 0], kernel)
            mapping[..., 1] = interpolate_replace_nans(mapping[..., 1], kernel)
            mapping[np.isnan(mapping[..., 0]), 0] = -1
            mapping[np.isnan(mapping[..., 1]), 1] = -1
            mapping[..., 0] = snap_utils.interpolate_nans(mapping[..., 0])
            mapping[..., 1] = snap_utils.interpolate_nans(mapping[..., 1].T).T

            new_map_hw = (np.array(new_map_hw) * RadialDistorter.__scale_factor).astype(int)
            mapping = cv2.resize(mapping, (new_map_hw[1], new_map_hw[0]), cv2.INTER_AREA)
        
            self.__dist_mapping_dict[map_id] = (mapping * res_hw).astype(np.float32) 
            
        RadialDistorter.__mutex.release()
        return self.__dist_mapping_dict[map_id] 

