import logging
import cv2
import numpy as npy
from functools import wraps
from scipy.spatial.transform import Rotation as R
from math import isclose
import copy

import sphere_snap.utils as snap_utils
import sphere_snap.sphere_coor_projections as sphere_proj
from sphere_snap.snap_config import SnapConfig, ImageProjectionType
from sphere_snap import cupy_available, to_cp, to_np
from sphere_snap import custom_cupy_wrap, convert_to_cupy, convert_to_numpy
from sphere_snap.radial_distortion import RadialDistorter

class SphereSnap:
    """
    Reproject various image types by projecting them first on a sphere surface
    """

    def __init__(self, snap_config):
        """
        Given the input parameters, creates an SphereSnap object
        """
        self.snap_config = copy.deepcopy(snap_config) 
        self.snap_config.to_numpy()
        self.sphere_polygon_xyz = self.__create_snap_sphere_polygon()
        self.full_img_sphere_polygon_xyz = self.__create_source_img_sphere_polygon()


    def get_xyz_from_pixel(self, uv):
        """
        Returns the points on the sphere in xyz that are mapped to these uv (image) coordinates
        or None if neither point is in the view's bounds
        """
        uv = npy.array(uv).reshape(-1, 2).astype(int)
        in_bounds_indices = snap_utils.check_uv_in_bounds(uv, self.snap_config.out_hw)
        if in_bounds_indices.any():
            uv = uv[in_bounds_indices]
            xyz = npy.ones((uv.shape[0], 3), npy.float64)
            h, w = self.snap_config.out_hw
            fov_halftan = npy.tan(npy.radians(npy.array(self.snap_config.out_fov_deg) / 2))
            xyz[:, :2] = (((2*(uv+1) )/ npy.array([w, h])) -1) * fov_halftan
            # make all xyz position vectors unit norm
            xyz = self.__normalize_xyz(xyz)
            xyz = xyz.dot(self.snap_config.transform.as_matrix())  
            return xyz

    def get_spherical_from_pixel(self, uv):
        """
        Returns the points on the sphere (in spherical coordinates) that are mapped to these uv (image) coordinates
        or None if neither point is in the view's bounds
        """
        uv = npy.array(uv).reshape(-1, 2).astype(int)
        in_bounds_indices = snap_utils.check_uv_in_bounds(uv, self.snap_config.out_hw)
        if in_bounds_indices.any():
            uv = uv[in_bounds_indices]
            c_phi_theta = self.snap_config.center_polar
            # take only the rotation around the z axis
            view_roll = self.snap_config.transform.as_euler('yxz', degrees=False)[2]
            spherical = sphere_proj.pinhole_coor2spherical(uv,
                                                        self.snap_config.out_fov_deg,
                                                        self.snap_config.out_hw,
                                                        c_phi_theta, view_roll)
            return spherical

    def get_coor_from_pixel(self, uv):
        """
        Returns the coordinates in the source image that are mapped to this snap image pixel coordinates
        """
        spherical = self.get_spherical_from_pixel(uv)
        if spherical is None:
            return None

        coors = SphereSnap.spherical2image( spherical, self.snap_config)
        return coors

    def get_pixel_from_spherical(self, spherical, return_indices=False, np=npy):
        """
        Given the spherical coords (phi, theta) in radians, returns the corresponding uv points
        """
        # project sphere on image plane
        c_phi_theta = self.snap_config.center_polar
        view_roll = self.snap_config.transform.as_euler('yxz', degrees=False)[2]
        uv = sphere_proj.pinhole_spherical2coor._original(spherical,
                                                            to_cp(self.snap_config.out_fov_deg) if cupy_available() else self.snap_config.out_fov_deg,
                                                            self.snap_config.out_hw,
                                                            c_phi_theta, view_roll,np=np)

        # convert from spherical to cartesian
        xyz = sphere_proj.spherical2xyz._original(spherical, np=np)
        # check if points are facing the plane and they are in image bounds
        view_normal = self.snap_config.center_xyz
        if type(view_normal) == npy.ndarray and type(xyz) != npy.ndarray:
            view_normal=to_cp(view_normal)

        facing = xyz.dot(view_normal )
        # TODO: why nan?
        facing[np.isnan(facing)] = -1
        exclude_indices = (facing <= 0) | (uv[:, 0] < 0) | (uv[:, 1] < 0) | (uv[:, 0] >= self.snap_config.out_hw[1]) | (uv[:, 1] >= self.snap_config.out_hw[0])

        if return_indices is True:
            return uv, exclude_indices
        elif exclude_indices.any():
            uv[exclude_indices] = np.array([-1, -1])
        return uv


    def get_pixel_from_xyz(self, xyz, return_indices=False):
        """
        Given some xyz points on the normalised sphere surface
        returns the coordinates in the current image if exists and [-1,-1] if is outside
        optionally can return the outside indices
        """

        # make sure we get a list of xyz points, even if that means only 1
        xyz = xyz.reshape(-1, 3)

        # get snap direction and spherical coordinate of the image center
        c_phi_theta = self.snap_config.center_polar
        view_normal = self.snap_config.center_xyz

        # get spherical coordinates for each point
        spherical = sphere_proj.xyz2spherical(xyz)

        # project spherical on image plane
        view_roll = self.snap_config.transform.as_euler('yxz', degrees=False)[2]
        uv = sphere_proj.pinhole_coor2spherical(spherical,
                                                self.snap_config.out_fov_deg,
                                                self.snap_config.out_hw,
                                                c_phi_theta, view_roll)

        # check if points are facing the plane and they are in image bounds
        facing = xyz.dot(view_normal)
        # TODO: why nan?
        facing[npy.isnan(facing)] = -1

        exclude_indices = (facing <= 0) | (uv[:, 0] < 0) | (uv[:, 1] < 0) | (uv[:, 0] >= self.snap_config.out_hw[1]) | (
            uv[:, 1] >= self.snap_config.out_hw[0])
        if return_indices is True:
            return uv, exclude_indices
        elif exclude_indices.any():
            uv[exclude_indices] = npy.array([-1, -1])
        return uv
    

    def rotate_snap(self, new_rotation_matrix):
        """
        Keep all the same snap configuration bot different orientation
        """
        current_rotation = self.snap_config.transform.as_matrix()
        new_rotation = current_rotation.dot(new_rotation_matrix)
        self.snap_config.set_transform(R.from_matrix(new_rotation))
        self.sphere_polygon_xyz = self.__create_snap_sphere_polygon()

    def get_source_image_coors(self):
        """
        Returns an array of the snap out_hw size that has the image source coordinates for each pixel
        """
        # if cupy enabled we make the conversions explicit and call the _pipe function

        # get uv coordinates for each pixel in the view
        pinhole_coor = sphere_proj.get_2D_coor_grid._pipe(self.snap_config.out_hw)

        # get spherical coordinates for each pixel
        c_phi_theta = to_cp(self.snap_config.center_polar)
        view_roll = self.snap_config.transform.as_euler('yxz', degrees=False)[2]

        # to avoid copying back and forth from gpu to cpu between function calls
        spherical = sphere_proj.pinhole_coor2spherical._pipe(pinhole_coor,
                                                         to_cp(npy.array(self.snap_config.out_fov_deg)),
                                                         self.snap_config.out_hw,
                                                         c_phi_theta, view_roll)
        
        s = self.snap_config
        cp_snap = SnapConfig(s.transform.as_quat(), s.out_hw, to_np(s.out_fov_deg),
                            to_cp(s.source_img_hw),
                            source_img_fov_deg=to_cp(s.source_img_fov_deg),
                            source_dist_coeff= to_cp(s.source_dist_coeff),
                            source_img_type = s.source_img_type
                            )
        src_coor = SphereSnap.spherical2image._pipe(spherical,cp_snap)
        src_coor = to_np(src_coor)
        coors = src_coor.reshape((*self.snap_config.out_hw, 2))
        return coors


    def snap_to_perspective(self, source_img):
        # sanity checks
        if self.snap_config.source_img_type == ImageProjectionType.EQUI:
            h, w = self.snap_config.source_img_hw
            ar = w/h
            assert isclose(ar, 2, abs_tol=1e-2), f"Aspect ratio for equirectangular image is not 2:1. Resolution: [{w, h}]"
        elif self.snap_config.source_img_type == ImageProjectionType.FISHEYE_180 or self.snap_config.source_img_type == ImageProjectionType.HALF_EQUI:
            h, w = self.snap_config.source_img_hw
            ar = w/h
            assert isclose(ar, 1, abs_tol=1e-2), f"Aspect ratio for fisheye180 image is not 1:1. Resolution: [{w, h}]"


        # get coordinates from source image
        src_coor = self.get_source_image_coors()

        # resize image if description does not match reality
        if (source_img.shape[:2] != self.snap_config.source_img_hw[:2]).any():
            logging.warning("Source image dim " + str(source_img.shape) + " was different than expected: " +
                            str(self.snap_config.source_img_hw))
            source_img = cv2.resize(source_img, tuple(self.snap_config.source_img_hw[::-1]))

        # sample coordinates and create image
        warp_mode = True if (self.snap_config.source_img_type==ImageProjectionType.EQUI or self.snap_config.source_img_type==ImageProjectionType.HALF_EQUI) else False
        pers_img = npy.stack([snap_utils.sample_from_img(source_img[..., i], src_coor, clamp=warp_mode)
                                for i in range(source_img.shape[2])], axis=-1)
        return pers_img.astype(npy.uint8)

    @staticmethod
    def __undistort_coors(u_coor, hw, dist_coeff):
        """
        Extracts the raw coordinates from a list of coordinates in the undistorted image
        """
        distortion_map = RadialDistorter.get().coor_mapping(hw, dist_coeff)
        u_coor = u_coor.astype(int)
        # clip values
        in_bounds_indices = snap_utils.check_uv_in_bounds(u_coor, distortion_map.shape[:2])
        # extract real coordinates from distortion map
        coor = distortion_map[u_coor[in_bounds_indices, 1], u_coor[in_bounds_indices, 0]]
        # swap from vu, to uv
        coor[..., [0, 1]] = coor[..., [1, 0]]
        return coor, in_bounds_indices

    @custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
    def image2spherical(coors, snap_config, np=npy):
        """
        Transforms from image coordinates to sphere coordinates
        return the image sphere
        """
        s_hw = to_cp(snap_config.source_img_hw)

        if snap_config.source_img_type == ImageProjectionType.EQUI:
            return sphere_proj.equi_coor2spherical._original(coors, s_hw, np=np)
        elif snap_config.source_img_type == ImageProjectionType.HALF_EQUI:
            return sphere_proj.halfequi_coor2spherical._original(coors, s_hw, np=np)
        elif snap_config.source_img_type == ImageProjectionType.FISHEYE_180:
            return sphere_proj.fisheye180_coor2spherical._original(coors, s_hw, np=np)
        elif snap_config.source_img_type == ImageProjectionType.RADIAL_DISTORTED:
            assert snap_config.source_dist_coeff is not None, "Missing distortion coefficients !"
            dist_coeff = snap_config.source_dist_coeff
            # change from xy to yx
            coors = coors[...,[1,0]]
            # get undistorted coordinates, this is needed to correctly obtain spherical coordinates
            coors = sphere_proj.undistort(coors, np.array(dist_coeff), np.array(s_hw), normalize_values=True, np=np)
            coors = coors * np.array(s_hw)
            coors[...,[0,1]] = coors[...,[1,0]]
        
        return sphere_proj.pinhole_coor2spherical._original(coors,
                                                            np.array(snap_config.source_img_fov_deg),
                                                            np.array(s_hw),
                                                            np.array((0,0)), 0, np=np)

    @staticmethod
    @custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
    def spherical2image(spherical, snap_config, np=npy) :
        """
        Transforms from sphere coordinates to image coordinates
        return the image coordinates
        :param spherical: spherical coordinates for the sphere surface points that we want to project
        :param snap_config: a snap_config object containing metadata of the snap
        """
        if snap_config.source_img_type == ImageProjectionType.PINHOLE:
            coor = sphere_proj.pinhole_spherical2coor._original(spherical,
                                                                snap_config.source_img_fov_deg,
                                                                snap_config.source_img_hw, 
                                                                (0,0), 0, np=np)
            return coor
        elif snap_config.source_img_type == ImageProjectionType.RADIAL_DISTORTED:
            assert snap_config.source_dist_coeff is not None, "Missing distortion coefficients !"
            distortion_map = RadialDistorter.get().coor_mapping(snap_config.source_img_hw, snap_config.source_dist_coeff)
            coor = sphere_proj.pinhole_spherical2coor._original(spherical,
                                                                snap_config.source_img_fov_deg,
                                                                np.array(distortion_map.shape[:2]), 
                                                                (0,0), 0, np=np)
            coor = to_np(coor)
            distorted_coor, in_bounds_indices = SphereSnap.__undistort_coors(coor,
                                                                            to_np(snap_config.source_img_hw),
                                                                            to_np(snap_config.source_dist_coeff))
            coor[in_bounds_indices] = distorted_coor
            return coor

        elif snap_config.source_img_type == ImageProjectionType.FISHEYE_180:
            return sphere_proj.fisheye180_spherical2coor._original(spherical, snap_config.source_img_hw, np=np)
        elif snap_config.source_img_type == ImageProjectionType.EQUI:
            return sphere_proj.equi_spherical2coor._original(spherical, snap_config.source_img_hw, np=np)
        elif snap_config.source_img_type == ImageProjectionType.HALF_EQUI:
            return sphere_proj.halfequi_spherical2coor._original(spherical, snap_config.source_img_hw, np=np)
        else:
            assert False, "Not implemented yet !"


    @staticmethod
    def merge_multiple_snaps(output_hw, sphere_snaps, imgs, merge_method='average', target_type=ImageProjectionType.EQUI):
        """
        Given a list of images extracted from a source image, re-create an image 
        return the full image obtained from the source views
        :param output_hw: a tuple containing source image size(height, width)
        :param views: a list of views we want to merge back in the full image
        :param imgs: a list of imgs coresponding to the views 
        :param merge_method: average, sum, max, last
        :param target_type: image projection type (ImageProjectionType.EQUI, ImageProjectionType.FISHEYE_180..)
        """

        imgs = [to_cp(img) for img in imgs]

        np = npy
        if cupy_available():
            import cupy as cp
            np = cp

        s = sphere_snaps[0].snap_config
        cp_snap = SnapConfig(s.transform.as_quat(), s.out_hw, s.out_fov_deg,
                            to_cp(output_hw),
                            source_img_fov_deg=to_cp(s.source_img_fov_deg),
                            source_dist_coeff= to_cp(s.source_dist_coeff),
                            source_img_type = target_type # set the destination image type
                            )


        #check output type
        out_type = imgs[0].dtype
        nb_channels = imgs[0].shape[2] if len(imgs[0].shape) > 2  else 1
        target = np.zeros((output_hw[0], output_hw[1], nb_channels), np.float32)
        cnt = np.zeros((output_hw[0], output_hw[1], 1), np.uint8)
        
        # construct image uv coordinates
        img_all_coors = sphere_proj.get_2D_coor_grid._pipe(output_hw)

        # exclude region outside the circle
        if cp_snap.source_img_type == ImageProjectionType.FISHEYE_180:
            xy = img_all_coors-(np.array(output_hw)/2)
            r = np.linalg.norm(xy, axis=-1)
            img_all_coors = img_all_coors[r<(output_hw[0]/2)]

        img_all_sphere = SphereSnap.image2spherical._pipe(img_all_coors, cp_snap)
        for sphere_snap, snap_img in zip(sphere_snaps, imgs):
            # extract the coordinates in the view from each pixel in the source image
            view_coors, exclude_indices =  sphere_snap.get_pixel_from_spherical(img_all_sphere, return_indices=True, np=np)

            # filter out all the pixesl that are not inside the view
            view_coors = view_coors[~exclude_indices]
            # extact all source uv coordinates which are inside the view
            dst_coor = img_all_coors[~exclude_indices]

            try:
                # sample all pixel values from the view image
                view_coor_x, view_coor_y = np.split(view_coors.astype(int), 2, axis=-1)
                # make sure no np.nap in view_coor_x or y
                colors = snap_img[view_coor_y, view_coor_x]
            except IndexError:
                # strip nan from view_coors before converting to int
                view_coors_safe = np.nan_to_num(view_coors, copy=True, nan=0.0)

                # sample all pixel values from the view image
                view_coor_x, view_coor_y = np.split(view_coors_safe.astype(int), 2, axis=-1)

                # make sure no np.nap in view_coor_x or y
                colors = snap_img[view_coor_y, view_coor_x]

            colors = colors.reshape(-1,1,nb_channels)

            # write colors in full image destination
            dst_coor_x, dst_coor_y = np.split(dst_coor, 2, axis=-1)
            if merge_method in ['average', 'sum']:
                target[dst_coor_y,dst_coor_x] =  target[dst_coor_y,dst_coor_x] + colors
            elif merge_method == 'max':
                target[dst_coor_y,dst_coor_x] = np.maximum(target[dst_coor_y,dst_coor_x] , colors)
            elif merge_method == 'last':
                target[dst_coor_y,dst_coor_x] = colors
            else:
                assert False, "Unkown merge method!"

            cnt[dst_coor_y,dst_coor_x, 0] += 1

        #average out the accumulated colors
        if merge_method == 'average':
            filled_indices = np.nonzero(cnt[:,:,0])
            target[filled_indices] = target[filled_indices] / cnt[filled_indices]

        return to_np(target.astype(out_type))        
 

    @staticmethod
    def sample_polygon_from_image(points, hw, source_img_fov_deg, source_dist_coeff=None, return_polar=False):
        """Given a polygon on the sphere returns all the 2d points inside it from the source image
        :param points:  polygon xyz points on the sphere
        :param hw: resolution of the source image you want to sample from
        :param source_img_fov_deg: a tuple containing (hfov, vfov) of the perspective source image, if this is None the image is equirectangular
        :param source_dist_coeff: None if does not exist, array containing radial distortion parameters,
                        offset center should be normalised {lambda, center_offset_x, center_offest_y}
        :param return_polar: If true, it will return also the sphere projection of the points in polar coordinates
        """
        if source_dist_coeff is not None:
            distortion_map = RadialDistorter.get().coor_mapping(hw, source_dist_coeff)
            coor, polar = sphere_proj.sample_polygon(points, distortion_map.shape[:2], _)
            coor, in_bounds_indices = SphereSnap.__undistort_coors(coor, hw, source_dist_coeff)
            polar = polar[in_bounds_indices]        
        else:
            coor, polar = snap_utils.sample_polygon(points, hw, source_img_fov_deg)

        if return_polar is True:
            return coor, polar
        return coor
    

    @property
    def focal_length_yx(self):
        # https://codeyarns.com/tech/2015-09-08-how-to-compute-intrinsic-camera-matrix-for-a-camera.html
        fov_vh_rad = npy.deg2rad([self.snap_config.out_fov_deg[1], self.snap_config.out_fov_deg[0]])
        return (npy.array(self.snap_config.out_hw )/ 2) / npy.tan(fov_vh_rad / 2 )

    @property
    def K_matrix(self):
        """
        Returns intrinsics matrix (pinhole camera) based on snap config properties
        """
        focal = self.focal_length_yx
        center = npy.array(self.snap_config.out_hw) / 2

        K = npy.zeros([3, 3])
        K[0, 0] = focal[1]
        K[1, 1] = focal[0]
        K[0, 2] = center[1]
        K[1, 2] = center[0]
        K[2, 2] = 1.0
        return K

    # For call to repr(). Prints object's information
    def __repr__(self):
        return 'SphereSnap(%s)' % self.snap_config.name

    # For call to str(). Prints readable form
    def __str__(self):
        return 'SphereSnap(%s)' % self.snap_config.name


    def copy(self):
        """
        Returns a copy of this SphereSnap object
        """
        return SphereSnap(self.snap_config)

    def __create_source_img_sphere_polygon(self):
        """
        Creates the sphere polygon representing the source image projection on sphere
        """
        if self.snap_config.source_img_fov_deg is None:
            return None

        uv = npy.array([ [0, 0],
                [1, 0],
                [1, 1],
                [0, 1]
                ])
                            
        xyz = npy.ones((uv.shape[0], 3), npy.float64)
        fov_halftan = npy.tan(npy.radians(npy.array(self.snap_config.source_img_fov_deg) / 2))
        xyz[:, :2] = ((2*uv)-1) * fov_halftan
        xyz = self.__normalize_xyz(xyz)
        return xyz


    def __normalize_xyz(self, xyz):
        """
        Normalizes the xyz vectors
        """
        prev_shape = xyz.shape
        # keepdims=True for broadcasting when dividing below by the norm
        norm = npy.linalg.norm(xyz.reshape(-1, 3), axis=-1, keepdims=True)
        xyz /= norm
        return xyz.reshape(prev_shape)
    
    def __create_snap_sphere_polygon(self):
        boundery_points = npy.array([[0, 0],
                                    [self.snap_config.out_hw[1] - 1, 0],
                                    [self.snap_config.out_hw[1] - 1, self.snap_config.out_hw[0] - 1],
                                    [0, self.snap_config.out_hw[0] - 1],
                                    [0, 0]])

        input_xyz  = SphereSnap.get_xyz_from_pixel(self, boundery_points)
        return input_xyz
