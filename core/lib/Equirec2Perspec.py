import math
import cv2
import numpy as np


class Equirectangular:
    def __init__(self, img):
        self._img = img
        [self._height, self._width, _] = self._img.shape

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = self._height
        equ_w = self._width 
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        w_len = np.tan(np.radians(wFOV / 2.0))
        h_len = np.tan(np.radians(hFOV / 2.0))

        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len, width), [height, 1])
        z_map = -np.tile(np.linspace(-h_len, h_len, height), [width, 1]).T

        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.stack((x_map, y_map, z_map), axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2])
        lon = np.arctan2(xyz[:, 1], xyz[:, 0])

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180

        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 180 * equ_cy + equ_cy

        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return persp

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

class EquirectangularFixed:
    def __init__(self, img):
        self._img = img
        self._height, self._width, _ = img.shape

    def _hequirect_to_xyz(self, i, j):
        """Vectorized conversion from half equirectangular to 3D Cartesian coordinates."""
        phi = (i / self._width) * math.pi / 2
        theta = (j / self._height) * math.pi / 2

        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta)
        z = np.cos(theta) * np.cos(phi)

        return x, y, z

    def _xyz_to_flat(self, x, y, z, width, height, h_fov=360, v_fov=180):
        """Vectorized conversion from 3D Cartesian coordinates to a flat image projection."""
        theta = np.arccos(z)
        uf = width * (0.5 + np.arctan2(x, z) / (2 * np.pi))
        vf = height * (0.5 + np.arcsin(y) / np.pi)

        visible = np.logical_and.reduce((uf >= 0, uf < self._width, vf >= 0, vf < self._height, z >= 0))
        return uf, vf, visible

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        perspective_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Generate a grid of coordinates
        i, j = np.meshgrid(np.arange(width) - width / 2, np.arange(height) - height / 2)
        
        # Convert perspective coordinates to equirectangular coordinates
        x, y, z = self._hequirect_to_xyz(i, j)
        
        # Adjust for the angles THETA (around y-axis) and PHI (around x-axis)
        x_rot = x * math.cos(math.radians(THETA)) - z * math.sin(math.radians(THETA))
        z_temp = x * math.sin(math.radians(THETA)) + z * math.cos(math.radians(THETA))
        y_rot = y * math.cos(math.radians(-PHI)) - z_temp * math.sin(math.radians(-PHI))
        z_rot = y * math.sin(math.radians(-PHI)) + z_temp * math.cos(math.radians(-PHI))

        uf, vf, visible = self._xyz_to_flat(x_rot, y_rot, z_rot, self._width, self._height, FOV, FOV * (height / width))

        # Use vectorized indexing to update the perspective image
        perspective_img[visible] = self._img[vf[visible].astype(int), uf[visible].astype(int)]

        return perspective_img

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

