import cv2
import cupy as cp
import cupyx.scipy.ndimage as ndi
from functools import lru_cache

class Perspective:
    def __init__(self, img_name, FOV, THETA, PHI):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        self._img = cp.asarray(self._img)
        self._width, self._height, _ = self._img.shape

        self._init_params(FOV, THETA, PHI)

    def _init_params(self, FOV, THETA, PHI):
        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI
        self.hFOV = float(self._height) / self._width * FOV
        self.w_len = cp.tan(cp.radians(self.wFOV / 2.0))
        self.h_len = cp.tan(cp.radians(self.hFOV / 2.0))

        self.R1, self.R2 = self._calc_rotation_matrices()

    @lru_cache(maxsize=None)
    def _calc_rotation_matrices(self):
        y_axis = cp.array([0.0, 1.0, 0.0], cp.float32)
        z_axis = cp.array([0.0, 0.0, 1.0], cp.float32)

        [R1, _] = cv2.Rodrigues(cp.asnumpy(z_axis * cp.radians(self.THETA)))
        [R2, _] = cv2.Rodrigues(cp.asnumpy(cp.dot(cp.asarray(R1), y_axis) * cp.radians(-self.PHI)))

        R1 = cp.asarray(cp.linalg.inv(cp.asarray(R1)))
        R2 = cp.asarray(cp.linalg.inv(cp.asarray(R2)))

        return R1, R2

    def SetParameters(self, FOV, THETA, PHI):
        self._init_params(FOV, THETA, PHI)

    def GetEquirec(self, height, width):
        x, y = cp.meshgrid(cp.linspace(-180, 180, width, dtype=cp.float32), cp.linspace(90, -90, height, dtype=cp.float32))

        xyz = cp.zeros((height, width, 3), dtype=cp.float32)
        xyz[..., 0] = cp.cos(cp.radians(x)) * cp.cos(cp.radians(y))
        xyz[..., 1] = cp.sin(cp.radians(x)) * cp.cos(cp.radians(y))
        xyz[..., 2] = cp.sin(cp.radians(y))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = cp.dot(self.R2, xyz)
        xyz = cp.dot(self.R1, xyz).T
        xyz = xyz.reshape([height, width, 3])
        xyz /= xyz[..., 0, None]

        conditions = (-self.w_len < xyz[..., 1]) & (xyz[..., 1] < self.w_len) & (-self.h_len < xyz[..., 2]) & (xyz[..., 2] < self.h_len)

        lon_map = (xyz[..., 1] + self.w_len) / (2 * self.w_len) * self._width
        lat_map = (-xyz[..., 2] + self.h_len) / (2 * self.h_len) * self._height

        coordinates = cp.stack([cp.where(conditions, lat_map, 0), cp.where(conditions, lon_map, 0)], axis=0).astype(cp.float32)

        persp = cp.empty((height, width, self._img.shape[2]), dtype=self._img.dtype)
        for i in range(self._img.shape[2]):
            ndi.map_coordinates(self._img[..., i], coordinates, output=persp[..., i], order=1, mode='nearest')

        mask = conditions[..., cp.newaxis]  # Compute mask
        persp *= mask  # Apply mask to persp

        return cp.asnumpy(persp), cp.asnumpy(mask)

    def resetDevice():
        device = cuda.get_current_device()
        device.reset()