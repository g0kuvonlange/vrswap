import numpy as npy
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
from . import cupy_available, custom_cupy_wrap, convert_to_cupy, convert_to_numpy, to_np


__CONSISTENCY_THRESHOLD = 0.12

@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def construct_map(distorted_coor, undistorted_coor, res_hw, np=npy):
    """ 
    Given to sets of coordinates creates a map between them
    returns map from undistorted to distorted coordinates, 
    the map might contain NaN values where there are not distortion calculated
    """
    new_hw = (res_hw*undistorted_coor.max(axis=0)).astype(int)
    distorted_uv = np.array(undistorted_coor*(res_hw-1)).astype(int)

    # we need to only select an un coordinate once that is why we use uniquq
    # unique is not fully suported in cupy so we need to explicitly use numpy
    distorted_uv = to_np(distorted_uv)
    distorted_uv, unique_indices = npy.unique(distorted_uv, return_index=True, axis=0)
    dv, du = npy.split(distorted_uv, 2, axis=-1)

    map_undist2coef = np.ones((int(new_hw[0]), int(new_hw[1]), 2)).astype(np.float32) * -1
    map_undist2coef[dv, du] = distorted_coor[unique_indices].reshape(-1, 1, 2)/res_hw
    return map_undist2coef, new_hw


def sample_from_image_np(source_img, coor_x, coor_y ):
    return map_coordinates(source_img, [coor_y, coor_x],
                           order=1, mode='constant')[..., 0]


def sample_from_image_cp(source_img, coor_x, coor_y):
    import cupy as cp
    import numpy as np
    from cupyx.scipy.ndimage import map_coordinates

    e_img_gpu = cp.array(source_img)
    coords_gpu = cp.array([coor_y, coor_x])
    coords_gpu_1d = coords_gpu.reshape(len(coords_gpu), -1)
    gpu_map = map_coordinates(e_img_gpu, coords_gpu_1d, order=1)

    cpu_map = cp.asnumpy(gpu_map)
    cpu_map = cpu_map.transpose().reshape(coords_gpu.shape[1:])
    cpu_map = np.array(cpu_map.astype(np.float32))[..., 0]
    return cpu_map


def sample_from_img(source_img, coor_xy, clamp=True):
    """
    Constructs the image using the computed coordinates
        clamp: determines how texture is sampled when texture coord are out of the typical 0..1 range,
                If True texture edge pixels are stretched when outside of range.
    """
    import numpy as np
    coor_x, coor_y = np.split(coor_xy, 2, axis=-1)
    
    if clamp:
        coor_x = np.clip(coor_x, a_min=0, a_max=source_img.shape[1]-1)
        coor_y = np.clip(coor_y, a_min=0, a_max=source_img.shape[0]-1)

    if cupy_available():
        img = sample_from_image_cp(source_img, coor_x, coor_y)
    else:
        img = sample_from_image_np(source_img, coor_x, coor_y)
    return img


@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def sample_polygon(points, hw, full_img_fov_deg, np=npy):
    """
    Given a polygon on the sphere returns all the points inside it in eq coordinates
    :param points:  polygon xyz points on the sphere
    :param hw: resolution of the source image you want to sample from
    :param full_img_fov_deg: a tuple containing (hfov, vfov) of the perspective source image, if this is None the image is equirectangular
    """
    assert len(points) >= 3, "At least a triangle!"
    # a projection is needed only for perspective images
    projection = full_img_fov_deg is not None

    result = np.empty((0, 3))
    # rasterize poligon unsing triangle fan, obtain a set of xyz points
    for i in range(len(points)-2):

        # compute sample density
        mag_u = np.arccos(np.clip(points[0].dot(points[i+1]), a_min=-1, a_max=1))
        mag_v = np.arccos(np.clip(points[0].dot(points[i+2]), a_min=-1, a_max=1))
        mag = (mag_v+mag_u) / 2

        if projection:
            density = hw[1] * 1.5
        else:
            density = hw[1]/np.pi
            
        # create points using linear interpolation for the triangle
        u_rng = np.linspace(points[0], points[i+1], int(mag*density))
        v_rng = np.linspace(points[0], points[i+2], int(mag*density))

        mag = np.arccos(np.clip(points[i+1].dot(points[i+2]), a_min=-1, a_max=1))
        fill = np.linspace(u_rng, v_rng, int(mag * density))
        fill = fill.reshape(-1, 3)
        result = np.vstack((result, fill))
       
    # obtain the eq coordinates for the points
    polar = xyz2polar(result).reshape(-1, 2)
    if projection:
        coor = polar2coor_proj(polar, full_img_fov_deg, hw, (0, 0), plane_roll=0).astype(int)
    else:
        coor = polar2coor_eq(polar, hw).astype(int)

    # make sure we only keep unique values
    coor, indices = np.unique(coor, axis=0, return_index=True)
    return coor, polar[indices]


def check_uv_in_bounds(uv, hw):
    """
    Returns an array of True (for uv coordinates in array that are in hw bound) and False (otherwise)
    """
    return ((0 <= uv[:,0]) & (uv[:,0] < hw[1]) &
            (0 <= uv[:,1]) & (uv[:,1] < hw[0]))


def compute_focal_length_yx(fov, hw):
    """ Returns focal lenght given resolution and fov """
    hw = npy.array(hw)
    fov_vh = npy.array(fov[::-1])
    focal_length_yx = hw / 2 / npy.tan(npy.deg2rad(fov_vh) / 2)
    return focal_length_yx


def compute_fovs(focal_length_yx, hw):
    """ Returns pinhole camera fov given focal length and resolution """

    hw = npy.array(hw)
    focal_length_yx = npy.array(focal_length_yx)
    fov_vh = npy.degrees(npy.arctan(hw / (2 * focal_length_yx))) * 2
    return fov_vh[::-1]


def compute_hw(focal_length_yx, fov):
    """ Returns pinhole camera resolution given focal length and fov """
    focal_length_yx = npy.array(focal_length_yx)
    fov = npy.array(fov)

    fov_vh = npy.array(fov[::-1])
    hw = npy.round(2 * focal_length_yx *  npy.tan(npy.deg2rad(fov_vh) / 2)).astype(int)
    return hw


def check_fov_res_consistency(fov, hw, threshold=__CONSISTENCY_THRESHOLD):
    """
    check for square pixels given pinhole camera asumption
    """
    # default thr value is  0.117 for legacy resons as there ar many places in the code where we use
    # fov(125,94) and hw (1440,2560)
    fyx = compute_focal_length_yx(fov, hw)
    px_ar = fyx[1]/fyx[0]
    return npy.abs(1-(px_ar)) < threshold


def ensure_fov_res_consistency(fov, hw, threshold=__CONSISTENCY_THRESHOLD):
    """
    if we don't have square pixels we assume the vfov is correct and adjust hfov accordingly.
    """
    fyx = compute_focal_length_yx(fov, hw)
    px_ar = fyx[1]/fyx[0]
    res_ar = hw[0] / hw[1]
    # adjust the FOV depending on the AR of the image. If portrait, take hfov as reference, if landscape, vfov
    focal_length = fyx[1] if res_ar < 1 else fyx[0]
    if npy.abs(1 - px_ar) > threshold:
        fov = compute_fovs([focal_length, focal_length], hw)
    return fov

def rotation(yaw, pitch, roll):
    return R.from_euler("yxz",[yaw,pitch,roll], degrees=True)


def interpolate_nans(A):
    # all large empty space in the image are set to zero
    k_size = 9
    k = npy.ones((k_size,k_size))
    B = ndimage.convolve(A, k, mode='constant', cval=-1)
    A[B<0] = -1e-1
    # holes smaller than kernel size are interpolated
    ok = A > 0
    xp = ok.ravel().nonzero()[0]
    fp = A[ok]
    x  = (A==-1).ravel().nonzero()[0]
    A[A==-1] = npy.interp(x, xp, fp)
    return A
