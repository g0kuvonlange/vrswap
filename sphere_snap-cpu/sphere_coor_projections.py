import numpy as npy
from sphere_snap import custom_cupy_wrap, convert_to_cupy, convert_to_numpy, to_np

@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def xyz2spherical(xyz, np=npy):
    """
    Converts from 3D cartesian coordinates to spherical (phi, theta) coordinates
    These are the "inverse" formula of `spherical2xyz()`.
    """
    x, y, z = np.split(xyz, 3, axis=-1)
    # divide equations of x and z from `polar2unitxyz`
    phi = np.arctan2(x, z)
    # raise each x and z to power 2 => (cos(v))^2
    c = np.sqrt(x ** 2 + z ** 2)
    # divide previously computed cos(v) by y
    theta = np.arctan2(y, c)
    return np.concatenate([phi, theta], axis=-1)


@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def spherical2xyz(phi_theta: npy.ndarray, np=npy):
    """
    Converts from spherical coordinates (in radians) to unit sphere 3D cartesian coordinates.
    """
    phi, theta = np.split(phi_theta, 2, axis=-1)
    y = np.sin(theta)
    x = np.cos(theta) * np.sin(phi)
    z = np.cos(theta) * np.cos(phi)
    return np.concatenate([x, y, z], axis=-1)      


@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def equi_coor2spherical(coorxy, hw, np=npy):
    """
    Converts from equirectangular coordinates to spherical coordinates
    The origin (0, 0) of the equirectangular image is in the top left corner.
    The input order is (horizontal, vertical).
    The return order is (phi, theta).
    """
    h,w = hw
    coor_x, coor_y = np.split(coorxy, 2, axis=-1)
    phi = ((coor_x + 0.5) / w - 0.5) * 2 * np.pi
    theta = ((coor_y + 0.5) / h - 0.5) * np.pi
    return np.concatenate([phi, theta], axis=-1)

@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def halfequi_coor2spherical(coorxy, hw, np=npy):
    """
    Converts from half equirectangular coordinates to spherical coordinates
    The origin (0, 0) of the half-equirectangular image is in the top left corner.
    The input order is (horizontal, vertical).
    The return order is (phi, theta).
    """
    h,w = hw
    coor_x, coor_y = np.split(coorxy, 2, axis=-1)
    phi   = ((coor_x + 0.5) / w - 0.5) * np.pi
    theta = ((coor_y + 0.5) / h - 0.5) * np.pi
    return np.concatenate([phi, theta], axis=-1)


@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def fisheye180_coor2xyz(coorxy, hw, return_excluded_indices=False, np=npy):
    """
    Converts from fisheye coordinates to xyz coordinates
    The origin (0, 0) of the fisheye image is in the top left corner.
    The input order is (horizontal, vertical).
    The return is a xyz np array.
    """
    coorxy = (coorxy - np.array(hw[::-1])/2)/(np.array(hw[::-1])/2)
    coor_x, coor_y = np.split(coorxy, 2, axis=-1)
    r = np.sqrt(coor_x**2+coor_y**2)+1e-10 # avoid r == 0
    cos_phi = coor_x/r
    # recovered_phi is the angle between x and  y components of the xyz vector
    recovered_phi = np.arccos(cos_phi)
    recovered_phi[ coorxy[:,1] < 0] = recovered_phi[ coorxy[:,1] < 0]*-1

    #recovered_theta angle between xy point from the xy plane, and z component of xyz vector
    recovered_theta = (r*np.pi)/2 
    recovered_pyx = np.sin(recovered_theta) # the lenght of the xy point in xy plane
    recovered_z = np.cos(recovered_theta) # z component of xyz vec
    recovered_x = np.cos(recovered_phi)*recovered_pyx  # x component of xyz vec
    recovered_y = np.sin(recovered_phi)*recovered_pyx  # y component of xyz vec
    xyz = np.concatenate([recovered_x, recovered_y, recovered_z], axis=-1)
    indices = (r>1).reshape(-1)
    
    #ignore indices outside the radius 1 as they are not part of the hemisphere
#    xyz[indices] = np.array([0,0,-1])
    if return_excluded_indices:
        return xyz, indices
    return xyz

@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def fisheye180_coor2spherical(coorxy, hw, return_excluded_indices=False, np=npy):
    """
    Converts from fisheye coordinates to spherical coordinates
    The origin (0, 0) of the equirectangular image is in the top left corner.
    The input order is (horizontal, vertical).
    The return order is (phi, theta).
    """
    data = fisheye180_coor2xyz._original(coorxy, hw,
                                                        return_excluded_indices=return_excluded_indices,
                                                        np=np)
    if return_excluded_indices:
        xyz, excl_indices = data
        return xyz2spherical._original(xyz, np=np), excl_indices
    return xyz2spherical._original(data, np=np)    



@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def fisheye180_spherical2coor(polar, output_hw, np=npy):
    aperture = 180 * np.pi / 180
    height,width = output_hw
    xyz = spherical2xyz(polar)
    x,y,z =  np.split(xyz, 3, axis=-1)
    p_yx = np.sqrt(y ** 2 + x ** 2 )
    # Convert normalized cartesian coordinates to fisheye image coordinates
    r = 2 * np.arctan2(p_yx, z) / aperture
    phi = np.arctan2(y, x)
    u = (r * np.cos(phi)) * width/2 + width/2
    v = (r * np.sin(phi)) * height/2 + height/2
    # make black everything outside of the image
    u[r>1]=0
    v[r>1]=0
    return np.concatenate([u, v], axis=-1)

def __gnomonic_projection(polar, center_polar, np=npy):
    """
    Projecting points on the surface of sphere in a plane that is tangent to a point center_polar
    http://mathworld.wolfram.com/GnomonicProjection.html

    :param polar: polar (phi, theta) coordinates for the sphere surface points that we want to project
    :param center_polar: a tuple that contains the polar coordinates of the tangent plane center
    """
    u, v = np.split(polar, 2, axis=-1)
    cosc = np.sin(center_polar[1]) * np.sin(v) + np.cos(center_polar[1])*np.cos(v)*np.cos(u - center_polar[0])
    # x = horizontal, v = vertical
    x = np.cos(v) * np.sin(u - center_polar[0]) / cosc
    y = (np.cos(center_polar[1]) * np.sin(v) - np.sin(center_polar[1]) * np.cos(v) * np.cos(u - center_polar[0]))/cosc
    return x, y


def __inverse_gnomic_projection(x, y, center_polar, np=npy):
    """
    Projecting points from a plane that is tangent to a point center_polar onto the surface of sphere 
    http://mathworld.wolfram.com/GnomonicProjection.html
    The equations are optimized for performance according to http://speleotrove.com/pangazer/gnomonic_projection.html

    :param x: x (horizontal) coordinate of the points in the plane that we want to project
    :param y: y (vertical) coordinate of the points in the plane that we want to project
    :param center_polar: a tuple that contains the polar coordinates of the tangent plane center
    """
    xy = np.concatenate([x, y], axis=-1)
    p = np.linalg.norm(xy, axis=-1).reshape(-1, 1)
    c = np.arctan(p)
    cosc = 1.0/np.sqrt(1 + p*p)
    cos_cp = np.cos(center_polar)
    sin_cp = np.sin(center_polar)
    u = center_polar[0] +  np.arctan2(x, (cos_cp[1] - y * sin_cp[1])) 
    u = np.arctan2(np.sin(u), np.cos(u)) #change u range to [-pi, -pi]
    v = np.arcsin( (sin_cp[1] + (y * cos_cp[1])) * cosc)
    return u, v


def __rotate_origin_only(x, y, radians, np=npy):
    """
    Only rotate a point around the origin (0, 0).
    This rotation happens in 2D, clockwise.
    See https://en.wikipedia.org/wiki/Rotation_matrix - "Direction" for more details.
    """
    # TODO: change rotation to counter-clockwise, to be aligned with our coor sys?
    xx = x * np.cos(radians) + y * np.sin(radians)
    yy = -x * np.sin(radians) + y * np.cos(radians)
    return xx, yy


@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def pinhole_coor2spherical(uv, img_fov_deg, full_img_hw, c_phi_theta, plane_roll, np=npy):
    """ 
    Converts from perspective image plane coordinates to spherical (phi, theta) coordinates
    :param full_img_hw: a tuple containing source image size(height, width)
    :param img_fov_deg: a tuple containing (hfov, vfov) of the perspective source image
    :param uv: image coordinates for the 2d points on the plane we want to project
    :param c_phi_theta: a tuple that contains the polar coordinates of the tangent plane center
    :param plane_roll:  plane roll in radians
    """
    wh = full_img_hw[::-1]-1
    hv_fov = np.deg2rad(img_fov_deg)

    # compute the pixel factor in the source image.
    pixel_factor_wh = wh / np.tan(hv_fov/2)
    # shift coordinates to image center and from [0,s] to [-s,s]
    uv = (2*uv) - wh
    # compute uv coordinates for each pixel in the normalized plane
    xy = uv / pixel_factor_wh
    x, y = np.split(xy, 2, axis=-1)
    #apply view roll
    x, y = __rotate_origin_only(x, y, plane_roll, np=np)
    # project current plane patch view on sphere
    u, v = __inverse_gnomic_projection(x, y, c_phi_theta, np=np)
    spherical = np.concatenate([u, v], axis=-1)
    return spherical


@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def pinhole_spherical2coor(spherical, img_fov_deg, full_img_hw, c_phi_theta, plane_roll, np=npy):
    """
    Project points from the sphere surface to a tangential plane
    returns the pixel's coordinates in the perspective image

    :param spherical: spherical coordinates for the sphere surface points that we want to project
    :param img_fov_deg: a tuple containing (hfov, vfov) of the perspective source image
    :param full_img_hw: a tuple containing source image size(height, width)
    :param c_phi_theta: a tuple that contains the spherical coordinates of the tangent plane center
    :param plane_roll:  plane roll in radians
    """
    wh = full_img_hw[::-1] - 1
    hv_fov = np.radians(img_fov_deg)
    # project current sphere patch view on plane
    u, v = __gnomonic_projection(spherical, c_phi_theta, np=np)
    # apply view roll
    u, v = __rotate_origin_only(u, v, -plane_roll, np=np)
    uv = np.concatenate([u, v], axis=-1)
    # compute the pixel factor in the source image. 
    pixel_factor_wh = wh / np.tan(hv_fov/2)
    # compute uv coordinates for each pixel in output image
    coor_xy = uv * pixel_factor_wh
    # shift coordinates to image center and from [-s,s] to [0,s]
    coor_xy = coor_xy * 0.5 + wh * 0.5
    return coor_xy


@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def equi_spherical2coor(spherical, output_hw, np=npy):
    """ 
    For each pixel visible in the view image this function
        returns the pixel's coordinates in the source equirectangular image

    The order of coordinates in `spherical` is (phi, theta).
    The order of coordinates in the output is (horizontal=along width, vertical=along height).
    """
    h = output_hw[0]
    w = output_hw[1]
    phi, theta = np.split(spherical, 2, axis=-1)
    # (phi, theta) = (0, 0) corresponds to center of equi image
    coor_x = (phi / (2 * np.pi) + 0.5) * w - 0.5
    coor_y = (theta / np.pi + 0.5) * h - 0.5
    return np.concatenate([coor_x, coor_y], axis=-1)

@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def halfequi_spherical2coor(spherical, output_hw, np=npy):
    """ 
    For each pixel visible in the view image this function
        returns the pixel's coordinates in the source half-equirectangular image

    The order of coordinates in `spherical` is (phi, theta).
    The order of coordinates in the output is (horizontal=along width, vertical=along height).
    """
    h = output_hw[0]
    w = output_hw[1]
    phi, theta = np.split(spherical, 2, axis=-1)
    # (phi, theta) = (0, 0) corresponds to center of equi image
    coor_x = (phi   / np.pi + 0.5) * w - 0.5
    coor_y = (theta / np.pi + 0.5) * h - 0.5
    return np.concatenate([coor_x, coor_y], axis=-1)


def division_model1(yx, dist_coeff, hw, np=npy):
    """ 
    Fitzgibbon division model with one parameter 
    More details about this model can be found in the paper:
    https://www.robots.ox.ac.uk/~vgg/publications/papers/fitzgibbon01b.pdf
    :param yx: coordinates of the distorted pixels
    :param dist_coeff: array containing distortion parameters, offset center should be normalised {lamda, center_offset_x, center_offest_y}
    :param hw: image resolution
    :return: the normalized coordinates and 0 centered
    """
    center = dist_coeff[1:3]
    recentered_yx = (yx/hw - center)
    r = np.linalg.norm(recentered_yx, axis=-1)
    coeff =  (1 + dist_coeff[0] * np.power(r,2)) 
    coor = recentered_yx / coeff[..., np.newaxis]
    return coor


@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def undistort(yx, dist_coeff, hw, np=npy, normalize_values=False):
    """ 
    For each distorted image pixel `yx`, compute the coordinates in the undistorted image,
    using Fitzgibbon model with one parameter

    :param yx: coordinates of the distorted pixels
    :param dist_coeff: array containing distortion parameters, offset center should be normalised {lamda, center_offset_x, center_offest_y}
    :param hw: image resolution, (height, width)
    :param normalize_values: if True, do min-max normalization (in [0, 1] range) of the coordinates
    :return: the coordinates in the undistorted image
    """
    # pick some control points (corners and midpoints) to find maximum expansion of the coordinates
    max_coor = npy.array(to_np(hw)) - 1
    control_points = np.array([[0, 0], [max_coor[0], 0],
                               [0, max_coor[1]], max_coor,
                               [0, max_coor[1] / 2], [max_coor[0] / 2, 0],
                               [max_coor[0], max_coor[1] / 2], [max_coor[0] / 2, max_coor[1]]
                               ])

    coor = division_model1(yx, dist_coeff, hw, np=np)
    coor_cp = division_model1(control_points, dist_coeff, hw, np=np)
    # make coordinates start from 0
    coor = coor - coor_cp.min(axis=0)
    if normalize_values:
        # min-max normalization
        coor = coor / (coor_cp.max(axis=0) - coor_cp.min(axis=0))
    return coor

@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def get_2D_coor_grid(img_hw,np=npy):
    """ 
    Returns a matrix with the pixel coordinates
    """
    x_rng = np.linspace(0, img_hw[1]-1, num=img_hw[1])
    y_rng = np.linspace(0, img_hw[0]-1, num=img_hw[0])
    return np.stack(np.meshgrid(x_rng, y_rng), -1).reshape(-1, 2).astype(int)

@custom_cupy_wrap(convert_to_cupy, convert_to_numpy)
def xyz_to_hequirect(self, vec):
    """
    Maps a 3D coordinate on the sphere to its corresponding position on a half equirectangular image.
    :param vec: The 3D point(s) on the sphere. Can be a single 3D point or an array of points.
    :return: The 2D coordinate(s) on the half equirectangular image.
    """
    print("----------------> xyz_to_hequirect <--------------")
    # Ensure vec is a 2D array for batch processing
    vec = np.atleast_2d(vec)

    width, height = self.snap_config.out_hw

    phi   = np.arctan2(vec[:, 0], vec[:, 2]) / np.pi * 0.5
    theta = np.arcsin(vec[:, 1]) / np.pi * 0.5

    uf = self._scale(phi, width)
    vf = self._scale(theta, height)

    ui = np.floor(uf).astype(int)
    vi = np.floor(vf).astype(int)

    visible = (phi >= -0.5) & (phi <= 0.5)

    us = np.clip(ui[:, np.newaxis] + np.arange(-1, 3), 0, width - 1)
    vs = np.clip(vi[:, np.newaxis] + np.arange(-1, 3), 0, height - 1)

    du = uf - ui
    dv = vf - vi

    return us, vs, du, dv, visible

# Helper function for scaling
def _scale(self, val, max_val):
    """Helper function to scale values."""
    return (val + 0.5) * max_val
