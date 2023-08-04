import numpy as np
from scipy.spatial.transform import Rotation as R
import sphere_snap.utils as snap_utils
import sphere_snap.sphere_coor_projections as sphere_proj
from enum import Enum

class ImageProjectionType(Enum):
    EQUI = 'equirectangular'
    HALF_EQUI = 'half_equirectangular'
    PINHOLE = 'pinhole'
    FISHEYE_180 = 'fisheye180'
    RADIAL_DISTORTED = 'distorted_pinhole'



class SnapConfig:
    """
    Contains all properties needed to construct an image snap
    """

    def __init__(self, orientation_quat,
                 out_hw,
                 out_fov_deg,
                 source_img_hw,
                 source_img_fov_deg=None,
                 source_dist_coeff=None,
                 source_img_type=ImageProjectionType.EQUI
                ):

        """
        Given the input parameters, creates an ImageViewProperties object

        The camera coordinate system is (xy in the screen plane, z+ going in the screen):
             ^
           / z
         /          x
        |------------>
        |
        |
        |  y
        v
        where z is the viewing direction of the camera.
        This system follows the right-hand rule.

        :param fov_deg: a tuple containing (hfov, vfov)
        :param orientation_quat: array_like (4,1) of an orientation
        :param out_hw: a tuple containing output image size(height, width)
        :param source_dist_coeff: array containing distortion parameters, center offset should be normalised
                                {lambda, center_offset_x, center_offset_y} (Default: None)
        :param source_img_hw: a tuple containing source image size(height, width)
        :param source_img_fov_deg: a tuple containing (hfov, vfov) of the perspective source image, if this is None the image is equirectangular
        :param source_img_type: ImageProjectionType of the source image eg. (PINHOLE, EQUI, HALF_EQUI, FISHEYE_180, RADIAL_DISTORTED), default is EQUI.
        """

        rot_matrix = R.from_quat(orientation_quat).as_matrix()

        self.out_fov_deg = out_fov_deg
        self.out_hw = out_hw
        self.source_img_hw = source_img_hw
        self.source_img_fov_deg = source_img_fov_deg
        self.source_img_type = source_img_type

        # check fov and resolution consistency
        assert snap_utils.check_fov_res_consistency(out_fov_deg, out_hw), f"The created snap resolution {out_hw} is not consistent with fov {out_fov_deg} !"

        if (source_img_fov_deg is not None) and (snap_utils.check_fov_res_consistency(source_img_fov_deg, source_img_hw) == False):
            assert False , f"The image resolution {source_img_hw} is not consistent with fov {source_img_fov_deg} !"

        self.source_dist_coeff = source_dist_coeff
        self.set_transform(R.from_matrix(rot_matrix))

    def to_numpy(self):
        self.out_fov_deg=np.array(self.out_fov_deg)
        self.out_hw=np.array(self.out_hw)
        self.source_img_hw=np.array(self.source_img_hw)
        self.source_dist_coeff=np.array(self.source_dist_coeff) if self.source_dist_coeff is not None else self.source_dist_coeff
        self.source_img_fov_deg= np.array(self.source_img_fov_deg) if self.source_img_fov_deg is not None else  self.source_img_fov_deg

    @classmethod
    def createFromPoint(cls, center_point_xyz,
                        out_hw,
                        out_fov_deg,
                        source_img_hw,
                        source_img_fov_deg=None,
                        source_dist_coeff=None,
                        source_img_type=ImageProjectionType.EQUI,
                        quaternion = None,
                        ):
        """
        Given the input parameters, creates an ImageViewProperties object
        :param center_point_xyz: a point on the sphere where we want the view to be centered;
                for a point to be on the sphere, its corresponding position vector should have
                magnitude of 1 (= radius of sphere) and x, y, z in [-1, 1]
        :param out_fov_deg: a tuple containing (hfov, vfov)
        :param out_hw: a tuple containing output image size(height, width)
        :param source_dist_coeff: array containing distortion parameters,center offset should be normalised 
                                {lambda, center_offset_x, center_offest_y} (Default: None)
        :param source_img_hw: a tuple containing source image size(height, width)
        :param source_img_fov_deg: a tuple containing (hfov, vfov) of the perspective source image, if this is None the image is equirectangular
        :param quaternion: optional array_like (4,1) of an orientation, usually used to specify the upright orientation
        """                        

        # get corresponding (phi, theta) in radians of the point and transform them to degrees
        rot_uv_deg = np.degrees(sphere_proj.xyz2spherical(center_point_xyz))

        # if upright information exists, adjust rotation such that the targeted point remains the same
        if quaternion is not None:
            upright_matrix = R.from_quat(quaternion).as_matrix()
            rot_uv = sphere_proj.xyz2spherical(center_point_xyz.dot(upright_matrix.T))
            rot_uv_deg = np.degrees(rot_uv)

        rot_obj = R.from_euler('yxz', [-rot_uv_deg[0], rot_uv_deg[1], 0], degrees=True)
        if quaternion is not None:
            rot_obj = R.from_matrix(rot_obj.as_matrix().dot(R.from_quat(quaternion).as_matrix()))

        return cls( rot_obj.as_quat(),
                    out_hw, out_fov_deg, 
                    source_img_hw,
                    source_img_fov_deg=source_img_fov_deg, 
                    source_dist_coeff=source_dist_coeff,
                    source_img_type=source_img_type)

                   
    def _construct_name_for_snap(self):
        quat = self.transform.as_quat()
        dist_coeff = np.zeros(3) if self.source_dist_coeff is None else self.source_dist_coeff
        fov_str = "360" if self.source_img_fov_deg is None else f"{self.source_img_fov_deg[0]}_{self.source_img_fov_deg[1]}"
        return f"{self.out_fov_deg[0]}_{self.out_fov_deg[1]}_{quat[0]:0.4f}_{quat[1]:0.4f}" \
               f"_{quat[2]:0.4f}_{quat[3]:0.4f}_{self.out_hw[0]}_{self.out_hw[1]}" \
               f"_{dist_coeff[0]:0.4f}__{dist_coeff[1]:0.4f}__{dist_coeff[2]:0.4f}" \
               f"_{self.source_img_hw[0]}_{self.source_img_hw[1]}_{fov_str}_{self.source_img_type}"

    def __construct_center_snap_values(self):
        # this is the camera viewing point
        self.center_xyz = np.array([0,0,1])
        # (0, 0, 1) * R from (u, v, roll) to (0, 0, 1) * another R
        # now, viewing point is the point that corresponds to (u, v, roll), again rotated
        self.center_xyz = self.center_xyz.dot(self.transform.as_matrix())
        self.center_polar = sphere_proj.xyz2spherical(self.center_xyz)

    def set_transform(self, transform):
        self.transform = transform
        self.name = self._construct_name_for_snap()
        self.__construct_center_snap_values()

    # For call to repr(). Prints object's information
    def __repr__(self):
        return 'SnapConfig(%s)' % (self.name)

    # For call to str(). Prints readable form
    def __str__(self):
        return 'SnapConfig(%s)' % (self.name)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)