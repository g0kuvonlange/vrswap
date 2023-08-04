import cv2
import numpy as np
from  sphere_snap.sphere_snap import SphereSnap
from sphere_snap.snap_config import SnapConfig
import sphere_snap.utils as snap_utils
from scipy.spatial.transform import Rotation as R

class TopViewProjector:
    def __init__(self, out_hw, fov_deg, height_meters, nb_of_views=5):
        """
        Extracts a top view image from an image (equirectangular/fisheye/pinhole prjections)
        :param out_hw: output resolution of the BEV image
        :param fov_deg: field of view in degress of the top view image
        :param height_meters: height in meters of the top view image
        :param nb_of_views: number of views used to construct the BEV(default 5)
        """

        self.out_hw = out_hw
        self.fov_deg = fov_deg
        self.height_meters = height_meters
 
        #create top camera mapping
        self.top_camera_mapping, self.px_per_m = TopViewProjector.__top_camera_mapping(self.out_hw, self.fov_deg, self.height_meters)
        self.nb_of_views = nb_of_views


        
    def __tv_image_from_snap(self, snap_config, upright_transform, full_image, view_height_meters):
        """
        Creates the birds eye view from a single image view
        """

        snap = SphereSnap(snap_config)
        m = snap_config.transform.as_matrix().dot(upright_transform)
        snap_config.set_transform(R.from_matrix(m)) 

        #extract uprighted image
        uprighted_snap = SphereSnap(snap_config)
        image = uprighted_snap.snap_to_perspective(full_image)
        
        #compute the inverse projection matrix 
        imv_p = TopViewProjector.__get_imv_P(snap, view_height_meters)
        ipm = np.linalg.inv(imv_p.dot(self.top_camera_mapping))
        
        #extract bev from view and remove noise using the mask
        bev = cv2.warpPerspective(image, ipm, (self.out_hw[1], self.out_hw[0]), flags=cv2.INTER_NEAREST)
        mask = TopViewProjector.__create_camera_mask(snap.snap_config.transform.as_euler('yxz', degrees=True)[0], self.out_hw)
        bev[mask] = 0
        return bev
    
    
    def get_snap_img_from_tv(self, tv_img, snap, image_height):
        """
        Projects the top view image onto the pinhole ground plane
        :param tv_img: TV image obtain using the same instance of TopViewProjector
        :param snap: an image view from the equirectangular image
        :param image_height: height in meters from the ground of the image view
        """
                    
        #project bev in view plane
        imv_p = TopViewProjector.__get_imv_P(snap, image_height)
        ipm = imv_p.dot(self.top_camera_mapping)
        img = cv2.warpPerspective(tv_img, ipm, (snap.snap_config.out_hw[1], snap.snap_config.out_hw[0]), flags=cv2.INTER_NEAREST)

        #obtain mask in view plane
        mask = TopViewProjector.__create_camera_mask(snap.snap_config.transform.as_euler('yxz', degrees=True)[0], tv_img.shape[:2]) * 255
        mask = cv2.warpPerspective(mask * np.ones((mask.shape[0],mask.shape[1],3)),
                                    ipm,
                                    snap.snap_config.out_hw[::-1],
                                    flags=cv2.INTER_NEAREST)
        img[mask[:,:,0] == 255] = [0,0,0]
        return img

    @staticmethod
    def __stich_images(tv_images):
        """
        Stiches the top view images obtained from image views to a top down bev image of the entire 360 image
        """
        birds_eye_view = np.zeros(tv_images[0].shape, dtype=np.uint8)
        for tv_image in tv_images:
            mask = np.any(tv_image != (0, 0, 0), axis=-1)
            birds_eye_view[mask] = tv_image[mask]
        return birds_eye_view

    @staticmethod
    def __top_camera_mapping(out_hw, fov_deg, height_meters):
        """
        Creates the mapping between a camera and the top down view
        """
        fov_hv_rad = np.deg2rad([fov_deg[1], fov_deg[0]])
        fx, fy = (np.array(out_hw)/2) / np.tan(fov_hv_rad / 2 ) 


        px_per_m = ( fy / height_meters, fx / height_meters)
        shift = (out_hw[0] / 2.0, out_hw[1] / 2.0)
        M = np.array([
                      [0.0, 1.0 / px_per_m[0], -shift[0] / px_per_m[0]],
                      [0.0, 0.0, 0.0], 
                      [1.0 / px_per_m[1], 0.0, -shift[1] / px_per_m[1]],
                      [0.0, 0.0, 1.0],
                    ])
        return M,px_per_m

    @staticmethod
    def __get_imv_P(snap, height):
        """
        Creates a projection matrix using pibhole model intrinsics, imv orientaton and
        a specified height from the ground
        :param snap: a snap from the equirectangular image
        :param height: height in meters from the ground of the camera
        """
        K = snap.K_matrix
        Rt = np.zeros([3, 4])
        R =  snap.snap_config.transform.as_matrix()
        Rt[0:3, 0:3] = R
        Rt[0:3, 3] = R.dot(np.array([0,height, 0]))
        return K.dot(Rt)

    @staticmethod
    def __create_camera_mask(yaw, output_res):
        """
        Creates a mask to later clip invalid parts from transformed image
        """
        mask = np.zeros((output_res[0], output_res[1], 3), dtype=np.bool)
        theta_y = np.linspace(output_res[0] / 2, -output_res[0] / 2, num=output_res[0], dtype=np.int)
        theta_x = np.linspace(-output_res[1] / 2, output_res[1] / 2, num=output_res[1], dtype=np.int)
        theta = np.zeros((output_res[0], output_res[1], 2), dtype=np.float64)
        theta[..., :2] = np.stack(np.meshgrid(theta_x, theta_y), -1)
        theta = np.rad2deg(np.arctan2(theta[:,:,1], theta[:,:,0])) - yaw
        mask[np.where(np.logical_and(90 < abs(theta), abs(theta) < 270))] = True
        return mask
    
    def get_view_optimal_res(self):
        """
        Returns the optimal resolution
        considering the views are looking down, and each has a 90 deg fov view
        so is approximately covering a quarter of the bev the view rezolution 
        """

        optimal_res = (np.array(self.out_hw) / 2) * 1.2
        return optimal_res.astype(int)

    def get_top_view_configuration(self, source_img_hw):
        view_hw = self.get_view_optimal_res()
        fov_step = int(360/self.nb_of_views)
        return [ SnapConfig( snap_utils.rotation( i*fov_step, 45, 0).as_quat() , view_hw, (90,90), source_img_hw) for i in range(self.nb_of_views)]

    
    def extract_tv(self, e_img, upright_transform, e_img_height_meters):
        """
        Extracts a top view image from an equirectangular projection
        :param e_img: source equirectangular image
        :param upright_transform: orientation of the upvector for the source image
        :param e_img_height_meters: height in meters of the source image
        """

        tv_snap_configs = self.get_top_view_configuration(e_img.shape[:2])
        #extract top view for each snap
        tv_img_list = []
        for snap_config in tv_snap_configs:
            bev_view_img = self.__tv_image_from_snap(snap_config,
                                                      upright_transform, 
                                                      e_img, 
                                                      e_img_height_meters)
            tv_img_list.append(bev_view_img)

        #stich all the snap into one image
        bev_img = TopViewProjector.__stich_images(tv_img_list)
        return bev_img

