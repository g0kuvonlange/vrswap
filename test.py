"""
This module demonstrates the conversion of an equirectangular image to a half-equirectangular image using SphereSnap.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from sphere_snap.sphere_snap import SphereSnap, SnapConfig, ImageProjectionType
from sphere_snap.reprojections import get_cube_map_faces
from sphere_snap import sphere_coor_projections as sphere_proj
from sphere_snap import utils as snap_utils
from sphere_snap.snap_config import SnapConfig
from sphere_snap.snap_config import ImageProjectionType
import matplotlib.pyplot as plt

def __rot(yaw, pitch):
    return R.from_euler("yxz",[yaw,-pitch,0], degrees=True).as_quat()

# Read the equirect_test_pattern.png image
equirectangular_image = cv2.imread('equirect_test_pattern.png', cv2.IMREAD_COLOR)

# Convert the equirectangular image to a cubemap image
face_res = equirectangular_image.shape[1]//4
# Inline get_cube_map_faces function
snap_configs = [SnapConfig( __rot(90*i,0), (face_res,face_res),(90,90), equirectangular_image.shape[:2])
                    for i in range(4)]
# top
snap_configs.append(SnapConfig( __rot(0,90), (face_res,face_res),(90,90), equirectangular_image.shape[:2]))
# bottom
snap_configs.append(SnapConfig( __rot(0,-90), (face_res,face_res),(90,90), equirectangular_image.shape[:2]))
cube_face_snap_front, cube_face_snap_left, cube_face_snap_back, cube_face_snap_right, cube_face_snap_top, cube_face_snap_bottom = [SphereSnap(c) for c in snap_configs]
cubemap_front, cubemap_left, cubemap_back, cubemap_right, cubemap_top, cubemap_bottom = [snap.snap_to_perspective(equirectangular_image) for snap in [cube_face_snap_front, cube_face_snap_left, cube_face_snap_back, cube_face_snap_right, cube_face_snap_top, cube_face_snap_bottom]]

# Display each cubemap image
#for i, img in enumerate(cubemap_images):
#    plt.figure()
#    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#    plt.title(f'Cubemap Image {i+1}')
#    plt.axis('off')
#    plt.show()

# Create a custom fov perspective and show it 

equi_persp_test_snapconfig = SnapConfig( __rot(-23,-25), (face_res,face_res), (18,18), equirectangular_image.shape[:2])
equi_persp_test_snap = SphereSnap(equi_persp_test_snapconfig)
equi_persp_test_image = equi_persp_test_snap.snap_to_perspective(equirectangular_image)

## Display the resulting image
#plt.figure()
#plt.imshow(cv2.cvtColor(equi_persp_test_image, cv2.COLOR_BGR2RGB))
#plt.title('Equirectangular to persp test image')
#plt.axis('off')
#plt.show()
#


# Replace the cubemap2equi function call with the actual code
face_size = cubemap_front.shape[0]
equi_hw = np.array([2*face_size, 4*face_size]).astype(int)
# Inline get_cube_map_faces function
snap_configs = [SnapConfig( __rot(90*i,0), (face_size,face_size),(90,90), equi_hw)
                    for i in range(4)]
# top
snap_configs.append(SnapConfig( __rot(0,90), (face_size,face_size),(90,90), equi_hw))
# bottom
snap_configs.append(SnapConfig( __rot(0,-90), (face_size,face_size),(90,90), equi_hw))
cube_face_snap_front, cube_face_snap_left, cube_face_snap_back, cube_face_snap_right, cube_face_snap_top, cube_face_snap_bottom = [SphereSnap(c) for c in snap_configs]
equirectangular_image_front = SphereSnap.merge_multiple_snaps(equi_hw, [cube_face_snap_front, cube_face_snap_left, cube_face_snap_right, cube_face_snap_top, cube_face_snap_bottom], [cubemap_front, cubemap_left, cubemap_right, cubemap_top, cubemap_bottom], target_type=ImageProjectionType.EQUI, merge_method="max")

## Display the resulting equirectangular image
#plt.figure()
#plt.imshow(cv2.cvtColor(equirectangular_image_front, cv2.COLOR_BGR2RGB))
#plt.title('Equirectangular Image Back from Cubemap')
#plt.axis('off')
#plt.show()

print("=== NEW START HERE ===")
# Generate the half-equirectangular image with 1:1 aspect ratio
#hequirect_hw = np.array([min(equi_hw), min(equi_hw)]).astype(int)
#hequirect_front = SphereSnap.merge_multiple_snaps(hequirect_hw, [cube_face_snap_front, cube_face_snap_left, cube_face_snap_right, cube_face_snap_back, cube_face_snap_top, cube_face_snap_bottom], [cubemap_front, cubemap_left, cubemap_right, cubemap_back, cubemap_top, cubemap_bottom], target_type=ImageProjectionType.HALF_EQUI, merge_method="max")

hequirect_front = cv2.imread('hequirect_test_pattern.png')
hequirect_hw = hequirect_front.shape[:2]
# Display the resulting half-equirectangular image
plt.figure()
plt.imshow(cv2.cvtColor(hequirect_front, cv2.COLOR_BGR2RGB))
plt.title('Half-Equirectangular Image Back from Cubemap')
plt.axis('off')
plt.show()
#
# Define the bounding box coordinates
x1, x2, y1, y2 = 1640, 1900, 1440, 1650


# Crop the hequirect_front image
hequirect_front_bbox = hequirect_front[y1:y2, x1:x2]

# Display the cropped image
plt.figure()
plt.imshow(cv2.cvtColor(hequirect_front_bbox, cv2.COLOR_BGR2RGB))
plt.title('Cropped Half-Equirectangular Image')
plt.axis('off')
plt.show()

print("===== START HERE ===")
# Calculate the center coordinates
x_center = (x1 + x2) / 2
y_center = (y1 + y2) / 2

# Calculate phi and theta
phi, theta = sphere_proj.halfequi_coor2spherical(np.array([x_center, y_center]), (hequirect_front.shape[1], hequirect_front.shape[0]))

# Calculate the FOV based on the size of the bounding box
fov_x = ((x2 - x1) / hequirect_front.shape[1]) * 180
fov_y = ((y2 - y1) / hequirect_front.shape[0]) * 90

crop_fov = max(fov_x, fov_y)
crop_length = max(x2 - x1, y2 - y1)

yaw = np.degrees(phi)
pitch = np.degrees(theta)

rotation_quat = R.from_euler("yxz", [-yaw, pitch, 0], degrees=True).as_quat()
adjusted_fov = snap_utils.ensure_fov_res_consistency((crop_fov, crop_fov), (crop_length, crop_length))
print(f"phi {phi} theta {theta} h {hequirect_front.shape[1]} w {hequirect_front.shape[0]} crop fov {crop_fov} crop_length {crop_length} adjusted_fov {adjusted_fov}")

snap_config = SnapConfig(
    orientation_quat=rotation_quat, 
    out_hw=(hequirect_front.shape[1], hequirect_front.shape[0]),
    out_fov_deg=adjusted_fov,
    source_img_hw=(hequirect_front.shape[1],hequirect_front.shape[0]),
    source_img_fov_deg=(180,180), 
    source_img_type=ImageProjectionType.HALF_EQUI
)



# Create a SphereSnap object using the snap_config
sphere_snap_obj = SphereSnap(snap_config)

# Generate a perspective image using the snap_to_perspective() method
perspective_image = sphere_snap_obj.snap_to_perspective(hequirect_front)

# Display the generated perspective image
plt.figure()
plt.imshow(cv2.cvtColor(perspective_image, cv2.COLOR_BGR2RGB))
plt.title('Perspective Image')
plt.axis('off')
plt.show()


# Set the perspective_image_modified to be all white
perspective_image_modified = 255 * np.ones_like(perspective_image)

# Create the modified_halfequi_image using pinhole2halfequi_single
#from sphere_snap.reprojections import pinhole2halfequi_single
#modified_halfequi_image = pinhole2halfequi_single(perspective_image_modified, (phi, theta), adjusted_fov, out_hw=(hequirect_front.shape[1], hequirect_front.shape[0]))
# Display the generated perspective image
plt.figure()
plt.imshow(cv2.cvtColor(perspective_image_modified, cv2.COLOR_BGR2RGB))
plt.title('Modified Perspective Image')
plt.axis('off')
plt.show()



perspective_image_config = SnapConfig(
    orientation_quat=rotation_quat, 
    out_hw=(hequirect_front.shape[1], hequirect_front.shape[0]),
    out_fov_deg=adjusted_fov,
    source_img_hw=(hequirect_front.shape[1],hequirect_front.shape[0]),
    source_img_fov_deg=(180,180), 
    source_img_type=ImageProjectionType.HALF_EQUI
)


perspective_image_snap = SphereSnap(perspective_image_config)

# Create a mask of the same size as perspective_image_modified with all pixels set to 1
perspective_image_mask = np.ones_like(perspective_image_modified)

# Generate the mask in the half-equirectangular space
reconstructed_hequi_mask = SphereSnap.merge_multiple_snaps((hequirect_front.shape[1],hequirect_front.shape[0]),
                                                           [perspective_image_snap], # snap object specifies destination position
                                                           [perspective_image_mask], # snap image contains mask pixels
                                                           target_type=ImageProjectionType.HALF_EQUI, # destination image type
                                                           )

# Generate the reconstructed_hequi image as before
reconstructed_hequi = SphereSnap.merge_multiple_snaps((hequirect_front.shape[1],hequirect_front.shape[0]),
                                                     [perspective_image_snap], # snap object specifies destination position
                                                     [perspective_image_modified], # snap image contains planar image pixels
                                                     target_type=ImageProjectionType.HALF_EQUI, # destination image type
                                                     )

# Find the non-zero pixels in the reconstructed_hequi_mask
non_zero_pixels = np.where(reconstructed_hequi_mask != 0)

# Create a copy of hequirect_front to modify
overlayed_image = hequirect_front.copy()

# Replace the non-zero pixels in hequirect_front with the corresponding pixels from reconstructed_hequi
overlayed_image[non_zero_pixels] = reconstructed_hequi[non_zero_pixels]

# Display the modified_halfequi_image
plt.figure()
plt.imshow(cv2.cvtColor(reconstructed_hequi, cv2.COLOR_BGR2RGB))
plt.title('Modified Half-Equirectangular Image')
cv2.imwrite('output.jpg', reconstructed_hequi)
plt.axis('off')
plt.show()

# overlay the colors of reconstructed_hequi onto hequirect_front

# Find the non-black pixels in the reconstructed_hequi image
non_black_pixels = np.where(np.any(reconstructed_hequi != [0, 0, 0], axis=-1))

# Create a copy of hequirect_front to modify
overlayed_image = hequirect_front.copy()

# Replace the non-black pixels in hequirect_front with the corresponding pixels from reconstructed_hequi
overlayed_image[non_black_pixels] = reconstructed_hequi[non_black_pixels]

# Display the overlayed_image
plt.figure()
plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
plt.title('Overlayed Half-Equirectangular Image')
plt.axis('off')
plt.show()

#
## define a spheresnap for the half equi original (again)
#
#rotation_quat = R.from_euler("yxz", [0, 0, 0], degrees=True).as_quat()
#orig_snap_config = SnapConfig(
#    orientation_quat=rotation_quat, 
#    out_hw=(hequirect_front.shape[1], hequirect_front.shape[0]),
#    out_fov_deg=(180,180),
#    source_img_hw=(hequirect_front.shape[1],hequirect_front.shape[0]),
#    source_img_fov_deg=None,
#    source_img_type=ImageProjectionType.HALF_EQUI
#)
#
#orig_snap_obj = SphereSnap(orig_snap_config)
#
## merge the modified image into the originalg
#modded_hequirect_front = SphereSnap.merge_multiple_snaps(hequirect_hw, [orig_snap_obj, perspective_image_snap], [hequirect_front,perspective_image_modified], target_type=ImageProjectionType.HALF_EQUI, merge_method="last")
#
## Display the m,eged
#plt.figure()
#plt.imshow(cv2.cvtColor(modded_hequirect_front, cv2.COLOR_BGR2RGB))
#plt.title('Modified Injected Half-Equirectangular Image')
#plt.axis('off')
#plt.show()
#
#
