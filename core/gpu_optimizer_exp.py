from PIL import Image as PILImage
import subprocess
import shutil
import insightface
import core.globals
from core.swapper import get_face_swapper
from core.analyser import get_face, get_faces, get_face_analyser
from threading import Thread
import cv2
from tqdm import tqdm
import os
import time
from multiprocessing import Pool
import core.lib.Equirec2Perspec as E2P
import core.lib.Perspec2Equirec as P2E
from pathlib import Path
from math import pi
import numpy as np
from scipy.ndimage import gaussian_gradient_magnitude
import piexif
from core.pnginfo import load_pnginfo
from concurrent.futures import ProcessPoolExecutor

#creates a thread and returns value when joined
class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def upscaling_thread(frame_path):
    yes_upscaling = True

    result = upscale_image(frame_path)

    frame_number = os.path.splitext(os.path.basename(frame_path))[0]
    with open('./upscaled.txt', 'w') as file:
        file.write(str(frame_number))

    #returning if we got face and result frame 
    return yes_upscaling, result

def convert_thread(frame_path):
    yes_convert = True
    result = None

    frame_name = os.path.splitext(os.path.basename(frame_path))[0]  # 000001
    frame_folder = os.path.splitext(frame_path)[0]                  # D:/test/000001
    output_folder = os.path.dirname(frame_path)                     # D:/test
    processing_folder = output_folder + "/processing"

    result = persp2equir(frame_path, processing_folder)
    
    shutil.move(processing_folder + '/' + frame_name + '.png', output_folder + "/" + frame_name + ".png")
    if os.path.exists(processing_folder + '/' + frame_name + '_1.png'):
        os.remove(processing_folder + '/' + frame_name + '_1.png')
    if os.path.exists(processing_folder + '/' + frame_name + '_2.png'):
        os.remove(processing_folder + '/' + frame_name + '_2.png')
    if os.path.exists(frame_path):
        os.remove(frame_path)


    #returning if we got face and result frame 
    return yes_convert, result


def face_analyser_thread(frame_path, source_face, vr = True):
    yes_face = True
    result = None  # Initialize result
    
    # Load the frame
    frame = load_frame(frame_path)

    if vr:
        frame_name = os.path.splitext(os.path.basename(frame_path))[0]  # 000001
        frame_folder = os.path.splitext(frame_path)[0]                  # D:/test/000001
        output_folder = os.path.dirname(frame_path)                     # D:/test
        processing_folder = output_folder + "/processing"
        
        result = equir2pers(frame_path, processing_folder)

        left = f"{processing_folder}/{frame_name}_1.jpg"
        right = f"{processing_folder}/{frame_name}_2.jpg"

        img1 = perform_face_swap(left, source_face)
        img2 = perform_face_swap(right, source_face)

    return yes_face, result



def process_video_gpu(source_img, frame_paths, gpu_threads, swap_face, upscale_face, convert):
    frames_path = os.path.dirname(frame_paths[0])
    processing_path = os.path.dirname(frame_paths[0]) + "/processing"
    start_frame = os.path.splitext(os.path.basename(frame_paths[0]))[0]

    if upscale_face:
        for i in range(gpu_threads):
            launch_codeformer(processing_path)
            time.sleep(4)        
        return

    if swap_face:
        global face_analyser, swap
        swap = get_face_swapper()
        face_analyser = get_face_analyser()
        source_face = get_face(cv2.imread(source_img))

        # Create folder [frame_path]
        if not os.path.exists(processing_path):
            os.mkdir(processing_path)
    temp = []
    with tqdm(total=len(frame_paths), desc='Processing', unit="frame", dynamic_ncols=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as progress:
        for frame_path in frame_paths:
            if swap_face:
                frame_name = os.path.splitext(os.path.basename(frame_path))[0]
                output_folder = os.path.dirname(frame_path)
                processing_folder = output_folder + "/processing"

                left = f"{processing_folder}/{frame_name}_1.jpg"
                right = f"{processing_folder}/{frame_name}_2.jpg"

                left_exists = os.path.exists(left)
                right_exists = os.path.exists(right)

                if left_exists and right_exists:
                    print("Both left and right files exist.")
                    progress.set_postfix(status='S', refresh=True)
                    progress.update(1)
                else:
                    while len(temp) >= gpu_threads:
                        #we are order dependent, so we are forced to wait for first element to finish. When finished removing thread from the list
                        has_face, x = temp.pop(0).join()

                        if has_face:
                            progress.set_postfix(status='.', refresh=True)
                        else:
                            progress.set_postfix(status='S', refresh=True)
                        progress.update(1)
                    #adding new frame to the list and starting it 
                    temp.append(ThreadWithReturnValue(target=face_analyser_thread, args=(frame_path, source_face)))
                    temp[-1].start()
            
            if convert:
                while len(temp) >= gpu_threads:
                    has_result, x = temp.pop(0).join()

                    if has_result:
                        progress.set_postfix(status='.', refresh=True)
                    else:
                        progress.set_postfix(status='S', refresh=True)
                    progress.update(1)
                temp.append(ThreadWithReturnValue(target=convert_thread, args=(frame_path,)))
                temp[-1].start()



                    


def load_frame(frame_path):
    # Read the frame from the SBS VR video
    frame = cv2.imread(frame_path)

    return frame


def split_sbs_frame(frame):
    # Get the width of the frame
    width = frame.shape[1]

    # Split the frame into left and right halves
    left_half = frame[:, :width // 2]
    right_half = frame[:, width // 2:]

    return left_half, right_half


def perform_face_swap(frame_path, source_face):
    face_exists = os.path.exists(frame_path)

    if not face_exists:
        print("Face doesn't exist, skip")
        return

    theta, phi = load_exif_info(frame_path)
    frame = cv2.imread(frame_path) 

    target_face = get_face(frame)
    swapped_frame = frame

    if target_face:
        # Perform face swapping on the frame using source_face and target_face
        swapped_frame = swap.get(frame, target_face, source_face, paste_back=True)
        cv2.imwrite(frame_path, swapped_frame)
        store_exif_info(frame_path, theta, phi)

    return swapped_frame


def merge_sbs_frame(left_half, right_half):
    # Concatenate the left and right halves horizontally to merge them
    merged_frame = cv2.hconcat([left_half, right_half])

    return merged_frame


def save_frame(frame, frame_path):
    # Save the frame back to the frame path
    cv2.imwrite(frame_path, frame)

    # extract frame count from filename
    frame_number = os.path.splitext(os.path.basename(frame_path))[0]

    with open('./frames.txt', 'w') as file:
            file.write(str(frame_number))
    
    return frame


def launch_codeformer(input_folder):
        codeformer_command = f'python CodeFormer/inference_roop2.py -i "{input_folder}" -o "{input_folder}/upscaled" -w 0.85 -s 1 --face_upsample --rewrite'
        print(codeformer_command)
        subprocess.Popen(codeformer_command, shell=True)
        #subprocess.run(codeformer_command, shell=True, check=True)


def upscale_image(input_file, max_retries=5):
    for retry in range(max_retries):
        try:
            file_basename = os.path.splitext(os.path.basename(input_file))[0]
            codeformer_folder_name = os.path.splitext(input_file)[0]
            codeformer_generated_file = codeformer_folder_name + '/final_results/' + file_basename + ".png"
            codeformer_command = f'python CodeFormer/inference_codeformer.py -i "{input_file}" -o "{codeformer_folder_name}" -w 0.85 -s 1 --face_upsample'
            subprocess.run(codeformer_command, shell=True, check=True)
            shutil.move(codeformer_generated_file, codeformer_folder_name + ".png")
            shutil.rmtree(codeformer_folder_name)
            os.remove(input_file)
            return True

        except subprocess.CalledProcessError:
            # If the operation failed, wait for a short period
            time.sleep(10)

    # If the operation failed after max_retries, return False
    return False


def equir2pers(input_img, output_dir):    

    frame_name = os.path.splitext(os.path.basename(input_img))[0]

    face = get_faces(cv2.imread(input_img))
    face = face[:2]  # Consider the first two detected faces (left and right eyes)

    for idx, face_data in enumerate(face):
        bbox = face_data.bbox

        # Convert bounding box to ints
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        
        # Load equirectangular image
        equ = E2P.Equirectangular(input_img)    

        # Determine the center of the bounding box
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        # Normalize coordinates to range [-1, 1]
        x_center_normalized = x_center / (equ.get_width() / 2) - 1
        y_center_normalized = y_center / (equ.get_height() / 2) - 1

        # Convert normalized coordinates to spherical (theta, phi)
        theta = x_center_normalized * 180  # Theta ranges from -180 to 180 degrees
        phi = -y_center_normalized * 90  # Phi ranges from -90 to 90 degrees

        img = equ.GetPerspective(90, theta, phi, 1280, 1280)  # Generate perspective image
        output_path = os.path.join(output_dir, f'{frame_name}_{idx+1}.jpg')
        cv2.imwrite(output_path, img)
        store_exif_info(output_path, theta, phi)


def persp2equir_run(input_img, theta, phi, orig_img):
    orig_height, orig_width = orig_img.shape[:2]
    persp = P2E.Perspective(input_img, 90, theta, phi)
    img, _ = persp.GetEquirec(orig_height, orig_width)

    # Compute masks and apply feathering
    mask = np.sum(img, axis=2) > 0
    feathered_mask = apply_feathering(mask)

    return img, feathered_mask


def persp2equir(frame_path, input_folder):
    orig_img = load_frame(frame_path)
    orig_height, orig_width = orig_img.shape[:2]

    frame_name = os.path.splitext(os.path.basename(frame_path))[0]

    input_img1 = input_folder + '/' + frame_name + '_1.png'
    input_img2 = input_folder + '/' + frame_name + '_2.png'

    img1 = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)
    img2 = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)

    with ProcessPoolExecutor(max_workers=2) as executor:
        if os.path.exists(input_img1):
            theta1, phi1 = load_pnginfo(input_img1)
            img1, feathered_mask1 = executor.submit(persp2equir_run, input_img1, theta1, phi1, orig_img).result()
        if os.path.exists(input_img2):
            theta2, phi2 = load_pnginfo(input_img2)
            img2, feathered_mask2 = executor.submit(persp2equir_run, input_img2, theta2, phi2, orig_img).result()

    # Trim the right half of img1 (left eye) and the left half of img2 (right eye)
    half_width = orig_img.shape[1] // 2
    img1[:, half_width:] = 0
    img2[:, :half_width] = 0

    # Create a mask of where the transformed perspectives are not black
    mask1 = np.sum(img1, axis=2) > 0
    mask2 = np.sum(img2, axis=2) > 0

    # Create composite images by blending the feathered perspectives with the original image
    composite1 = orig_img * (1 - feathered_mask1[..., np.newaxis]) + img1 * feathered_mask1[..., np.newaxis]
    composite2 = orig_img * (1 - feathered_mask2[..., np.newaxis]) + img2 * feathered_mask2[..., np.newaxis]

    # Replace pixels in the original image with those from the transformed perspectives respecting the feathered masks
    mask1 = np.sum(img1, axis=2) > 0
    mask2 = np.sum(img2, axis=2) > 0
    orig_img[mask1] = composite1[mask1]
    orig_img[mask2] = composite2[mask2]

    # Save the result
    result = cv2.imwrite(input_folder + '/' + frame_name + '.png', orig_img)

    return result
    

def apply_feathering(mask):
    gradient = gaussian_gradient_magnitude(mask.astype(float), sigma=5)
    feathered_mask = 1 - gradient / np.max(gradient)
    return feathered_mask


def store_exif_info(image_path, theta, phi):
    # load Exif data of images
    exif_data = piexif.load(image_path)

    # Convert phi and theta to string
    phi_str = "{:.15f}".format(phi)
    theta_str = "{:.15f}".format(theta)

    # Store phi and theta as a custom user comment
    user_comment = f"phi={phi_str}, theta={theta_str}"
    exif_data["Exif"][piexif.ExifIFD.UserComment] = user_comment.encode("utf-8")

    # Encode and Save the EXIF data
    exif_bytes = piexif.dump(exif_data)
    piexif.insert(exif_bytes, image_path)

def load_exif_info(image_path):
    exif_data = piexif.load(image_path)

    # Retrieve the user comment from the EXIF data
    user_comment = exif_data["Exif"].get(piexif.ExifIFD.UserComment)

    # Extract the phi and theta values from the user comment
    if user_comment:
        user_comment = user_comment.decode("utf-8")
        parts = user_comment.split(", ")
        phi = float(parts[0].split("=")[1])
        theta = float(parts[1].split("=")[1])

        return theta, phi