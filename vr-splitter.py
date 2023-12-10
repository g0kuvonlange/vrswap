#!/usr/bin/env python
# pylint: disable=E501
import os
import glob
import argparse
import signal
import sys
from threading import Thread
import threading
import time
import logging
import json
import torch
import cv2
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
from numba import cuda
import core.globals
from core.analyser import get_faces
import sphere_snap.utils as snap_utils
import sphere_snap.sphere_coor_projections as sphere_proj
from sphere_snap.snap_config import SnapConfig, ImageProjectionType
from sphere_snap.sphere_snap import SphereSnap
from recognition.arcface_onnx import ArcFaceONNX

# set basic logging level
logging.basicConfig(level=logging.INFO, force=True)

# Add a global variable to count the number of times Ctrl+C is pressed
ctrl_c_counter = 0

try:
    from pydantic.utils import deep_update
except Exception:
    from pydantic.v1.utils import deep_update

if 'ROCMExecutionProvider' in core.globals.providers:
    del torch


def resetDevice():
    device = cuda.get_current_device()
    device.reset()


def pre_check():
    if sys.version_info < (3, 9):
        logging.error('Python version is not supported - please upgrade to 3.9 or higher')
        sys.exit(1)
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'w600k_r50.onnx')
    if not os.path.isfile(model_path):
        logging.error('File "w600k_r50.onnx" does not exist!')
        sys.exit(1)
    if '--gpu' in sys.argv:
        NVIDIA_PROVIDERS = ['CUDAExecutionProvider', 'TensorrtExecutionProvider']
        if len(list(set(core.globals.providers) - set(NVIDIA_PROVIDERS))) == 1:
            CUDA_VERSION = torch.version.cuda
            CUDNN_VERSION = torch.backends.cudnn.version()
            if not torch.cuda.is_available() or not CUDA_VERSION:
                logging.error("You are using --gpu flag but CUDA isn't available or properly installed on your system.")
                sys.exit(1)
            if CUDA_VERSION > '11.8':
                logging.error(f"CUDA version {CUDA_VERSION} is not supported - please downgrade to 11.8")
                sys.exit(1)
            if CUDA_VERSION < '11.4':
                logging.error(f"CUDA version {CUDA_VERSION} is not supported - please upgrade to 11.8")
                sys.exit(1)
            if CUDNN_VERSION < 8220:
                logging.error(f"CUDNN version {CUDNN_VERSION} is not supported - please upgrade to 8.9.1")
                sys.exit(1)
            if CUDNN_VERSION > 8910:
                logging.error(f"CUDNN version {CUDNN_VERSION} is not supported - please downgrade to 8.9.1")
                sys.exit(1)
    else:
        core.globals.providers = ['CPUExecutionProvider']


# Global list to keep track of all threads
all_threads = []


# creates a thread and returns value when joined
class ThreadWithReturnValue(Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, timeout=None):
        Thread.join(self, timeout)
        return self._return


def process_frame_thread(frame_path, foo):
    yes_face = True
    result = None  # Initialize result

    # Load the frame
    frame = cv2.imread(frame_path)

    # Split in L and R framesa
    width = frame.shape[1]
    half_width = width // 2
    frame_l = frame[:, :half_width]
    frame_r = frame[:, half_width:]

    frame_name = os.path.splitext(os.path.basename(frame_path))[0]  # 000001
    output_folder = os.path.dirname(frame_path)                     # D:/test
    processing_folder = output_folder + "/processing"

    process_frame_side(frame_l, frame_name, processing_folder, "L")
    process_frame_side(frame_r, frame_name, processing_folder, "R")

    return yes_face, result


def process_vr_frames(frame_paths):

    global vr_det_thresh, persp_det_thresh, recognizer, facelist, framedata, framedatafile_path, processing_path, continue_processing
    processing_path = os.path.dirname(frame_paths[0]) + "/processing"
    continue_processing = True
    recognizer = ArcFaceONNX('w600k_r50.onnx')
    recognizer.prepare(0)
    facelist = []
    framecount = 0

    # Create folder [frame_path]
    if not os.path.exists(processing_path):
        os.mkdir(processing_path)

    framedatafile_path = os.path.join(processing_path, '_data.json')
    if os.path.exists(framedatafile_path):
        with lock:
            with open(framedatafile_path, 'r') as f:
                framedata = json.load(f)
            # load each npy embedding for the faces
            for face in framedata['faces']:
                facelist.append([face, np.load(processing_path + "/_data_face_" + str(face) + ".npy")])
            framecount = len(framedata['frames'].keys())
            facecount = len(framedata['faces'])
            logging.info(f"Data files detected, loaded {framecount} frames and {facecount} faces from previous run")

    else:
        framedata = {'frames': {}, 'faces': []}

    temp = []
    with tqdm(initial=framecount, total=len(frame_paths), desc='Processing', unit="frame", mininterval=1.0, smoothing=0.1, dynamic_ncols=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as progress:
        save_time = time.time()
        for i, frame_path in enumerate(frame_paths):
            frame_name = os.path.splitext(os.path.basename(frame_path))[0]

            if frame_name not in framedata['frames'].keys():
                while len(temp) >= int(gpuThreads):
                    # we are order dependent, so we are forced to wait for first element to finish. When finished removing thread from the list
                    has_face, x = temp.pop(0).join()

                    if has_face:
                        progress.set_postfix(status='processing', refresh=True)
                    else:
                        progress.set_postfix(status='skipping', refresh=True)
                    progress.update(1)
                if continue_processing:
                    # adding new frame to the list and starting it
                    thread = ThreadWithReturnValue(target=process_frame_thread, args=(frame_path, None))
                    temp.append(thread)
                    all_threads.append(thread)
                    thread.start()

            # Check if 30 seconds have passed
            if time.time() - save_time >= 30:
                logging.debug("Periodically saving data...")
                store_frame_data()
                save_time = time.time()


def findFace(frame_name, input_img, side, phi, theta):
    # Convert equirectangular coordinates to spherical
    height, width = input_img.shape[:2]
    # Calculate the FOV based on the size of the bounding box
    crop_width = width // 3  # Example crop width, adjust as needed
    crop_height = height // 3  # Example crop height, adjust as needed
    # make sure they're always multiples of 16
    crop_width = (crop_width + 15) // 16 * 16
    crop_height = (crop_height + 15) // 16 * 16
    # and make sure it's smaller than the image itself
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)
    # Calculate the FOV based on the size of the bounding box
    fov_x = (crop_width / width) * 180
    fov_y = (crop_height / height) * 90
    # get the largest of the 2 FOVs and lengths
    crop_fov = max(fov_x, fov_y)
    crop_length = max(crop_width, crop_height)
    yaw = np.degrees(phi)
    pitch = np.degrees(theta)
    # Output dimensions
    out_hw = (crop_length, crop_length)
    rotation_quat = R.from_euler("yxz", [-yaw, pitch, 0], degrees=True).as_quat()
    adjusted_fov = snap_utils.ensure_fov_res_consistency((crop_fov, crop_fov), (crop_length, crop_length))
    snap_config = SnapConfig(
        orientation_quat=rotation_quat,
        out_hw=out_hw,
        out_fov_deg=adjusted_fov,
        source_img_hw=(height, width),
        source_img_fov_deg=(180, 180),
        source_img_type=ImageProjectionType.HALF_EQUI
    )
    sphere_snap_obj = SphereSnap(snap_config)
    persp_img = sphere_snap_obj.snap_to_perspective(input_img)
    return persp_img, adjusted_fov


def extractFace(input_img, face, phi=None, theta=None, crop_fov=None):
    height, width = input_img.shape[:2]
    bbox = face.bbox
    x1, y1, x2, y2 = map(int, bbox)
    if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
        # TODO: find out why this happens; maybe adjust for it (invert negative vale, add up to other value on same axis
        logging.debug(f"NEGATIVE BBOX, ADJUSTING - x1 {x1} x2 {x2} y1 {y1} y2 {y2}")
        if x1 < 0:
            x2 = x2 + (x1 * -1)
            x1 = 0
        elif x2 > width:  # eg x2 = 2022, width= 2000
            x1 = x1 - (x2 - width)
            x2 = width
        if y1 < 0:
            y2 = y2 + (y1 * -1)
            y1 = 0
        elif y2 > height:  # eg y2 = 2022, height = 2000
            y1 = y1 - (y2 - height)
            y2 = height
        if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
            logging.debug(f"NEGATIVE BBOX AFTER ADJUSTMENTS, CANNOT EXTRACT - x1 {x1} x2 {x2} y1 {y1} y2 {y2}")
            return None, None, None, None
        else:
            logging.debug(f"NEGATIVE BBOX, ADJUSTED - x1 {x1} x2 {x2} y1 {y1} y2 {y2}")

    # Determine the center of the bounding box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Convert equirectangular coordinates to spherical
    if not phi or not theta:
        phi, theta = sphere_proj.halfequi_coor2spherical(np.array([x_center, y_center]), (width, height))

    # use crop sizes at 200% of bounding box
    crop_width = (x2 - x1) * 2
    crop_height = (y2 - y1) * 2

    # make sure faces are always at least 256x256
    crop_width = max(crop_width, 256)
    crop_height = max(crop_height, 256)

    # make sure they're always multiples of 16
    crop_width = (crop_width + 15) // 16 * 16
    crop_height = (crop_height + 15) // 16 * 16

    # and make sure it's smaller than the image itself
    crop_width = min(crop_width, width)
    crop_height = min(crop_height, height)

    # Calculate the FOV based on the size of the bounding box
    fov_x = (crop_width / width) * 180
    fov_y = (crop_height / height) * 90

    # get the largest of the 2 FOVs and lengths
    crop_fov = max(fov_x, fov_y)
    crop_length = max(crop_width, crop_height)

    yaw = np.degrees(phi)
    pitch = np.degrees(theta)

    # Output dimensions
    out_hw = (crop_length, crop_length)

    rotation_quat = R.from_euler("yxz", [-yaw, pitch, 0], degrees=True).as_quat()
    adjusted_fov = snap_utils.ensure_fov_res_consistency((crop_fov, crop_fov), (crop_length, crop_length))

    snap_config = SnapConfig(
        orientation_quat=rotation_quat,
        out_hw=out_hw,
        out_fov_deg=adjusted_fov,
        source_img_hw=(height, width),
        source_img_fov_deg=(180, 180),
        source_img_type=ImageProjectionType.HALF_EQUI
    )

    sphere_snap_obj = SphereSnap(snap_config)
    persp_img = sphere_snap_obj.snap_to_perspective(input_img)

    return persp_img, theta, phi, adjusted_fov


def update_frame_data(frame_name, side, vr_frame_face_index, persp_frame_face_index, output_dir, theta, phi, fov):
    global framedata, processing_path

    # add to global framedata dict
    frame_metadata = {'frames':
                      {f'{frame_name}':
                       {f'{side}':
                        {f'{vr_frame_face_index}':
                         {f'{persp_frame_face_index}':
                          {'theta': str(theta),
                           'phi': str(phi),
                           'fov': str(fov),
                           }
                          }
                         }
                        }
                       }
                      }

    framedata = deep_update(framedata, frame_metadata)


def store_frame_data():
    global lock, framedatafile_path, framedata, facelist

    logging.debug("Grabbing write lock")
    with lock:
        with open(framedatafile_path, 'w') as f:
            json.dump(framedata, f, indent=2, sort_keys=True)
            logging.debug(f"JSON Dumped data to {framedatafile_path}")
        for face in facelist:
            if not os.path.exists(processing_path + "/_data_face_" + str(face) + ".npy"):
                np.save(processing_path + "/_data_face_" + str(face[0]) + ".npy", face[1])


def get_known_face_dir(face, output_dir):
    global facelist, framedata, similarity

    # For each face seen on a perspective frame, we check if we have seen it before
    # this is done by comparing the embedding using a similarity comparison
    # Each unique embedding gets a number, and each embedding is stored in a list in framedata
    # The perspective frames are stored in subdirs suffixed with that number.

    # if we already have one or more face embeddings in facelist, compare against them
    for known_face in facelist:
        sim = recognizer.compute_sim(known_face[1], face.embedding)
        if sim >= similarity:
            video_face_index = known_face[0]
            logging.debug(f'Existing face detected {video_face_index} - sim {sim}')
            face_dir = output_dir + '/' + 'face_' + str(video_face_index)
            return face_dir

    # If we are here, this is either the first face, or we have not seen this face before
    video_face_index = len(facelist) + 1
    logging.info(f'NEW face detected, creating subdir for video_face_index {video_face_index}')
    # create the face index subdir
    face_dir = output_dir + '/' + 'face_' + str(video_face_index)
    # can be racy so wrap in try
    if not os.path.exists(face_dir):
        try:
            os.mkdir(face_dir)
        except FileExistsError:
            True
    # append index number to seen faces list
    framedata['faces'].append(video_face_index)
    # append index + embedding to facelist
    facelist.append([video_face_index, face.embedding])
    # TODO: should we really store data here? could be slowing us down
    # store frame data
    return face_dir


def process_frame_side(img, frame_name, output_dir, side):
    global framedata, persp_det_thresh, vr_det_thresh

    # Overall workflow per single-eye equirect image:
    # 1. Run face detector on VR frame. Yields 0 or more faces depending on vr_det_thresh
    # 2. For each found face, create a perspective image
    # 3. On this perpective image, run get_face detection again. Yields 0 or more faces depending on persp_det_thresh
    # 4. Compare each persp_face to the faces prevously seen in facelist. If known, allocate an index and store in same folder.
    #    If unknown, store in new folder. Repeat for each face.
    # 5. If no vr_faces were found, run a blind perspective frame extraction and goto 3.
    #    Faces may not be detectable in extremely warped edges, so this a best-effort. TODO: improve this method.

    vr_faces = get_faces(vr_det_thresh, img)  # Notice it's get_faces, assuming you're using a method that gets all faces.

    vr_facecount = len(vr_faces)
    for vr_frame_face_index, vr_face in enumerate(vr_faces):

        # Fetch the perspective for this face in this side of the frame
        persp_img, theta, phi, crop_fov = extractFace(img, vr_face)
        if not theta:
            continue

        # now we have to rerun face detection because the VR frame might have yielded not-a-face
        persp_faces = get_faces(persp_det_thresh, persp_img)
        persp_facecount = len(persp_faces)

        # hopefully we have just one face here.
        for persp_frame_face_index, persp_face in enumerate(persp_faces):
            face_dir = get_known_face_dir(persp_face, output_dir)
            # File naming scheme broken down:
            # frame_name: the original video frame numbering filename
            # side: eye, either L or R
            # vr_frame_face_index: a numbered list of faces detected on one side of a VR frame
            # persp_frame_face_index: a numbered list of faces detected on a perspective frame, based on the location of a detected face on the VR frame.
            persp_image_output_path = os.path.join(face_dir, f'{frame_name}_{side}_{vr_frame_face_index}_{persp_frame_face_index}.jpg')
            cv2.imwrite(persp_image_output_path, persp_img)

            update_frame_data(frame_name, side, vr_frame_face_index, persp_frame_face_index, output_dir, theta, phi, crop_fov)  # Assuming you want the horizontal fov

        if persp_facecount < 1:
            logging.debug(f"frame {frame_name} VR FACE DETECTED BUT PERSP FACE NOT FOUND")

    if vr_facecount < 1:
        # FACE DETECTOR DETECTED NO FACE WHATSOEVER ON VR FRAME
        logging.debug(f"frame {frame_name} NO VR FACE DETECTED AT ALL, STILL RUNNING FACEFIND")
        # Set phi and theta to point down at a 45-degree angle to avoid the black area
        phi = 0  # Center along the horizontal axis
        theta = -np.pi / 3
        # For 60 degrees, use: theta = -np.pi / 3
        # For 90 degrees, use: theta = -np.pi / 2

        # Call findFace with the manually set phi and theta values
        persp_img, crop_fov = findFace(frame_name, img, side, phi, theta)

        persp_faces = get_faces(persp_det_thresh, persp_img)
        persp_facecount = len(persp_faces)

        # We set an arbitrary vr_frame_face_index, e.g. 99
        vr_frame_face_index = 99

        # hopefully we have just one face here.
        for persp_frame_face_index, persp_face in enumerate(persp_faces):
            face_dir = get_known_face_dir(persp_face, output_dir)
            # File naming scheme broken down:
            # frame_name: the original video frame numbering filename
            # side: eye, either L or R
            # vr_frame_face_index: a numbered list of faces detected on one side of a VR frame
            # persp_frame_face_index: a numbered list of faces detected on a perspective frame, based on the location of a detected face on the VR frame.
            persp_image_output_path = os.path.join(face_dir, f'{frame_name}_{side}_{vr_frame_face_index}_{persp_frame_face_index}.jpg')
            cv2.imwrite(persp_image_output_path, persp_img)

            update_frame_data(frame_name, side, vr_frame_face_index, persp_frame_face_index, output_dir, theta, phi, crop_fov)  # Assuming you want the horizontal fov


def signal_handler(sig, frame):
    global continue_processing, all_threads, ctrl_c_counter
    ctrl_c_counter += 1
    print('You pressed Ctrl+C! Saving state..')
    store_frame_data()
    if ctrl_c_counter > 1:
        print('Force quitting...')
        for thread in all_threads:
            try:
                # Terminate the thread
                thread._stop()
            except Exception:
                pass
        sys.exit(0)
    else:
        print('Gracefully exiting...')
        continue_processing = False
        # Wait for all threads to complete
        for thread in all_threads:
            while thread.is_alive():
                thread.join(timeout=1)
        store_frame_data()
        sys.exit(0)


if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_folder", help="Frames folder", required=True)
    parser.add_argument("--similarity", help="Value between 0.0 (hardly the same)) and 1.0 (exactly the same) to group similar faces by, default 0.15", default=0.15, type=float)
    parser.add_argument("--gpu_threads", help="Threads", default=10, type=int)
    parser.add_argument('--gpu', help='use gpu', dest='gpu', action='store_true', default=False)
    parser.add_argument('--vr-detection-threshold', help='Value between 0.0 (imprecise) to 1.0 (very precise) to detect a face on the warped VR image, default 0.3. WARNING: re-running with altered values can result in mismapped frames when merging!', dest='vr_det_thresh', default=0.3, type=float)
    parser.add_argument('--detection-threshold', help='Value between 0.0 (imprecise) to 1.0 (very precise) to detect a face on the normalized perspective image, default 0.55. WARNING: re-running with altered values can result in mismapped frames when merging!', dest='persp_det_thresh', default=0.55, type=float)
    args = parser.parse_args()

    framesFolder = args.frames_folder
    similarity = args.similarity
    gpuThreads = args.gpu_threads
    vr_det_thresh = args.vr_det_thresh
    persp_det_thresh = args.persp_det_thresh

    # Create a lock for thread-safe file writing
    lock = threading.Lock()

    sep = "/"
    if os.name == "nt":
        sep = "\\"

    pre_check()

    processingPath = framesFolder + "/processing"

    framePaths = []
    for framePath in glob.glob(framesFolder + "/*.jpg"):
        framePaths.append(framePath)

    framePaths = tuple(sorted(framePaths, key=lambda x: int(x.split(sep)[-1].replace(".jpg", ""))))

    # Set up signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)

    logging.info("splitting in progress...")
    process_vr_frames(framePaths)

    logging.info("completed; waiting for threads to finish...")
    # Wait for all threads to complete
    for thread in all_threads:
        thread.join()
    logging.info("Saving data...")
    store_frame_data()
