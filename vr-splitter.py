#!/usr/bin/env python
# pylint: disable=E501
import os
import glob
import argparse
import signal
import sys
import shutil
import torch
import time
import core.globals
#from core.swapper import get_face_swapper
from core.analyser import get_face, get_faces, get_face_analyser
from threading import Thread
import threading
import cv2
from tqdm import tqdm
import numpy as np
import sphere_snap.utils as snap_utils
import sphere_snap.sphere_coor_projections as sphere_proj
from sphere_snap.snap_config import SnapConfig, ImageProjectionType
from sphere_snap.sphere_snap import SphereSnap
from scipy.spatial.transform import Rotation as R
from numba import cuda
import json
from recognition.arcface_onnx import ArcFaceONNX
import logging

logging.basicConfig(level=logging.INFO)

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
    if not shutil.which('ffmpeg'):
        logging.error('ffmpeg is not installed!')
        sys.exit(1)
    #model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'inswapper_128.onnx')
    #if not os.path.isfile(model_path):
    #    logging.error('File "inswapper_128.onnx" does not exist!')
    #    sys.exit(1)
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

    def join(self, *args):
        Thread.join(self, *args)
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

    results_l = process_frame_side(frame_l, frame_name, processing_folder, "L")
    results_r = process_frame_side(frame_r, frame_name, processing_folder, "R")

    logging.debug(results_l)
    logging.debug(results_r)

    # left = f"{processing_folder}/{frame_name}_L_0.jpg"
    # right = f"{processing_folder}/{frame_name}_R_0.jpg"

    # img1 = perform_face_swap(left, source_face)
    # img2 = perform_face_swap(right, source_face)

    return yes_face, result


def process_vr_frames(frame_paths):

    global face_analyser, swap, rec, facelist, framedata, framedatafile_path, processing_path, continue_processing
    processing_path = os.path.dirname(frame_paths[0]) + "/processing"
    continue_processing = True
    # swap = get_face_swapper()
    face_analyser = get_face_analyser()
    # source_face = get_face(cv2.imread(source_img))
    rec = ArcFaceONNX('w600k_r50.onnx')
    rec.prepare(0)
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

#def perform_face_swap(frame_path, source_face):
#    face_exists = os.path.exists(frame_path)
#
#    if not face_exists:
#        logging.debug("Face doesn't exist, skip")
#        return
#    else:
#        logging.debug(f"Swapping {frame_path}")
#
#    frame = cv2.imread(frame_path)
#
#    target_faces = get_faces(frame)
#    swapped_frame = frame
#
#    if target_faces:
#        for target_face in target_faces:
#            # Perform face swapping on the frame using source_face and target_face
#            swapped_frame = swap.get(frame, target_face, source_face, paste_back=True)
#            cv2.imwrite(frame_path, swapped_frame)
#    else:
#        logging.debug(f"No face in {frame_path}")
#    return swapped_frame
#
def extractFace(frame_name, input_img, face, output_dir, side, frame_face_index, video_face_index):
    bbox = face.bbox
    x1, y1, x2, y2 = map(int, bbox)
    if x1 < 0 or x2 < 0 or y1 < 0 or y2 < 0:
        logging.debug(f"frame {frame_name} side {side} face {face.index} gender {face.gender} age {face.age} score {face.det_score} bbox {face.bbox} x1 {x1} x2 {x2} y1 {y1} y2 {y2} - NEGATIVE BBOX")
        return

    # Determine the center of the bounding box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    # Convert equirectangular coordinates to spherical
    height, width = input_img.shape[:2]
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
    logging.debug(f"frame {frame_name} side {side} face {face.index} gender {face.gender} age {face.age} score {face.det_score} frame_face {frame_face_index} bbox {face.bbox} x1 {x1} x2 {x2} y1 {y1} y2 {y2} yaw {yaw} pitch {pitch} phi {phi} theta {theta} h {height} w {width} crop fov {crop_fov} crop_length {crop_length} adjusted_fov {adjusted_fov}")

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

    # Actually swap the iamge
    # swapped_frame = swap.get(persp_img, face, source_face, paste_back=True)

    output_path = os.path.join(output_dir, f'{frame_name}_{side}_{frame_face_index}_{video_face_index}.jpg')
    cv2.imwrite(output_path, persp_img)

    # bbox_crop_img = input_img[y1:y2, x1:x2]
    # bbox_output_path = os.path.join(output_dir, f'{frame_name}_{side}_{frame_face_index}_bbox.jpg')
    # cv2.imwrite(bbox_output_path, bbox_crop_img)

    update_frame_data(frame_name, side, frame_face_index, video_face_index, output_dir, theta, phi, crop_fov)  # Assuming you want the horizontal fov
    return True

def update_frame_data(frame_name, side, frame_face_index, video_face_index, output_dir, theta, phi, fov):
    global framedata, processing_path

    # add to global framedata dict
    frame_metadata = {'frames':
                      {f'{frame_name}':
                       {f'{side}':
                        {f'{frame_face_index}':
                         {'theta': str(theta),
                          'phi': str(phi),
                          'fov': str(fov),
                          'video_face_index': str(video_face_index)
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


def process_frame_side(img, frame_name, output_dir, side):
    global framedata

    faces = get_faces(img)  # Notice it's get_faces, assuming you're using a method that gets all faces.

    facecount = len(faces)
    logging.debug(f'FOUND {facecount} FACES')
    for frame_face_index, face in enumerate(faces):
        if face and face.det_score > 0.55:
            # compare face to facelist to determine where to store
            # TODO: store in json and load on resume
            if len(facelist) < 1:
                # first face to add.
                logging.info('No faces seen yet, adding first face to facelist')
                video_face_index = 1
                facelist.append([1, face.embedding])
                # create the face index subdir
                face_dir = output_dir + '/' + 'face_' + str(video_face_index)
                if not os.path.exists(face_dir):
                    os.mkdir(face_dir)
                # update framedata
                framedata['faces'].append(video_face_index)
                # store frame data
                store_frame_data()
            else:
                known = False
                for known_face in facelist:
                    sim = rec.compute_sim(known_face[1], face.embedding)
                    if sim >= similarity:
                        video_face_index = known_face[0]
                        known = True
                        logging.debug(f'Existing face detected {video_face_index} - sim {sim}')
                if not known:
                    video_face_index = len(facelist) + 1
                    logging.info(f'NEW face detected, creating subdir for video_face_index {video_face_index}')
                    # create the face index subdir
                    face_dir = output_dir + '/' + 'face_' + str(video_face_index)
                    if not os.path.exists(face_dir):
                        os.mkdir(face_dir)
                    framedata['faces'].append(video_face_index)
                    facelist.append([video_face_index, face.embedding])
                    # store frame data
                    store_frame_data()

            extract_dir = output_dir + '/' + 'face_' + str(video_face_index)
            extractFace(frame_name, img, face, extract_dir, side, frame_face_index, video_face_index)
        elif face.det_score <= 0.55:
            logging.debug(f"frame {frame_name} face {frame_face_index} gender {face.gender} score {face.det_score} SKIP - FACE DETECTION SCORE TOO LOW")


# Add a global variable to count the number of times Ctrl+C is pressed
ctrl_c_counter = 0

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
    parser.add_argument("--similarity", help="Value between 0.0 (hardly the same)) and 1.0 (exactly the same) to group similar faces by", default=0.2, type=float)
    parser.add_argument("--gpu_threads", help="Threads", default=10, type=int)
    parser.add_argument('--gpu', help='use gpu', dest='gpu', action='store_true', default=False)
    args = parser.parse_args()

    framesFolder = args.frames_folder
    similarity = args.similarity
    gpuThreads = args.gpu_threads

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
