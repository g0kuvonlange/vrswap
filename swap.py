import os
import time
import glob
import subprocess
import argparse
import sys
from PIL import Image as PILImage
import subprocess
import shutil
import torch
import insightface
import core.globals
from pathlib import Path
from core.swapper import get_face_swapper
from core.analyser import get_face, get_faces, get_face_analyser
from threading import Thread
import threading
import cv2
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
from core.lib import Equirec2Perspec as E2P, Perspec2Equirec as P2E
from pathlib import Path
from math import pi
import numpy as np
import cupy as cp
from scipy.ndimage import gaussian_gradient_magnitude

from numba import cuda
import json

if 'ROCMExecutionProvider' in core.globals.providers:
    del torch

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--frames_folder", help="Frames folder")
parser.add_argument("--face", help="Source Face")
parser.add_argument("--gpu_threads", help="Threads", default=5, type=int)
parser.add_argument('--gpu', help='use gpu', dest='gpu', action='store_true', default=False)
args = parser.parse_args()

framesFolder = args.frames_folder
sourceFace = args.face
gpuThreads = args.gpu_threads

# Create a lock for thread-safe file writing
lock = threading.Lock()

sep = "/"
if os.name == "nt":
    sep = "\\"

def resetDevice():
    device = cuda.get_current_device()
    device.reset()


def pre_check():
    if sys.version_info < (3, 9):
        quit('Python version is not supported - please upgrade to 3.9 or higher')
    if not shutil.which('ffmpeg'):
        quit('ffmpeg is not installed!')
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'inswapper_128.onnx')
    if not os.path.isfile(model_path):
        quit('File "inswapper_128.onnx" does not exist!')
    if '--gpu' in sys.argv:
        NVIDIA_PROVIDERS = ['CUDAExecutionProvider', 'TensorrtExecutionProvider']
        if len(list(set(core.globals.providers) - set(NVIDIA_PROVIDERS))) == 1:
            CUDA_VERSION = torch.version.cuda
            CUDNN_VERSION = torch.backends.cudnn.version()
            if not torch.cuda.is_available() or not CUDA_VERSION:
                quit("You are using --gpu flag but CUDA isn't available or properly installed on your system.")
            if CUDA_VERSION > '11.8':
                quit(f"CUDA version {CUDA_VERSION} is not supported - please downgrade to 11.8")
            if CUDA_VERSION < '11.4':
                quit(f"CUDA version {CUDA_VERSION} is not supported - please upgrade to 11.8")
            if CUDNN_VERSION < 8220:
                quit(f"CUDNN version {CUDNN_VERSION} is not supported - please upgrade to 8.9.1")
            if CUDNN_VERSION > 8910:
                quit(f"CUDNN version {CUDNN_VERSION} is not supported - please downgrade to 8.9.1")
    else:
        core.globals.providers = ['CPUExecutionProvider']


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


def face_analyser_thread(frame_path, source_face, vr = True):
    yes_face = True
    result = None  # Initialize result
    
    # Load the frame
    frame = cv2.imread(frame_path)
    
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]  # 000001
    frame_folder = os.path.splitext(frame_path)[0]                  # D:/test/000001
    output_folder = os.path.dirname(frame_path)                     # D:/test
    processing_folder = output_folder + "/processing"
    
    result = equir2pers(frame_path, processing_folder)

    left = f"{processing_folder}/{frame_name}_L.jpg"
    right = f"{processing_folder}/{frame_name}_R.jpg"

    img1 = perform_face_swap(left, source_face)
    img2 = perform_face_swap(right, source_face)

    return yes_face, result


def process_frames(source_img, frame_paths):
    frames_path = os.path.dirname(frame_paths[0])
    processing_path = os.path.dirname(frame_paths[0]) + "/processing"
    start_frame = os.path.splitext(os.path.basename(frame_paths[0]))[0]

    global face_analyser, swap
    swap = get_face_swapper()
    face_analyser = get_face_analyser()
    source_face = get_face(cv2.imread(source_img))

    # Create folder [frame_path]
    if not os.path.exists(processing_path):
        os.mkdir(processing_path)

    temp = []
    frame_counter = 0
    with tqdm(total=len(frame_paths), desc='Processing', unit="frame", dynamic_ncols=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as progress:
        for frame_path in frame_paths:
            frame_name = os.path.splitext(os.path.basename(frame_path))[0]
            output_folder = os.path.dirname(frame_path)
            processing_folder = output_folder + "/processing"

            left_jpg = f"{processing_folder}/{frame_name}_L.jpg"
            right_jpg = f"{processing_folder}/{frame_name}_R.jpg"
            left_png = f"{processing_folder}/{frame_name}_L.png"
            right_png = f"{processing_folder}/{frame_name}_R.png"

            left_exists = os.path.exists(left_jpg) or os.path.exists(left_png)
            right_exists = os.path.exists(right_jpg) or os.path.exists(right_png)

            if left_exists and right_exists:
                print("Both left and right files exist.")
                progress.set_postfix(status='S', refresh=True)
                progress.update(1)
            else:
                while len(temp) >= int(gpuThreads):
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



def perform_face_swap(frame_path, source_face):
    face_exists = os.path.exists(frame_path)

    if not face_exists:
        print("Face doesn't exist, skip")
        return

    frame = cv2.imread(frame_path) 

    target_faces = get_faces(frame)
    swapped_frame = frame

    if target_faces:
        for target_face in target_faces:
            # Perform face swapping on the frame using source_face and target_face
            swapped_frame = swap.get(frame, target_face, source_face, paste_back=True)
            cv2.imwrite(frame_path, swapped_frame)
    return swapped_frame



def extractFace(frame_name, input_img, face, output_dir, side):
    bbox = face.bbox

    # Load equirectangular image
    equ = E2P.Equirectangular(input_img)   

    # Convert bounding box to ints
    x1, y1, x2, y2 = map(int, bbox)

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
    output_path = os.path.join(output_dir, f'{frame_name}_{side}.jpg')
    cv2.imwrite(output_path, img)
    #store_exif_info(output_path, theta, phi)
    storeInfo(frame_name, side, output_dir, theta, phi)


def storeInfo(frame_name, side, output_dir, theta, phi):
    exif_data = {
        f'theta{side}': str(theta),
        f'phi{side}': str(phi)
    }

    parent_dir = os.path.dirname(output_dir)
    data_file = os.path.join(parent_dir, '_data.json')

    with lock:
        data = {}
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                data = json.load(f)
        
        if frame_name in data:
            data[frame_name].update(exif_data)
        else:
            data[frame_name] = exif_data
        
        with open(data_file, 'w') as f:
            json.dump(data, f)

def loadInfo(frame_number, output_dir, side):
    parent_dir = os.path.dirname(output_dir)
    data_file = os.path.join(parent_dir, '_data.json')

    with open(data_file, 'r') as f:
        data = json.load(f)

    if frame_number in data:
        theta = float(data[frame_number][f'theta{side}'])
        phi = float(data[frame_number][f'phi{side}'])
        return theta, phi
    else:
        raise ValueError(f"Frame number {frame_number} not found in {data_file}")



def equir2pers(input_img, output_dir):    
    frame_name = os.path.splitext(os.path.basename(input_img))[0]

    img = cv2.imread(input_img)
    faces = get_faces(img)  # Notice it's get_faces, assuming you're using a method that gets all faces.

    width = img.shape[1]

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        x_center = (x1 + x2) / 2

        if x_center < width / 2:
            extractFace(frame_name, input_img, face, output_dir, "L") 
        else:
            extractFace(frame_name, input_img, face, output_dir, "R") 


if __name__ == '__main__':
    pre_check()

    processingPath = framesFolder + "/processing"

    framePaths = []
    for framePath in glob.glob(framesFolder + "/*.jpg"):
        if not framePath.endswith('_p.jpg'):
            framePaths.append(framePath)

    framePaths = tuple(sorted(framePaths, key=lambda x: int(x.split(sep)[-1].replace(".jpg", ""))))

    print("swapping in progress...")
    process_frames(sourceFace, framePaths)