import time
import os
import cv2
import sys
import numpy as np
import cupy as cp
from scipy.ndimage import gaussian_gradient_magnitude
import threading
import core.lib.Perspec2Equirec as P2E
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
import subprocess
import json

from tqdm import tqdm
from threading import Thread
from numba import cuda

import shutil
import glob

FRAME_CHUNK_SIZE = 500
GPU_THREADS = 9

sep = "/"
if os.name == "nt":
    sep = "\\"

def resetDevice():
    device = cuda.get_current_device()
    device.reset()

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--frames_folder", help="Frames folder")
args = parser.parse_args()

frames_folder = args.frames_folder

thread_locks = {}

def get_lock(frame_path):
    global thread_locks

    lock = thread_locks.get(frame_path)

    if lock is None:
        lock = threading.Lock()
        thread_locks[frame_path] = lock

    return lock

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def convert_thread(frame_path):
    # Get lock for the current frame
    lock = get_lock(frame_path)

    with lock:
        # The code inside this block is now thread-safe
        max_retries = 3
        retry_delay = 1  # seconds

        for retry in range(max_retries):
            try:
                frame_name = os.path.splitext(os.path.basename(frame_path))[0]
                output_folder = os.path.dirname(frame_path)
                processing_folder = output_folder + "/processing"

                persp2equir(frame_path, processing_folder)

                shutil.move(processing_folder + '/' + frame_name + '.jpg', output_folder + "/" + frame_name + "_p.jpg")
                if os.path.exists(processing_folder + '/' + frame_name + '_L.png'):
                    os.remove(processing_folder + '/' + frame_name + '_L.png')
                
                if os.path.exists(processing_folder + '/' + frame_name + '_R.png'):
                    os.remove(processing_folder + '/' + frame_name + '_R.png')
                
                # delete original frame
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                
                #os.rename(output_folder + "/" + frame_name + "_p.jpg", output_folder + "/" + frame_name + ".jpg")

                return True  # Success, no error message
            except Exception as e:
                print(f"Failed to convert frame: {e}")
                if retry < max_retries - 1:
                    print("Retrying...")
                    time.sleep(retry_delay)
                else:
                    print("Exceeded maximum retries. Adding frame_path to error-log.txt.")
                    write_error_log(frame_path)
        
    return False  # Success, no error message


def convert(frame_paths, gpu_threads):
    frames_path = os.path.dirname(frame_paths[0])
    processing_path = os.path.dirname(frame_paths[0]) + "/processing"

    temp = []
    with tqdm(total=len(frame_paths), desc='Processing', unit="frame", dynamic_ncols=True,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as progress:
        for frame_path in frame_paths:
            #convert_thread(frame_path)
            while len(temp) >= gpu_threads:
                has_result = temp.pop(0).join()

                if has_result:
                    progress.set_postfix(status='.', refresh=True)
                else:
                    progress.set_postfix(status='S', refresh=True)
                progress.update(1)

            time.sleep(1)
            temp.append(ThreadWithReturnValue(target=convert_thread, args=(frame_path,)))
            temp[-1].start()

        # Wait for all threads to finish
        while len(temp) > 0:
            has_result = temp.pop(0).join()

            if has_result:
                progress.set_postfix(status='.', refresh=True)
            else:
                progress.set_postfix(status='S', refresh=True)
            progress.update(1)


def persp2equir_single(input_img, theta, phi, orig_height, orig_width):
    persp = P2E.Perspective(input_img, 90, theta, phi)
    img, _ = persp.GetEquirec(orig_height, orig_width)

    # Compute masks and apply feathering
    mask = cp.sum(img, axis=2) > 0
    feathered_mask = apply_feathering(mask)

    return img, feathered_mask


def persp2equir(frame_path, input_folder):
    frame_number = os.path.splitext(os.path.basename(frame_path))[0]
    output_dir = os.path.dirname(frame_path)

    orig_img = load_frame(frame_path)
    orig_height, orig_width = orig_img.shape[:2]

    frame_name = os.path.splitext(os.path.basename(frame_path))[0]

    input_img1 = input_folder + '/' + frame_name + '_L.png'
    input_img2 = input_folder + '/' + frame_name + '_R.png'

    img1 = cp.zeros((orig_height, orig_width, 3), dtype=cp.uint8)
    img2 = cp.zeros((orig_height, orig_width, 3), dtype=cp.uint8)

    # Create a ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=2)

    # Submit tasks to executor
    if os.path.exists(input_img1):
        theta1, phi1 = loadInfo(frame_number, output_dir, "L")
        fut1 = executor.submit(persp2equir_single, input_img1, theta1, phi1, orig_height, orig_width)
    else:
        fut1 = None

    # Submit tasks to executor
    if os.path.exists(input_img2):
        theta2, phi2 = loadInfo(frame_number, output_dir, "R")
        fut2 = executor.submit(persp2equir_single, input_img2, theta2, phi2, orig_height, orig_width)
    else:
        fut2 = None

    # Gather results
    img1, feathered_mask1 = fut1.result() if fut1 else (img1, None)
    img2, feathered_mask2 = fut2.result() if fut2 else (img2, None)

    executor.shutdown(wait=True)

    half_width = orig_width // 2
    img1[:, half_width:] = 0
    img2[:, :half_width] = 0

    if fut1 and cp.sum(img1) > 0:
        # Create composite images by blending the feathered perspectives with the original image
        composite1 = orig_img * (1 - feathered_mask1[..., cp.newaxis]) + img1 * feathered_mask1[..., cp.newaxis]

        # Replace pixels in the original image with those from the transformed perspectives respecting the feathered masks
        mask1 = cp.sum(img1, axis=2) > 0
        orig_img[mask1] = composite1[mask1]

    if fut2 and cp.sum(img2) > 0:
        # Create composite images by blending the feathered perspectives with the original image
        composite2 = orig_img * (1 - feathered_mask2[..., cp.newaxis]) + img2 * feathered_mask2[..., cp.newaxis]

        # Replace pixels in the original image with those from the transformed perspectives respecting the feathered masks
        mask2 = cp.sum(img2, axis=2) > 0
        orig_img[mask2] = composite2[mask2]

    # Save the result
    result = cv2.imwrite(input_folder + '/' + frame_name + '.jpg', orig_img)

    return result


def apply_feathering(mask):
    gradient_x = cv2.Sobel(mask.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(mask.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    feathered_mask = 1 - gradient_magnitude / cp.amax(gradient_magnitude)
    return feathered_mask


# Create a lock for thread-safe file writing
lock = threading.Lock()


def loadInfo(frame_number, output_dir, side):
    data_file = os.path.join(output_dir, '_data.json')

    with open(data_file, 'r') as f:
        data = json.load(f)

    if frame_number in data:
        theta = float(data[frame_number][f'theta{side}'])
        phi = float(data[frame_number][f'phi{side}'])
        return theta, phi
    else:
        raise ValueError(f"Frame number {frame_number} not found in {data_file}")


def load_frame(frame_path):
    # Read the frame from the SBS VR video
    frame = cv2.imread(frame_path)

    return frame


def write_error_log(frame_path):
    with open("error-log.txt", "a") as file:
        file.write(frame_path + "\n")

def relaunch():
    resetDevice()
    os.execv(sys.executable, ['python'] + sys.argv)

def get_frame_number(frame_path):
    frame_name = os.path.basename(frame_path)
    frame_number = int(re.search(r'\d+', frame_name).group())
    return frame_number

if __name__ == '__main__':
    frame_paths = [path for path in glob.glob(frames_folder + "/*.jpg") if not path.endswith("_p.jpg")]
    frame_paths.sort(key=lambda x: get_frame_number(x))

    filtered_frame_paths = []

    for frame_path in frame_paths:
        frame_name = os.path.basename(frame_path)

        # Check for the presence of corresponding L and R files
        l_path = os.path.join(frames_folder, 'processing', frame_name.replace('.jpg', '_L.png'))
        r_path = os.path.join(frames_folder, 'processing', frame_name.replace('.jpg', '_R.png'))

        if os.path.exists(l_path) or os.path.exists(r_path):
            frame_number = get_frame_number(frame_name)
            filtered_frame_paths.append((frame_number, frame_path))
        else:
            os.rename(frame_path, os.path.join(frames_folder, frame_name + "_p.jpg"))

        if len(filtered_frame_paths) == FRAME_CHUNK_SIZE:
            break

    # Sorting the frames based on their frame_number
    filtered_frame_paths.sort(key=lambda x: x[0])

    # Extracting the frame_paths from the filtered_frame_paths list
    frame_paths = [frame_path for _, frame_path in filtered_frame_paths]

    convert(frame_paths, GPU_THREADS)
    print("Conversion successful!")
    relaunch()

