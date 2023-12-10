#!/usr/bin/env python
# pylint: disable=E501
import time
import os
import cv2
import sys
import signal
import numpy as np
import argparse
import threading
import re
import json
from pathlib import Path

# import sphere_snap.utils as snap_utils
# import sphere_snap.sphere_coor_projections as sphere_proj
from sphere_snap.snap_config import SnapConfig, ImageProjectionType
from sphere_snap.sphere_snap import SphereSnap
# import sphere_snap.reprojections as rpr
from scipy.spatial.transform import Rotation as R


from tqdm import tqdm
from threading import Thread
import logging

logging.basicConfig(level=logging.INFO, force=True)

try:
    from pydantic.utils import deep_update
except Exception:
    from pydantic.v1.utils import deep_update

all_threads = []


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


def merge_frame_thread(frame_path, foo):

    global framedata, gpu_threads
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]
    output_folder = os.path.dirname(frame_path)

    # Check again to make sure we have this frame in framedata; if not, something went wrong
    if frame_name not in framedata['frames']:
        logging.debug(f'Frame {frame_name} missing in framedata - should not have been queued')
        return False

    # next, if already done, skip it
    if 'done' in framedata['frames'][frame_name] and framedata['frames'][frame_name]['done']:
        logging.debug(f'Frame {frame_name} already done, skipping')
        return False

    # Load full L+R VR frame
    vr_frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    vr_height, vr_width = vr_frame.shape[:2]

    # Split in L and R framesa
    half_width = vr_width // 2

    # define modified frames per side
    modified_frame = {}
    modified_frame['L'] = None
    modified_frame['R'] = None
    logging.debug(f"{framedata['frames'][frame_name]}")

    # Process L and R separately in a loop
    for side in ['L', 'R']:
        # process only when this side exists in the frame data

        # but only do work on this side if any work needs doing
        if side not in framedata['frames'][frame_name]:
            logging.debug(f'No faces found for {side} on frame {frame_name}, skipping')
            continue
        else:
            faces_per_side = len(framedata['frames'][frame_name][side].keys())
            logging.debug(f'Processing {faces_per_side} faces for {side} on frame {frame_name}')

        # Split the original frame in the correct half
        if side == 'L':
            hequirect_frame = vr_frame[:, :half_width]
        elif side == 'R':
            hequirect_frame = vr_frame[:, half_width:]

        snaps = []
        snapimages = []
        snapimagemasks = []

        for vrface, vrdata in framedata['frames'][frame_name][side].items():
            for face, facedata in vrdata.items():
                logging.debug(f'Frame {frame_name} vrface {vrface} vrdata {vrdata} face {face} facedata {facedata}')
                # ensure we have a path for this face, or we just skip
                if 'path' not in facedata:
                    logging.debug(f'Frame {frame_name} side {side} face {face} has no file path, skipping')
                    continue
                # convert each perspective image back to a hequirect one using a spheresnap and the image phi, theta and dimensions
                # NOTE: faces are list not by index, but by face number. So face can be 1,2, or can be 2 or 2,4,8 etc.

                # TODO: float32?
                phi = float(facedata['phi'])
                theta = float(facedata['theta'])
                fov = eval(facedata['fov'])
                yaw = np.degrees(phi)
                pitch = np.degrees(theta)

                # define the rotation quat
                rotation_quat = R.from_euler("yxz", [-yaw, pitch, 0], degrees=True).as_quat()

                # read the image and append to images, but as float32 or spheresnap might fail
                image = cv2.imread(facedata['path'], cv2.IMREAD_COLOR)

                # set up the snap config for the perspective image
                perspective_image_config = SnapConfig(
                    orientation_quat=rotation_quat,
                    out_hw=image.shape[:2],
                    out_fov_deg=fov,
                    source_img_hw=(vr_height, half_width),
                    source_img_fov_deg=(180, 180),
                    source_img_type=ImageProjectionType.HALF_EQUI
                )

                perspective_image_snap = SphereSnap(perspective_image_config)

                # append the spheresnap to the snaps list
                snaps.append(perspective_image_snap)

                # apped the image to snapimages as float32
                snapimages.append(np.float32(image))

                # create a mask with all pixels set to 1
                mask = np.ones_like(image)

                snapimagemasks.append(mask)

        # ensure we have work - if len(snaps) < 1, skip.
        if len(snaps) > 0 and len(snapimages) > 0:
            # create the reconstructed hequirectangular image based on the snaps and snapimages
            reconstructed_hequi = SphereSnap.merge_multiple_snaps((vr_height, half_width),
                                                                  snaps, snapimages,
                                                                  target_type=ImageProjectionType.HALF_EQUI)
            # create a mask for the same
            # reconstructed_hequi_mask = SphereSnap.merge_multiple_snaps((vr_height, half_width),
            #                                                      snaps, snapimagemasks,
            #                                                      target_type=ImageProjectionType.HALF_EQUI)

            # Find the non-black pixels in the reconstructed_hequi
            non_black_pixels = np.where(np.any(reconstructed_hequi != [0, 0, 0], axis=-1))

            # Create a copy of hequirect_frame to modify
            hequirect_frame_modified = hequirect_frame.copy()

            # Fill in the non-black pixels
            hequirect_frame_modified[non_black_pixels] = reconstructed_hequi[non_black_pixels]

            # Update the modified frame per side
            modified_frame[side] = hequirect_frame_modified

    # Now we have to merge the modified_frame[L] or [R] back in the original image if they are not None.
    # If we have only one  side modified, load the original from vr_frame
    if modified_frame['L'] is None and modified_frame['R'] is None:
        logging.debug(f'No work to be done for frame {frame_name}')
        return False
    elif modified_frame['L'] is None:
        modified_frame['L'] = vr_frame[:, :half_width]
    elif modified_frame['R'] is None:
        modified_frame['R'] = vr_frame[:, half_width:]

    # Create an empty combined_img array
    combined_img = np.zeros((vr_width, vr_height, 3), dtype=vr_frame.dtype)

    # Assign values using np.hstack
    combined_img = np.hstack((modified_frame['L'], modified_frame['R']))

    # Save the combined image
    result = cv2.imwrite(output_folder + '/' + frame_name + '.jpg', combined_img)

    # update this frame to be done
    framedata['frames'][frame_name]['done'] = True

    return result


def process_split_frames(framedata, frames):

    global frames_path, processing_path, continue_processing

    continue_processing = True
    framecount = 0

    # update framecount with the remainder of frames left to process, based on the framedata dictionary. each frame will get a deep_update that it is done processing, e.g. framedata['frames'][frame]['done'] == True
    # TODO: rework this vs skipping in thread
    # completed = 0
    # for frame in framedata['frames'].keys():
    #    if 'done' in framedata['frames'][frame]:
    #        completed = completed + 1
    # framecount = completed

    temp = []
    logging.debug(f'initial {framecount} total {len(frames)}')
    with tqdm(initial=framecount, total=len(frames), desc='Processing', unit="frame", mininterval=1.0, smoothing=50/len(frames), dynamic_ncols=True,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as progress:
        save_time = time.time()
        for frame_path in frames:
            while len(temp) >= int(gpu_threads):
                has_result = temp.pop(0).join()

                if has_result:
                    progress.set_postfix(status='processing', refresh=True)
                else:
                    progress.set_postfix(status='skipping', refresh=True)
                progress.update(1)
            if continue_processing:
                # adding new frame to the list and starting it
                thread = ThreadWithReturnValue(target=merge_frame_thread, args=(frame_path, None))
                temp.append(thread)
                all_threads.append(thread)
                thread.start()

        # Check if 30 seconds have passed
        if time.time() - save_time >= 30:
            store_frame_data()
            save_time = time.time()


def store_frame_data():
    global lock, framedatafile_path, framedata

    with lock:
        with open(framedatafile_path, 'w') as f:
            logging.debug("storing data...")
            json.dump(framedata, f, indent=2, sort_keys=True)


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
    parser.add_argument("--gpu_threads", help="Threads", default=1, type=int)
    args = parser.parse_args()

    # Create a lock for thread-safe file writing
    lock = threading.Lock()

    global frames_path, processing_path, gpu_threads
    frames_path = args.frames_folder
    gpu_threads = args.gpu_threads

    # Load data json to get all frame info
    processing_path = os.path.join(frames_path, 'processing')
    print(processing_path)
    if not os.path.exists(processing_path):
        logging.fatal(f"No 'processing' folder found under {frames_path}, cannot proceed")
        sys.exit(1)

    framedatafile_path = os.path.join(processing_path, '_data.json')
    if not os.path.exists(framedatafile_path):
        logging.fatal(f"No '_data.json' file found in 'processing' folder under {frames_path}, cannot proceed")
        sys.exit(1)
    with lock:
        with open(framedatafile_path, 'r') as f:
            framedata = json.load(f)

    # traverse processed frames folder recursively, looking for all the frames regardless of directory; exit with error if duplicates found
    processed_split_file_paths = list(Path(processing_path).rglob('*.jpg'))
    processed_split_files = [file.name for file in processed_split_file_paths]
    if len(processed_split_files) != len(set(processed_split_files)):
        logging.fatal("Duplicate files found below processing folders. Please ensure there is only one file per frame/side/face.")
        sys.exit(1)
    else:
        logging.info(f"Found {len(processed_split_file_paths)} images to merge back into {len(framedata['frames'].keys())} frames. Mapping ")

    # update framedata dict with files if needed
    framedata_update = {}
    framedata_update['frames'] = {}
    # Process each side in a separate separately so as to not overwrite the framedata_update['frames'][frame]
    for jpg in processed_split_file_paths:
        pattern = r'(\d+)_(\w+)_(\d+)_(\d+)\.jpg'
        match = re.match(pattern, jpg.name)
        if match:
            frame_name, frame_side, vr_frame_face_index, persp_frame_face_index = match.groups()
            if frame_name in framedata['frames'] and frame_side in framedata['frames'][frame_name] and vr_frame_face_index in framedata['frames'][frame_name][frame_side] and persp_frame_face_index in framedata['frames'][frame_name][frame_side][vr_frame_face_index] and 'theta' in framedata['frames'][frame_name][frame_side][vr_frame_face_index][persp_frame_face_index]:
                if frame_name not in framedata_update['frames']:
                    framedata_update['frames'][frame_name] = {}
                if frame_side not in framedata_update['frames'][frame_name]:
                    framedata_update['frames'][frame_name][frame_side] = {}
                if vr_frame_face_index not in framedata_update['frames'][frame_name][frame_side]:
                    framedata_update['frames'][frame_name][frame_side][vr_frame_face_index] = {}
                if persp_frame_face_index not in framedata_update['frames'][frame_name][frame_side][vr_frame_face_index]:
                    framedata_update['frames'][frame_name][frame_side][vr_frame_face_index][persp_frame_face_index] = {}
                framedata_update['frames'][frame_name][frame_side][vr_frame_face_index][persp_frame_face_index]['path'] = jpg.as_posix()
            else:
                logging.warning(f'File {jpg.name} has no corresponding data in _data.json, cannot merge')
        else:
            logging.error(f"Failed to parse {jpg.name}")

    logging.debug("deep_update framedata")
    framedata = deep_update(framedata, framedata_update)
    # logging.debug(json.dumps(framedata, indent=1))

    # now collect the list of source frames, and compile the list of frames to process based on whether if we have an entry in framedata['frames'] for that frame
    # TODO: generate a separate list of frames that have a path associated with them. currently it still parses the framedata for frames with no jpg
    frames = []
    vr_frame_paths = list(Path(frames_path).glob('[0-9]*[0-9].jpg'))
    for frame_path in vr_frame_paths:
        pattern = r'(\d+)\.jpg'
        match = re.match(pattern, frame_path.name)
        if match:
            frames.append(frame_path.as_posix())
        else:
            logging.warn(f'Found frame {frame_path.name} source file but no split frames for it, skipping')

    # Set up signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)

    # start main thread
    process_split_frames(framedata, frames)
    print("Conversion successful!")
