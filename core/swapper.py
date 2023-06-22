import os
import subprocess
import shutil
from tqdm import tqdm
import cv2
import insightface
import core.globals
from core.analyser import get_face

FACE_SWAPPER = None


def get_face_swapper():
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../inswapper_128.onnx')
        FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=core.globals.providers)
    return FACE_SWAPPER

def load_sbs_vr_frame(frame_path):
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


def perform_face_swap(frame, source_face):
    # Apply face swapping logic to the frame using source_face

    # Example code:
    # Perform face detection on the frame to get the target face coordinates
    target_face = get_face(frame)

    # Perform face swapping on the frame using source_face and target_face
    swapped_frame = get_face_swapper().get(frame, target_face, source_face, paste_back=True)

    return swapped_frame


def merge_sbs_frame(left_half, right_half):
    # Concatenate the left and right halves horizontally to merge them
    merged_frame = cv2.hconcat([left_half, right_half])

    return merged_frame


def save_sbs_vr_frame(frame, frame_path):
    # Save the frame back to the frame path
    cv2.imwrite(frame_path, frame)

    # extract frame count from filename
    frame_number = os.path.splitext(os.path.basename(frame_path))[0]

    with open('./frames.txt', 'w') as file:
            file.write(str(frame_number))


def process_video(source_img, frame_paths, swap_face, upscale_face):
    print("swap " + str(swap_face))
    print("upscale " + str(upscale_face))
    source_face = get_face(cv2.imread(source_img))
    with tqdm(total=len(frame_paths), desc="Processing", unit="frame", dynamic_ncols=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as progress:
        for frame_path in frame_paths:
            if swap_face:
                # Load the SBS VR video frame
                vr_frame = load_sbs_vr_frame(frame_path)

                # Split the frame into left and right halves
                left_half, right_half = split_sbs_frame(vr_frame)

                # Perform face swapping on the left half
                swapped_left_half = perform_face_swap(left_half, source_face)

                # Perform face swapping on the right half
                swapped_right_half = perform_face_swap(right_half, source_face)

                # Merge the swapped halves back into a single SBS frame
                swapped_vr_frame = merge_sbs_frame(swapped_left_half, swapped_right_half)

                # Save the modified frame back to the frame path
                save_sbs_vr_frame(swapped_vr_frame, frame_path)

            if upscale_face:
                upscale_image(frame_path)

            progress.set_postfix(status='.', refresh=True)
        
        print('Swapping completed!')


def process_img(source_img, target_path, output_file):
    source_face = get_face(cv2.imread(source_img))
        # Load the SBS VR video frame
    vr_frame = load_sbs_vr_frame(target_path)

    # Split the frame into left and right halves
    left_half, right_half = split_sbs_frame(vr_frame)

    # Perform face swapping on the left half
    swapped_left_half = perform_face_swap(left_half, source_face)

    # Perform face swapping on the right half
    swapped_right_half = perform_face_swap(right_half, source_face)

    # Merge the swapped halves back into a single SBS frame
    swapped_vr_frame = merge_sbs_frame(swapped_left_half, swapped_right_half)

    # Save the modified frame back to the frame path
    save_sbs_vr_frame(swapped_vr_frame, frame_path)

def upscale_image(input_file):
    # Define the output file path
    file_basename = os.path.splitext(os.path.basename(input_file))[0]
    gfpgan_generated_file = os.path.splitext(input_file)[0] + '/final_results/' + file_basename + ".png"
    gfpgan_command = f'python CodeFormer/inference_codeformer.py -i "{input_file}" -o {os.path.splitext(input_file)[0]} -w 0.7 -s 1 --face_upsample'
    subprocess.run(gfpgan_command, shell=True, check=True)
    resize_image(gfpgan_generated_file, input_file)
    shutil.rmtree(os.path.splitext(input_file)[0])

    with open('./upscaled.txt', 'w') as file:
        file.write(str(frame_number))


def resize_image(input_file, export_location):
    src = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)
    #percent by which the image is resized
    scale_percent = 50

    #calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv2.resize(src, dsize)

    cv2.imwrite(export_location,output)
    
    # extract frame count from filename
    frame_number = os.path.splitext(os.path.basename(export_location))[0]

    with open('./frames.txt', 'w') as file:
            file.write(str(frame_number))