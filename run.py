#!/usr/bin/env python3

import platform
import signal
import sys
import shutil
import glob
import argparse
import multiprocessing as mp
import os
import torch
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import asksaveasfilename
import webbrowser
import psutil
import cv2
import threading
from PIL import Image as PILImage
from PIL import Image, ImageTk
import core.globals
from core.swapper import process_video, process_img
from core.utils import is_img, detect_fps, set_fps, create_video, add_audio, extract_frames, rreplace
from core.analyser import get_face
from core.gpu_optimizer import process_video_gpu
if 'ROCMExecutionProvider' in core.globals.providers:
    del torch

pool = None
args = {}

signal.signal(signal.SIGINT, lambda signal_number, frame: quit())
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--face', help='use this face', dest='source_img')
parser.add_argument('-t', '--target', help='replace this face', dest='target_path')
parser.add_argument('-o', '--output', help='save output to this file', dest='output_file')
parser.add_argument('--gpu', help='use gpu', dest='gpu', action='store_true', default=False)
parser.add_argument('--swap-face', help='maintain original fps', dest='swap_face', action='store_true', default=False)
parser.add_argument('--upscale-face', help='keep frames directory', dest='upscale_face', action='store_true', default=False)
parser.add_argument('--convert', help='convert faces back to equirectangular', dest='convert', action='store_true', default=False)
parser.add_argument('--max-memory', help='maximum amount of RAM in GB to be used', type=int)
parser.add_argument('--max-cores', help='number of cores to be use for CPU mode', dest='cores_count', type=int, default=max(psutil.cpu_count() - 2, 2))
parser.add_argument('--gpu-threads', help='number of threads for gpu to run in parallel', dest='gpu_threads', type=int, default=4)

for name, value in vars(parser.parse_args()).items():
    args[name] = value

sep = "/"
if os.name == "nt":
    sep = "\\"


def limit_resources():
    if args['max_memory']:
        memory = args['max_memory'] * 1024 * 1024 * 1024
        if str(platform.system()).lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


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


def start_processing():
    swap_face = bool(args["swap_face"])
    upscale_face = bool(args["upscale_face"])
    convert = bool(args["convert"])
    if args['gpu']:
        #process_video(args['source_img'], args["frame_paths"], swap_face, upscale_face)
        process_video_gpu(args['source_img'], 
                    args['frame_paths'], 
                    int(args['gpu_threads']),
                    swap_face,
                    upscale_face,
                    convert)
        return
    frame_paths = args["frame_paths"]
    n = len(frame_paths)//(args['cores_count'])
    processes = []
    for i in range(0, len(frame_paths), n):
        p = pool.apply_async(process_video, args=(args['source_img'], frame_paths[i:i+n], swap_face, upscale_face))
        processes.append(p)
    for p in processes:
        p.get()
    pool.close()
    pool.join()


def preview_image(image_path):
    img = Image.open(image_path)
    img = img.resize((180, 180), Image.ANTIALIAS)
    photo_img = ImageTk.PhotoImage(img)
    left_frame = tk.Frame(window)
    left_frame.place(x=60, y=100)
    img_label = tk.Label(left_frame, image=photo_img)
    img_label.image = photo_img
    img_label.pack()


def preview_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((180, 180), Image.ANTIALIAS)
        photo_img = ImageTk.PhotoImage(img)
        right_frame = tk.Frame(window)
        right_frame.place(x=360, y=100)
        img_label = tk.Label(right_frame, image=photo_img)
        img_label.image = photo_img
        img_label.pack()

    cap.release()


def select_face():
    args['source_img'] = filedialog.askopenfilename(title="Select a face")
    preview_image(args['source_img'])


def select_target():
    args['target_path'] = filedialog.askdirectory(title="Select a target folder")
    #threading.Thread(target=preview_video, args=(args['target_path'],)).start()


def toggle_swap_face():
    args['swap_face'] = int(swap_face.get())


def toggle_upscale_face():
    args['upscale_face'] = int(upscale_face.get())

def toggle_convert():
    args['convert'] = int(convert.get())


def save_file():
    filename, ext = 'output.mp4', '.mp4'
    if is_img(args['target_path']):
        filename, ext = 'output.png', '.png'
    args['output_file'] = asksaveasfilename(initialfile=filename, defaultextension=ext, filetypes=[("All Files","*.*"),("Videos","*.mp4")])


def status(string):
    if 'cli_mode' in args:
        print("Status: " + string)
    else:
        status_label["text"] = "Status: " + string
        window.update()


def start():
    swap_face = bool(args["swap_face"])
    upscale_face = bool(args["upscale_face"])
    convert = bool(args["convert"])
    if not args['source_img'] or not os.path.isfile(args['source_img']):
        print("\n[WARNING] Please select an image containing a face.")
        return
    # elif not args['target_path'] or not os.path.isfile(args['target_path']):
    #     print("\n[WARNING] Please select a video/image to swap face in.")
    #     return
    # if not args['output_file']:
    #     target_path = args['target_path']
    #     args['output_file'] = rreplace(target_path, "/", "/swapped-", 1) if "/" in target_path else "swapped-" + target_path
    global pool
    pool = mp.Pool(args['cores_count'])
    target_path = args['target_path']
    test_face = get_face(cv2.imread(args['source_img']))
    if not test_face:
        print("\n[WARNING] No face detected in source image. Please try with another one.\n")
        return
    if is_img(target_path):
        process_img(args['source_img'], target_path, args['output_file'])
        status("swap successful!")
        return

    video_name_full = target_path.split("/")[-1]
    video_name = os.path.splitext(video_name_full)[0]
    #output_dir = os.path.dirname(target_path) + "/" + video_name
    output_dir = target_path
    Path(output_dir).mkdir(exist_ok=True)
    #status("detecting video's FPS...")
    #fps, exact_fps = detect_fps(target_path)
    #if not args['keep_fps'] and fps > 30:
    #    this_path = output_dir + "/" + video_name + ".mp4"
    #    set_fps(target_path, this_path, 30)
    #    target_path, exact_fps = this_path, 30
    #else:
    #    shutil.copy(target_path, output_dir)
    #status("extracting frames...")
    #extract_frames(target_path, output_dir)

    read_file = './frames.txt'
    
    if upscale_face:
        read_file = './upscaled.txt'
    
    if os.path.isfile(read_file) and os.stat(read_file).st_size != 0:
        with open(read_file, 'r') as file:
            last_processed_frame = int(file.read().strip())
    else:
        last_processed_frame = 0

    frame_paths = []
    for frame_path in glob.glob(output_dir + "/*.jpg"):
        frame_number = int(os.path.splitext(os.path.basename(frame_path))[0])
        if frame_number > last_processed_frame:
            frame_paths.append(frame_path)

    frame_paths = tuple(sorted(frame_paths, key=lambda x: int(x.split(sep)[-1].replace(".jpg", ""))))
    frame_paths = [f for f in frame_paths if not f.endswith('.png')]
    args['frame_paths'] = frame_paths

    status("swapping in progress...")
    start_processing()
    #status("creating video...")
    #create_video(video_name, exact_fps, output_dir)
    #status("adding audio...")
    #add_audio(output_dir, target_path, video_name_full, args['keep_frames'], args['output_file'])
    #save_path = args['output_file'] if args['output_file'] else output_dir + "/" + video_name + ".mp4"
    #print("\n\nVideo saved as:", save_path, "\n\n")
    status("swap successful!")


if __name__ == "__main__":
    global status_label, window

    pre_check()
    limit_resources()

    if args['source_img']:
        args['cli_mode'] = True
        start()
        quit()
    window = tk.Tk()
    window.geometry("600x700")
    window.title("roop")
    window.configure(bg="#2d3436")
    window.resizable(width=False, height=False)

    # Contact information
    support_link = tk.Label(window, text="Donate to project <3", fg="#fd79a8", bg="#2d3436", cursor="hand2", font=("Arial", 8))
    support_link.place(x=180,y=20,width=250,height=30)
    support_link.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/sponsors/s0md3v"))

    # Select a face button
    face_button = tk.Button(window, text="Select a face", command=select_face, bg="#2d3436", fg="#74b9ff", highlightthickness=4, relief="flat", highlightbackground="#74b9ff", activebackground="#74b9ff", borderwidth=4)
    face_button.place(x=60,y=320,width=180,height=80)

    # Select a target button
    target_button = tk.Button(window, text="Select a target", command=select_target, bg="#2d3436", fg="#74b9ff", highlightthickness=4, relief="flat", highlightbackground="#74b9ff", activebackground="#74b9ff", borderwidth=4)
    target_button.place(x=360,y=320,width=180,height=80)

    # Swap Face
    swap_face = tk.IntVar(None, args['swap_face'])
    face_checkbox = tk.Checkbutton(window, relief="groove", activebackground="#2d3436", activeforeground="#74b9ff", selectcolor="black", text="Swap face", fg="#dfe6e9", borderwidth=0, highlightthickness=0, bg="#2d3436", variable=swap_face, command=toggle_swap_face)
    face_checkbox.place(x=30,y=400,width=240,height=31)

    # Upscale Face
    upscale_face = tk.IntVar(None, args['upscale_face'])
    upscale_checkbox = tk.Checkbutton(window, relief="groove", activebackground="#2d3436", activeforeground="#74b9ff", selectcolor="black", text="Upscale face", fg="#dfe6e9", borderwidth=0, highlightthickness=0, bg="#2d3436", variable=upscale_face, command=toggle_upscale_face)
    upscale_checkbox.place(x=37,y=450,width=240,height=31)

    # Convert Perspective to Equarectangular
    convert = tk.IntVar(None, args['convert'])
    convert_checkbox = tk.Checkbutton(window, relief="groove", activebackground="#2d3436", activeforeground="#74b9ff", selectcolor="black", text="Convert Back", fg="#dfe6e9", borderwidth=0, highlightthickness=0, bg="#2d3436", variable=convert, command=toggle_convert)
    convert_checkbox.place(x=37,y=500,width=240,height=31)

    # Start button
    start_button = tk.Button(window, text="Start", bg="#f1c40f", relief="flat", borderwidth=0, highlightthickness=0, command=lambda: [start()])
    #start_button = tk.Button(window, text="Start", bg="#f1c40f", relief="flat", borderwidth=0, highlightthickness=0, command=lambda: [save_file(), start()])
    start_button.place(x=240,y=560,width=120,height=49)

    # Status label
    status_label = tk.Label(window, width=580, justify="center", text="Status: waiting for input...", fg="#2ecc71", bg="#2d3436")
    status_label.place(x=10,y=640,width=580,height=30)
    
    window.mainloop()