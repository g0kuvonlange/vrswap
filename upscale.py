import os
import time
import subprocess
import argparse

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--frames_folder", help="Frames folder")
parser.add_argument("--threads", help="Threads", default=4, type=int)
args = parser.parse_args()

frames_folder = args.frames_folder
threads = args.threads

def launch_codeformer(input_folder, processes):
    codeformer_command = f'python CodeFormer/inference_roop.py -i "{input_folder}" -o "{input_folder}/upscaled" -w 0.85 -s 1 --face_upsample --rewrite'
    print(codeformer_command)
    process = subprocess.Popen(codeformer_command, shell=True)
    processes.append(process)

def main():
    processing_path = frames_folder + "/processing"
    processes = []

    while True:
        # Check the status of running processes
        for process in processes[:]:
            if process.poll() is not None:  # Process has finished
                processes.remove(process)

        # Launch new processes if there is room
        if len(processes) < threads:
            launch_codeformer(processing_path, processes)
            time.sleep(4)

        # Break the loop if all processes have finished
        if len(processes) == 0:
            break

        # Add a delay before checking the process status again
        time.sleep(5)

    print("All CodeFormer processes completed.")

if __name__ == '__main__':
    main()
