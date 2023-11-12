# VRSWAP
A set of tools to do face swapping on equirectangular VR180 videos

## Installation
### Linux or WSL2
Tested on Debian, both natively and under WSL2. Make sure you have python 3.10.
```
python3.10 -m virtualenv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.0.1 torchaudio torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

```
### Windows
```
conda create -n vrswap
conda activate vrswap
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge cupy cuda-version=11.8
pip install -r requirements.txt
```

## Notes

- Paths can be Linux (`/some/where/`) or Windows (`d:\some\where\`) style, both are supported.
- JPG is used for speed and sheer size of the frames; some minor quality loss is to be expected but should be imperceptible to the human eye
- Nvidia GPU is required (CUDA), 2070 and up should work (around 6-8gb VRAM)
- The working folder is used to create temporary images and a json file to track progress and frame data. Ensure you have enough space.


## Usage
1. Convert your VR180 equirectangular video to a directory of frames using e.g. ffmpeg:
	 `ffmpeg -i /path/to/VR/video.mp4 -pix_fmt yuvj444p -qmin 1 -qscale:v 1 /path/to/working/folder/%04d.jpg `

2. Convert the VR180 frame into two perspective frames (one per eye) that crop on the face:
	`python vr-splitter.py --frames_folder /path/to/working/folder`

3. Subdirectories are created under processing/ with different faces. First delete all frames that are invalid (misdetected). Then Use whatever tool you prefer to manipulate the images - either change them in-place, or write the new images to separate folders. When done, delete all frames that should not be merged back, as the next step will recursively traverse all folders under processing/ and will look for 0 or 1 matching image - more than 1 of the same is a fatal error.

4. Reinsert the swapped (and upscaled) left and right perspective crops back into the equirectangular frames:
	`python vr-merger.py --frames_folder /path/to/working/folder`
	
5. Encode the frames back to video (H265 recommended); note this is just an example, use the source video FPS etc:
	`fmpeg -r 59.997 -i /path/to/working/folder/%04d.jpg -c:v libx265  -c:a aac -preset fast -crf 18 -vf scale=in_range=limited:out_range=full -pix_fmt yuvj420p -movflags faststart /path/to/output/video.mp4`
