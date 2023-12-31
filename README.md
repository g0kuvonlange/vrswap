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
git clone https://github.com/sczhou/CodeFormer

```
### Windows
```
conda create -n vrswap
conda activate vrswap
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge cupy
pip install -r requirements.txt
git clone https://github.com/sczhou/CodeFormer
```

## Notes

- Paths can be Linux (`/some/where/`) or Windows (`d:\some\where\`) style, both are supported.
- JPG is used for speed and sheer size of the frames; some minor quality loss is to be expected but should be imperceptible to the human eye
- Nvidia GPU is required (CUDA), 2070 and up should work (around 6-8gb VRAM)
- The working folder is used to create temporary images and a json file to track progress and frame data. Ensure you have enough space.


## Usage
1. Convert your VR180 equirectangular video to a directory of frames using e.g. ffmpeg:
	 `ffmpeg -i /path/to/VR/video.mp4 -pix_fmt yuvj444p  -qscale:v 1 /path/to/working/folder/%04d.jpg `

2. Convert the VR180 frame into two perspective frames (one per eye) that crop on the face, and then swap the face (E2P + SWAP):
	`python swap.py --frames_folder /path/to/working/folder --face /path/to/face/to/swap/into/frames.jpg --gpu`

3. (Optional) Upscale the cropped perspective face-swapped frames using codeformer
	`python upscale.py --frames_folder /path/to/working/folder`

4. Reinsert the swapped (and upscaled) left and right perspective crops back into the equirectangular frames (P2E):
	`python convert.py --frames_folder /path/to/working/folder`
	
5. Encode the frames back to video (H265 recommended); note this is just an example, use the source video FPS etc:
	`fmpeg -r 59.997 -i /path/to/working/folder/%04d.jpg -c:v libx265  -c:a aac -preset fast -crf 18 -vf scale=in_range=limited:out_range=full -pix_fmt yuvj420p -movflags faststart /path/to/output/video.mp4`
