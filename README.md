conda create -n vrswap
conda activate vrswap

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -c conda-forge cupy
pip install piexif
pip install insightface
pip install Pillow
pip install tqdm
pip install onnxruntime-gpu
pip install opencv-python
pip install psutil