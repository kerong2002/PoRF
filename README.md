# Reproduction_of_PoRF
Reproduction of PoRF and use it on lab`s digital theater

## Usage
### Data Convention
The data will be organized as follows:
```
porf_data
|---dtu
    |---<case_name>
        |-- cameras.npz        #  GT camera parameters
        |-- cameras_colmap.npz #  COLMAP camera parameters 
        |-- image
            |-- 000.png        # target image for each view filled by Colmap
            |-- 001.png
            ...
        |-- images
            |-- 000.png        # all image for each view
            |-- 001.png
            ...
        |-- colmap_matches
            |-- 000000.npz     # matches exported from COLMAP
            |-- 000001.npz
            ...
        |-- sparse_points.ply  # ply file create by img2poses.py
        |-- sparse_points_interest.ply  # ply file modified to remove noise
        |-- database.db        # database file create by COLMAP
        ...

exp_dtu
|---<case_name>
    |-- dtu_sift_porf
        |-- meshes       
            |-- 00xxxxxx.ply   # target ply file trained by PoRF
        ...

```

### Setup
```
git clone https://github.com/Cerosop/Reproduction_of_PoRF.git
cd porf

conda create -n porf_reproduction python=3.9
conda activate porf_reproduction
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Running
```
python surface.py
```

## Acknowledgement
Some code snippets are borrowed from [PoRF](https://github.com/ActiveVisionLab/porf). Thanks for these great projects.
