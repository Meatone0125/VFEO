# VFEO
Download [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html) and extract `Stanford3dDataset_v1.2_Aligned_Version.zip` to `$S3DIS_DIR/data`, where `$S3DIS_DIR` is set to dataset directory.

Train
```
python learning/main.py --epochs 350
```

Test
```
python learning/main.py --epochs -1 --resume RESUME
```
