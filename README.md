# VFEO
Download [S3DIS Dataset](http://buildingparser.stanford.edu/dataset.html) and extract `Stanford3dDataset_v1.2_Aligned_Version.zip` to `$S3DIS_DIR/data`, where `$S3DIS_DIR` is set to dataset directory.

### Requirements:

- environment:

  ```
  Ubuntu 20.04
  ```

- install python package: 

  ```
  ./Anaconda3-5.1.0-Linux-x86_64.sh
  ```
  
- install PyTorch :

  ```
  conda install pytorch==1.11.0
  ```

### Train
```
python learning/main.py
```

### Test
```
python learning/main.py --epochs -1 --resume RESUME
```
### Citation:
```
 
```
