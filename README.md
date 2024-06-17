# Point Cloud Semantic Segmentation by Adaptively Fusing Information with Varying Distances
by Zefeng Jiang, Baochen Yao, Kangkang Song, Xiaojie Qiu, and Chengbin Peng, details are in [paper]().


In this work, we argue that adaptive varying-distance feature aggregation and discrimination can improve the effect of point cloud semantic segmentation. The proposed approach consists of three steps. First, we agglomerate points into superpoints and construct a superpoint graph as many traditional approaches. Second, we propose a novel varying-distance autoencoder to help each superpoint adaptively assimilate information from different distances. Third, we propose a discrimination loss to constrain the embedding space so that superpoints belonging to the same semantic class can get closer and vice versa. 

### Dataset:

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
