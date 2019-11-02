# Hierarchical-Random-Walk-network-for-Hyperspectral-and-LiDAR-classification

This example implements the paper in review [Joint Classification of Hyperspectral and LiDAR Data Using Hierarchical Random Walk and Deep CNN Architecture]

A Joint Classification method of Hyperspectral and LiDAR Data Using Hierarchical Random Walk and Deep CNN Architecture. Reach a quite high classification accuracy. Evaluated on the dataset of Houston, Trento and MUUFL. 

## Prerequisites
- Python 2.7 or 3.6
- Packages
```
pip install -r requirements.txt
```

## Usage

### dataset utilization

Use Gramm-Schmidt method in ENVI to merge HSI and LiDAR-based DSM

**Please modify line 10-23 in *data_util_c.py* for the dataset details.**

### Training
1. Train the merged HSI and LiDAR-based DSM
```
python main.py --train merge --epochs 20 --modelname ./logs/weights/hsi.h5
```
2. Train LiDAR
```
python main.py --train lidar --epochs 20 --modelname ./logs/weights/lidar.h5
```
3. Train two branches
```
python main.py --train finetune --epochs 20 --modelname ./logs/weights/model.h5
```
save pred.npy and index.npy in （.mat）model

### Hierarchical Random Walk Optimization

run HBRW.m in Matlab 

## Results
All the results are cited from original paper. More details can be found in the paper.

| dataset  	 | Kappa | OA      |
|---------- |-------  |--------|
| Houston  | 93.09%| 93.61%|
| Trento    | 98.48%| 98.86% |
| MUUFL    | 92.52%| 94.31% |

## Citation
```

```
## TODO
1. pytorch version.
2. more flexiable dataset utilization
