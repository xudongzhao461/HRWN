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

### Data set links

1. Houston dataset were introduced for the 2013 IEEE GRSS Data Fusion contest. Data set links comes from http://www.grss-ieee.org/community/technical-committees/data-fusion/2013-ieee-grss-data-fusion-contest/

2. The authors would like to thank Dr. P. Ghamisi for providing the Trento Data. 

3. The MUUFL Gulfport Hyperspectral and LIDAR Data [1][2] is Available from https://github.com/GatorSense/MUUFLGulfport/.

[1] P. Gader, A. Zare, R. Close, J. Aitken, G. Tuell, “MUUFL Gulfport Hyperspectral and LiDAR Airborne Data Set,” University of Florida, Gainesville, FL, Tech. Rep. REP-2013-570, Oct. 2013.

[2] X. Du and A. Zare, “Technical Report: Scene Label Ground Truth Map for MUUFL Gulfport Data Set,” University of Florida, Gainesville, FL, Tech. Rep. 20170417, Apr. 2017. Available: http://ufdc.ufl.edu/IR00009711/00001.

### dataset utilization

Use Gramm-Schmidt method in ENVI to merge HSI and LiDAR-based DSM

**Please modify line 10-23 in *data_util_c.py* for the dataset details.**

### Training

Train the merged HSI and LiDAR-based DSM
```
python main.py --train merge --epochs 20 
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
