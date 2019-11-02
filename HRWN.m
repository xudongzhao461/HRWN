%% demo for Joint Classification of Hyperspectral and LiDAR Data Using Hierarchical Random Walk and Deep CNN Architecture
%--------------Brief description-------------------------------------------
%
% 
% This demo implements HRWN destriping for Hyperspectral and LiDAR Data
% classification
%
%
% More details in [1]:
%
% [1] Xudong Zhao£¬Ran Tao£¬Wei Li ,Hengchao Li and Qian Du,  " Joint
% Classification of Hyperspectral and LiDAR Data Using Hierarchical Random 
% Walk and Deep CNN Architecture," in peer review IEEE Trans. on Geoscience and Remote Sensing.
%
% contact: zhaoxudong@bit.edu.cn (Xudong Zhao), rantao@bit.edu.cn, liwei089@ieee.org (Wei Li),
% 

clc;clear all;

load('muufpred1111.mat');   %   pred matrix saved from  main.py
load('muufind1111.mat');    %   index saved from main.py
load('muuf_merge.mat');     %   the merged data
Data=data;
load('muuf_lidar.mat');     %   the LiDAR-based DSM data
lidar=data(:,:,1);
load('muuf_mask_test_150.mat');     %   samples for test 
load('muuf_mask_train_150.mat');    %	samples for train
[r,c,b]=size(Data);
[GT, ~,GTlabel]=find(reshape(mask_test,[r*c,1]));
GT=GT';
GTlabel=double(GTlabel');
[seeds,~,labels]=find(reshape(mask_train,[r*c,1]));
seeds=seeds';
labels=double(labels');
P=zeros(r,c,11);

for i=1:size(index,2)
P(index(1,i)+1,index(2,i)+1,:)=p(i,:);
end
%% hierarchical random walk optimization
[HBRWresult,probability] = RWOptimize(lidar,seeds',labels',0.1^5,P,0.7,1); 
[OA,kappa, AA,CA]= calcError((GTlabel-1), HBRWresult(GT)-1,[1:11]);           
aaa=im2bw(mask_test+mask_train,0.5);
map=reshape(ERWresult,[r,c]).*aaa;
figure;
imshow(map,[]);