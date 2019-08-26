*Original Paper: 
Kim, Sookyung, et al. "Deep-hurricane-tracker: Tracking and forecasting extreme climate events." 2019 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2019.*

# **Hurricane Tracking Dataset**
Spatio-temporal Climate Simulation Data containing hurricane and accoriding heat-map labels

## Input and label
#### Loading data #### 
```
import numpy as np
x=np.load("hurricane_image.npy") # input: shape=(3408, 10, 128, 257, 6)
y=np.load(""hurricane_label.npy") # output: shape=(3408, 10, 128, 257, 1)
```
#### Input: hurricane_image.npy ####
We use 20-year-long records from 1996 to 2015 of **Community Atmospheric Model v5 (CAM5)** dataset. It contains snapshots of the global atmospheric states for every three hours. (1 timestep = 3 hours) Each snapshot contains multiple physical variables among which we use the six most important climate variables to define hurricane from scientific literature, such as PSL (Sea level pressure),  U850 (Zonal wind), V850 (Meriodional wind), PRECT (Precipitation), TS (Surface temperature), QREFHT (Reference high humidity) by order.

 From global scaled CAM5 reananlysis data, we only colleted region around **Nothern hemisphere** which is **180 degree to 340 degree longditude and 0 degree to 60 degree latitude**. For the purpose of training proposed tracking model, we fixed time length as 10 (which is 30 hours long).
The input image size is 128 times 257 pixels with around 0.50 degree (aroudn 50.0 km) resolution.


The details of input data, **hurricane_image.npy**, is following,

shape: (3408, 10, 128, 257, 6)


| Axis | Description | Size |
| :---         |     :---:      |          ---: |
| 0   | Number of hurricane videos     | 87837   |
| 1     | Time        | 10 (=30 hours)      |
| 2   | Width     | 128 (= 64 degree = 6400 km)   |
| 3     | Height        | 257 (= 128.5 degree  = 12850 km)     |
| 4     | Climate Variables        | 6 (by order: ['PSL', 'U850','V850','PRECT','TS','QREFHT'] )     |


#### Output: hurricane_label.npy####
As ground truth hurricane location in Input data, we used the corresponding TECA labels, which contain the latitude and longitude of each hurricane and the diameter of hurricane-force winds.

