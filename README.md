*Original Paper: 
Kim, Sookyung, et al. "Deep-hurricane-tracker: Tracking and forecasting extreme climate events." 2019 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2019:  https://ieeexplore.ieee.org/abstract/document/8658402.*

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

 From global scaled CAM5 reananlysis data, we only colleted region around **Nothern hemisphere** which is **180 degree to 340 degree longitude and 0 degree to 60 degree latitude**. For the purpose of training proposed tracking model, we fixed time length as 10 (which is 30 hours long).
The input image size is 128 times 257 pixels with around 0.50 degree (aroudn 50.0 km) resolution.

Basically, the file contains multiple spatio-temporal hurricane data (video) channeled by 6 climate variables with time length 10.
All climate variables are normalized between 0 and 1 accordingly. (Each channels has max value as 1 and min value as 0)

The details of input data, **hurricane_image.npy**, is following,

shape: (3408, 10, 128, 257, 6)


| Axis | Description | Size |
| :---         |     :---:      |          ---: |
| 0   | Number of hurricane videos     | 87837   |
| 1     | Time        | 10 (=30 hours)      |
| 2   | Width (as direction of longitude)    | 128 (= 64 degree = 6400 km)   |
| 3     | Height (as direction of latitude)       | 257 (= 128.5 degree  = 12850 km)     |
| 4     | Climate Variables        | 6 (by order: ['PSL', 'U850','V850','PRECT','TS','QREFHT'] )     |


#### Output: hurricane_label.npy####
As ground truth hurricane location in Input data, we used the corresponding TECA (*Byna, Surendra, et al. "Teca: Petascale pattern recognition for climate science." International Conference on Computer Analysis of Images and Patterns. Springer, Cham, 2015.: https://link.springer.com/chapter/10.1007/978-3-319-23117-4_37*) labels.
The TECA labels contain spatial coordinate (latitude, longitude) of each hurricane and the diameter of hurricane-force winds. We synthesize the ground-truth density maps as the same size with input data based on Gaussian mixtures.

The details of input data, **hurricane_label.npy**, is following,

shape: (3408, 10, 128, 257, 1)


| Axis | Description | Size |
| :---         |     :---:      |          ---: |
| 0   | Number of hurricane heat-map     | 87837   |
| 1     | Time        | 10 (=30 hours)      |
| 2   | Width (as direction of longitude)    | 128 (= 64 degree = 6400 km)   |
| 3     | Height (as direction of latitude)       | 257 (= 128.5 degree  = 12850 km)     |
| 4     | channel       | 1 (Pixel-level probability that hurricane exsists on that location: value between 0~1)     |


#### Visualization ####

Below is the visualization of 3 channels(1st - 3rd rows) among Input (hurricane_image.npy) file and according Output heat-map (4th row: hurricane_label.npy)
![Alt text](viz.png?width=300 "Title")


