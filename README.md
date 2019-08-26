*Original Paper: 
Kim, Soo Kyung, et al. "Learning to Focus and Track Extreme Climate Events". 30th British Machine Vision Conference(BMVC) (2019)*

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
