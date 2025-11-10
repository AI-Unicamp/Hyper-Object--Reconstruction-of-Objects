import numpy as np 

# data parameters
crop_size = 32 #320

# data info
bands = np.arange(400, 1000, 10)

# training parameters
batch_size = 1 # valor original: 2
epochs = 1  # valor original: 14
init_lr = 4e-4