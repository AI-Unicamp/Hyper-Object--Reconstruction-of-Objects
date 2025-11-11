import numpy as np 

# data parameters
crop_size = 256 # valor original: 320

# data info
bands = np.arange(400, 1000, 10)

# training parameters
batch_size = 3 # valor original: 2
epochs = 1000  # valor original: 14
init_lr = 4e-4 # valor original: 4e-4