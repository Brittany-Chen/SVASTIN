# RRDB
nf = 3
gc = 32

# Super parameters
eps = 8/255
clamp = 2.0
channels_in = 3
lr = 1e-5
lr2 = 1/255
lr_min = 1e-7
epochs = 501
weight_decay = 1e-5
init_scale = 0.01
# regulation parameter
lambda_a = 3
beta_a = 0.4
gama_a = 100
lambda_b = 1
beta_b = 0.01

# Super loss

lamda_guide = 1
lamda_low_frequency = 1
lamda_per = 0.001

# Train:

betas = (0.5, 0.999)
weight_step = 300
gamma = 0.99
