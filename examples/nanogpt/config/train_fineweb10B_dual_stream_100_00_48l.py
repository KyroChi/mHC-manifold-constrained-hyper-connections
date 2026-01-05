# FineWeb10B with Convex Dual Stream (48 layers) - 1.0/0.0 weights
# Effectively just pre-layer norm stream (standard transformer)

out_dir = "out-fineweb10B-dual-stream-100-00-48l"
wandb_project = "nanogpt-mhc"
dataset = "fineweb10B"

block_size = 1024
n_layer = 48
n_head = 6
n_embd = 150
dropout = 0.0
bias = False

batch_size = 8
gradient_accumulation_steps = 4
max_iters = 5000
eval_interval = 500
log_interval = 10
eval_iters = 100

learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 200
lr_decay_iters = 5000
min_lr = 6e-5

dtype = "bfloat16"

hc_num_streams = 1
hc_num_fracs = 1
hc_disable = True

dual_stream = True
dual_stream_weight = 1.0

