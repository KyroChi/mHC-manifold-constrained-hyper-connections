# FineWeb10B with Convex Dual Stream (48 layers) - 0.90/0.10 weights
# Two parallel residual streams: pre-layer norm (0.90) + post-layer norm (0.10)
#
# Usage:
#   python train.py config/train_fineweb10B_dual_stream_90_10_48l.py

out_dir = "out-fineweb10B-dual-stream-90-10-48l"
wandb_project = "nanogpt-mhc"

dataset = "fineweb10B"

# model
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

# optimizer
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 6e-5

# dtype
dtype = "bfloat16"

# hyper-connections: DISABLED
hc_num_streams = 1
hc_num_fracs = 1
hc_disable = True

# dual stream: ENABLED with 0.90 pre-LN, 0.10 post-LN
dual_stream = True
dual_stream_weight = 0.9

