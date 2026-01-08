# OpenWebText with HyperConnections (124M parameters)
# GPT-2 base architecture with 4-stream HyperConnections
#
# Usage:
#   python train.py config/train_owt_124m_hc.py
#   torchrun --standalone --nproc_per_node=4 train.py config/train_owt_124m_hc.py

wandb_log = True
wandb_project = "nanogpt-mhc"
wandb_run_name = "gpt2-124M-hc"

out_dir = "out-owt-124m-hc"

dataset = "openwebtext"

# model (GPT-2 base: 124M params)
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# training
# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
gradient_accumulation_steps = 5 * 8  # will be adjusted by train.py based on actual GPU count

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# optimizer
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
warmup_iters = 2000
min_lr = 6e-5

# dtype
dtype = "bfloat16"

# hyper-connections: ENABLED (4 streams)
hc_num_streams = 4
hc_num_fracs = 1
hc_disable = False

