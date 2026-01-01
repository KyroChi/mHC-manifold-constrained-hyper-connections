import json
import math
import os
import time

import torch

from model import GPT, GPTConfig

out_dir = "out"
eval_interval = 200
log_interval = 10
eval_iters = 200
max_iters = 2000

batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

learning_rate = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95

warmup_iters = 200
lr_decay_iters = 2000
min_lr = 6e-5

gradient_accumulation_steps = 1

seed = 1337

dataset = "shakespeare_char"

hc_num_streams = 1
hc_num_fracs = 1
hc_disable = True

exec(open(os.path.join(os.path.dirname(__file__), "configurator.py")).read())

torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device == "cuda" else "cpu"

data_dir = os.path.join(os.path.dirname(__file__), "data", dataset)
train_path = os.path.join(data_dir, "train.bin")
val_path = os.path.join(data_dir, "val.bin")
meta_path = os.path.join(data_dir, "meta.json")

train_data = torch.load(train_path)
val_data = torch.load(val_path)

with open(meta_path, "r") as f:
    meta = json.load(f)

vocab_size = meta["vocab_size"]

model_config = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
    hc_num_streams=hc_num_streams,
    hc_num_fracs=hc_num_fracs,
    hc_disable=hc_disable,
)

model = GPT(model_config)
model.to(device)

optimizer = model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(beta1, beta2),
    device_type=device_type,
)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])

    return x.to(device), y.to(device)


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with torch.no_grad():
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


iter_num = 0
best_val_loss = 1e9
loss = None

while iter_num <= max_iters:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            os.makedirs(out_dir, exist_ok=True)
            checkpoint = {
                "model": model.state_dict(),
                "config": model_config.__dict__,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    t0 = time.time()

    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        x, y = get_batch("train")
        _, loss = model(x, y)
        loss = loss / gradient_accumulation_steps
        loss.backward()

    optimizer.step()

    loss_item = loss.item() if loss is not None else float("nan")
    dt = time.time() - t0

    if iter_num % log_interval == 0:
        print(
            f"iter {iter_num}: loss {loss_item:.4f}, lr {lr:.5f}, time {dt * 1000:.2f}ms"
        )

    iter_num += 1
