# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out/shakespeare'
device = 'cpu'  # device to run on, either 'cpu', 'gpu', or 'mps'
compile = False # do not torch compile the model

eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 20
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'default'

dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 64
pos_encoding = 'alibi'

# baby GPT model :)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 2_000
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially