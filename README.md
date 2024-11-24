# Running config

## Environment 
```shell
conda activate /cluster/home/xiuming.li/miniconda3/envs/torchdrug
```

## Configuration

All config files are stored in folder `config/`, here is an example:
```yaml
gpu: 0 # GPU id

# Which reaction fingerpring to use, two options: drfp or morgan_fp. morgan_fp is used for baseline.
rxn_feat: 'drfp'

# Main model structure, gvp for protein pocket encoder, schnet for substrate encoder.
# Two options: gvp-schnet, gvp-painn
model: 'gvp-schnet'

# Interaction method, two options: cross-attention, pairformer
interaction_method: 'cross-attention'

# When interaction_method is cross-attention, batch_size can be 128(24G memory GPU) or 256(48G memory GPU)
batch_size: 128

# When interaction_method is pairformer, batch_size can only be 2(24G memory GPU) or 4(48G memory GPU)
# batch_size: 4

# Training data path. server ip: 123.6.102.203
train_path: '/home/liuy/data/SynBio/enzyme-reaction-pairs/training/v10/remove_no_ec/with_atom_limits/train.csv'
valid_path: '/home/liuy/data/SynBio/enzyme-reaction-pairs/training/v10/remove_no_ec/with_atom_limits/test.csv'
test_path: '/home/liuy/data/SynBio/enzyme-reaction-pairs/training/v10/remove_no_ec/with_atom_limits/test.csv'

## Training data on hpc server
# train_path: '/cluster/home/xiuming.li/data/SynBio/enzyme-reaction-pairs/training/v10/remove_no_ec/with_atom_limits/train.csv'
# valid_path: '/cluster/home/xiuming.li/data/SynBio/enzyme-reaction-pairs/training/v10/remove_no_ec/with_atom_limits/test.csv'
# test_path: '/cluster/home/xiuming.li/data/SynBio/enzyme-reaction-pairs/training/v10/remove_no_ec/with_atom_limits/test.csv'

# Checkpoint path. Remember to change the path to your own.
ckpt_dir: '/home/liuy/code/SynBio/enzyme-rxn-prediction/checkpoints/V10/no_ec_rxns/esm_gvp-schnet_drfp_cross-attn_lr1e-4_decay-0.95'

# No need to change these parameters
data_version: 'V10'
recall_method: 'selenzyme'
data_split_by: 'all'
auto_load_data: False
data_dir: ''
use_structure: True
use_drfp: True
use_esm: True
label_column_name: 'Label'
```

## Runing
Recommend to add `--debug` parameter to run first. In debug mode, only a small part of the training data will be loaded, which is much faster.
```shell
/cluster/home/xiuming.li/miniconda3/envs/torchdrug/bin/python main.py --mode train --config config_local/main.yaml --debug
```

