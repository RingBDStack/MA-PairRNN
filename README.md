# MA-PairRNN
Code for "Pairwise Learning for Name Disambiguation in Large-Scale Heterogeneous Academic Networks"

## How to run the code: 

```bash
pip install -r requirements.txt
python main.py --model_name='complete-82' --train_test_ratio='82'
```

## Parameter explanation in args setting: 

```
learning_rate(float, default=5e-4): learning rate of the optimizer
cosineloss_coefficient(type=float, default=1): the cosine similarity loss's coefficient, while the cross entropy loss's coefficient is constantly set to 1.
meta_num(int, default=4): number of metapaths
feature_dim(int, default=300): original input feature's dim
embedding_dim(int, default=64): dim after node embedding
meta_preference_dim(int, default=32): preference vector dim used in preference attention
RNN_input_dim(int, default=64): the input dim of the PairRNN
SAGE_hidden_dim(int, default=64): the hidden layer dim of GraphSAGE node embedding method
RNN_hidden_dim(int, default=64): the hidden layer dim of PairRNN
RNN_layer_num(int, default=1): the number of hidden layers in PairRNN
meta_fuse(str, default="product_att"): choose which meta_fuse mothods, can be 'preference_att', 'product_att', 'product_ave_att' or 'mlp'
train_test_ratio(str, default="82"): the ratio training/testing, can be '28', '46', '64', '82'
model_name(str, default="default_name"): influence the log file name, highly recommended to set a name for discrimination
```

## Reference
```
@article{sun2020pairwise,
  title={Pairwise Learning for Name Disambiguation in Large-Scale Heterogeneous Academic Networks},
  author={Sun, Qingyun and Peng, Hao and Li, Jianxin and Wang, Senzhang and Dong, Xiangyu and Zhao, Liangxuan and Yu, Philip S and He, Lifang},
  journal={arXiv preprint arXiv:2008.13099},
  year={2020}
}
```
