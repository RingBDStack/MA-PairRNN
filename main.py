import argparse
from model import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--cosineloss_coefficient', type=float, default=1)

    parser.add_argument('--meta_num', type=int, default=4)
    parser.add_argument('--feature_dim', type=int, default=300)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--meta_preference_dim', type=int, default=32)
    parser.add_argument('--RNN_input_dim', type=int, default=64)

    parser.add_argument('--SAGE_hidden_dim', type=int, default=64)
    parser.add_argument('--RNN_hidden_dim', type=int, default=64)
    parser.add_argument('--RNN_layer_num', type=int, default=1)

    parser.add_argument('--label_size', type=int, default=2)
    parser.add_argument('--epoch_num', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_cuda', action="store_true", default=True)

    parser.add_argument('--gpu_id', help="GPU_ID", type=str, default="0")
    parser.add_argument('--model_name', type=str, default="default_name")
    parser.add_argument('--meta_fuse', type=str, default="product_att")
    parser.add_argument('--train_test_ratio', type=str, default="82")
    args = parser.parse_known_args()[0]
    train_total(args)
