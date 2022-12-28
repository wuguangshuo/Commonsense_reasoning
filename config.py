import argparse

def set_args():
    parser = argparse.ArgumentParser(description='Salient triple classification')
    parser.add_argument("--test_batch", default=100, type=int, help="Test every X updates steps.")
    parser.add_argument("--log_dir", type=str, default='./log/train.log', help='日志的存放位置')
    parser.add_argument('--seed', type=int, default=2022, help='设置随机种子')
    parser.add_argument('--n_split', type=int, default=5, help='交叉验证')
    parser.add_argument("--data_dir", default="data", type=str, help="The task data directory.")
    parser.add_argument("--model_dir", default="nghuyong/ernie-gram-zh", type=str,
                        help="The directory of pretrained models")
    parser.add_argument("--epsilon", default=0.3, type=float, help="Drop out rate")
    parser.add_argument("--use_fgm", default=True, type=bool, help="The task data directory.")
    # parser.add_argument("--model_dir", default="hfl/chinese-roberta-wwm-ext", type=str, help="The directory of pretrained models")
    parser.add_argument("--output_dir", default='output/save_dict/', type=str, help="The path of result data and models to be saved.")
    # models param
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training.")
    parser.add_argument("--lr", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--dropout", default=0.2, type=float, help="Drop out rate")
    parser.add_argument("--smoothing", default=0.05, type=float, help="Drop out rate")
    parser.add_argument("--epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--hidden_size', type=int, default=768,  help="random seed for initialization")
    args = parser.parse_args()
    return args
