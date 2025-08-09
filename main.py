import argparse
import time
import torch
import tool
from enviroment import *
from loader import Loader4MM
from model import MOME_model
from session import Session


def parse_args():   
    parser = argparse.ArgumentParser(description="MOME")

    # ----------------------- File Identification
    parser.add_argument('--suffix', type=str, default='default')

    # ----------------------- Device Setting
    parser.add_argument('--use_gpu', type=int, default=1)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--ckpt_start_epoch', type=int, default=0)

    # ------------------------ Training Setting
    parser.add_argument('--free_emb_dimension', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--eva_interval', type=int, default=10)
    parser.add_argument('--neg_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--early_stop', type=int, default=200)
    parser.add_argument('--topk', type=list, default=[10, 20, 30, 40, 50])
    parser.add_argument('--dataset', type=str, default='baby')

    # ----------------------- Regularizer coefficient
    parser.add_argument('--reg_coeff', type=float, default=1e-6)
    parser.add_argument('--penalty_coeff', type=float, default=50) 
    parser.add_argument('--align_coeff', type=float, default=1) 
    parser.add_argument('--moe_balance_coeff', type=float, default=1) 
    parser.add_argument('--fusion_balance_coeff', type=float, default=1) 


    # ----------------------- logger
    parser.add_argument('--log', type=int, default=0)
    parser.add_argument('--tensorboard', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)

    return parser.parse_args()



# ----------------------------------- Env Init -----------------------------------------------------------
tool.cprint('Init Env')
args = parse_args()
my_env = Env(args)
tool.cprint(f'---------- {my_env.args.suffix} ----------')
print(f'{my_env.args}')

# ----------------------------------- Dataset Init -----------------------------------------------------------

my_loader = Loader4MM(my_env)

tool.cprint('Init Dataset')

# ----------------------------------- Model Init -----------------------------------------------------------

my_model = MOME_model(my_env, my_loader)
if args.ckpt != None:
    my_model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
tool.cprint('Init Model')

# ----------------------------------- Session Init -----------------------------------------------------------

my_session = Session(my_env, my_model, my_loader)
tool.cprint('Init Session')

# ---------------------------------------- Main -----------------------------------------------------------

t = time.time()
my_session.train(my_env.args.epoch)
# my_session.save_memory()
# my_env.close_env()
tool.cprint(f'training stage cost time: {time.time() - t}')
if my_env.args.log:
    my_env.test_logger.info(f'--------- {my_env.args.suffix} best epoch {my_session.best_epoch}------------')

tool.cprint(f'--------- {my_env.args.suffix} best epoch {my_session.best_epoch}------------')
for top_k in args.topk:
    tool.cprint(f'hr@{top_k} = {my_session.test_hr[top_k]:.5f}, recall@{top_k} = {my_session.test_recall[top_k]:.5f}, ndcg@{top_k} = {my_session.test_ndcg[top_k]:.5f}')
    if my_env.args.log:
        my_env.test_logger.info(f'hr@{top_k} = {my_session.test_hr[top_k]:.5f}, recall@{top_k} = {my_session.test_recall[top_k]:.5f}, ndcg@{top_k} = {my_session.test_ndcg[top_k]:.5f}')
