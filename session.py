import os
import time
import random
import torch
from torch import autograd
from tqdm import tqdm
from collections import defaultdict

import numpy as np

import criterion
import tool
import loader
from evaluation import *
from tool import *
from criterion import *
from loader import *
import os
import time
import random
import torch
from torch import autograd
from tqdm import tqdm
from collections import defaultdict

import numpy as np



class Session(object):

    def __init__(self, env, model, loader):
        self.env = env
        self.model = model
        self.dataset = loader
        self.optimizer = torch.optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, self.model.parameters()), 'lr': self.env.args.lr}])
        self.bpr = BPR()
        self.align = MSE()
        self.early_stop = 0
        self.best_epoch = 0
        self.total_epoch = 0
        self.best_ndcg = defaultdict(float)
        self.best_hr = defaultdict(float)
        self.best_recall = defaultdict(float)
        self.test_ndcg = defaultdict(float)
        self.test_hr = defaultdict(float)
        self.test_recall = defaultdict(float)
        
    def train_epoch(self):
        t = time.time()
        self.model.train()
        self.total_epoch += 1
        S = PairSample(self.dataset)
        users = torch.Tensor(S[:, 0]).long()
        posItems = torch.Tensor(S[:, 1]).long()
        negItems = torch.Tensor(S[:, 2]).long()
        users = users.to(self.env.device)
        posItems = posItems.to(self.env.device)
        negItems = negItems.to(self.env.device)
        users, posItems, negItems = shuffle(users, posItems, negItems)
        total_batch = len(users) // self.env.args.batch_size + 1
        all_loss, all_bpr_loss, all_reg_loss, all_moe_balance_loss, all_fusion_balance_loss = 0., 0., 0., 0., 0.
        all_rbpr_loss, all_align_loss, all_grad_loss = 0., 0., 0.

        for user, pos_item, neg_item, in minibatch(users, posItems, negItems, batch_size=self.env.args.batch_size):
            
            user_emb, item_emb, text_emb, image_emb, moe_balance_loss, fusion_balance_loss = self.model()
            align_loss = self.align(item_emb, text_emb) + self.align(item_emb, image_emb)

            bpr_loss, reg_loss = self.bpr(user_emb, item_emb, user, pos_item, neg_item)
            
            align_loss = self.env.args.align_coeff * align_loss
            reg_loss = self.env.args.reg_coeff * reg_loss
            moe_balance_loss = self.env.args.moe_balance_coeff * moe_balance_loss
            fusion_balance_loss = self.env.args.fusion_balance_coeff * fusion_balance_loss
            loss = bpr_loss + reg_loss + align_loss + moe_balance_loss + fusion_balance_loss 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            all_loss += loss
            all_bpr_loss += bpr_loss
            all_reg_loss += reg_loss
            all_align_loss += align_loss
            all_moe_balance_loss += moe_balance_loss
            all_fusion_balance_loss += fusion_balance_loss
        return all_loss / total_batch, all_bpr_loss / total_batch, all_reg_loss / total_batch, all_align_loss/total_batch, all_moe_balance_loss/total_batch, all_fusion_balance_loss/total_batch, time.time() - t
 
    def train(self, epochs):
        for epoch in range(self.env.args.ckpt_start_epoch, epochs):
            loss, bpr_loss, reg_loss, align_loss, moe_balance_loss, fusion_balance_loss, train_time = self.train_epoch()
            print('-' * 30)
            print(
                f'TRAIN:epoch = {epoch}/{epochs} loss_s1 = {loss:.5f}, bpr_loss_s1 = {bpr_loss:.5f}, align_loss = {align_loss:.5f}, moe_balance_loss = {moe_balance_loss:.5f}, fusion_balance_loss = {fusion_balance_loss:.5f}, train_time = {train_time:.2f}')


            if epoch % self.env.args.eva_interval == 0:
                self.early_stop += self.env.args.eva_interval
                hr, recall, ndcg, val_time = self.test(mode='val', top_list=self.env.args.topk)

                if self.env.args.tensorboard:
                    for key in hr.keys():
                        self.env.w.add_scalar(
                            f'Val/hr@{key}', hr[key], self.total_epoch)
                        self.env.w.add_scalar(
                            f'Val/recall@{key}', hr[key], self.total_epoch)
                        self.env.w.add_scalar(
                            f'Val/ndcg@{key}', ndcg[key], self.total_epoch)
                key = list(hr.keys())[0]
                print(
                    f'epoch = {epoch} hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time:.2f}')

                if ndcg[list(hr.keys())[0]] > self.best_ndcg[list(hr.keys())[0]]:
                    thr, trecall, tndcg, test_time = self.test(mode='test', top_list=self.env.args.topk)
                    self.early_stop = 0
                    for key in thr.keys():
                        cprint(
                            f'epoch = {epoch} hr@{key} = {thr[key]:.5f}, recall@{key} = {trecall[key]:.5f}, ndcg@{key} = {tndcg[key]:.5f}, test_time = {test_time:.2f}')
                    cprint('----------------------')

                    for key in hr.keys():
                        self.best_hr[key] = hr[key]
                        self.best_recall[key] = recall[key]
                        self.best_ndcg[key] = ndcg[key]
                    for key in thr.keys():
                        self.test_hr[key] = thr[key]
                        self.test_recall[key] = trecall[key]
                        self.test_ndcg[key] = tndcg[key]
                    if self.env.args.save:
                        self.save_model(epoch)
                        print('save ckpt')
                    self.best_epoch = epoch
                    if self.env.args.log:
                        self.env.val_logger.info(f'EPOCH[{epoch}/{epochs}]')
                        for key in hr.keys():
                            self.env.val_logger.info(
                                f'hr@{key} = {hr[key]:.5f}, recall@{key} = {recall[key]:.5f}, ndcg@{key} = {ndcg[key]:.5f}, val_time = {val_time:.2f}')

            if self.env.args.log:
                self.env.train_logger.info(
                    f'EPOCH[{epoch}/{epochs}], loss = {loss:.5f}, bpr_loss = {bpr_loss:.5f}, reg_loss = {reg_loss:.5f}')

            if self.env.args.tensorboard:
                self.env.w.add_scalar(f'Train/loss', loss, self.total_epoch)
                self.env.w.add_scalar(
                    f'Train/bpr_loss', bpr_loss, self.total_epoch)
                self.env.w.add_scalar(
                    f'Train/reg_loss', reg_loss, self.total_epoch)

            if self.early_stop > self.env.args.early_stop // 1:
                break


    def test(self, mode='val', top_list=[50]):
        self.model.eval()
        t = time.time()
        user_emb, item_emb, _, _, _, _ = self.model()
        user_emb = user_emb.cpu().detach().numpy()
        item_emb = item_emb.cpu().detach().numpy()
        if mode == 'val':
            hr, recall, ndcg = num_faiss_evaluate(self.dataset.val_data,
                                                        list(
                                                            self.dataset.val_data.keys()),
                                                        list(
                                                            self.dataset.cold_item_index),
                                                        self.dataset.train_data,
                                                        top_list, user_emb, item_emb)
        else:
            hr, recall, ndcg = num_faiss_evaluate(self.dataset.test_data,
                                                             list(
                                                                     self.dataset.test_data.keys()),
                                                            list(
                                                                self.dataset.cold_item_index),
                                                             self.dataset.train_data,
                                                             top_list, user_emb, item_emb)

        return hr, recall, ndcg, time.time() - t

    def save_ckpt(self, path):
        torch.save(self.model.state_dict(), path)

    def save_model(self, current_epoch):
        model_state_file = os.path.join(
            self.env.CKPT_PATH, f'{self.env.args.suffix}_{self.env.args.penalty_coeff}_epoch{current_epoch}.pth')
        self.save_ckpt(model_state_file)
        if self.best_epoch is not None and current_epoch != self.best_epoch:
            old_model_state_file = os.path.join(
                self.env.CKPT_PATH, f'{self.env.args.suffix}_{self.env.args.penalty_coeff}_epoch{current_epoch}.pth')
            if os.path.exists(old_model_state_file):
                os.system('rm {}'.format(old_model_state_file))
