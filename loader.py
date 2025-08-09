import os
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm


class Loader4MM(torch.utils.data.Dataset):
    def __init__(self, env):

        self.env = env

        self.n_user = 0
        self.m_item = 0

        train_file = os.path.join(self.env.DATA_PATH, 'train.txt')
        val_file = os.path.join(self.env.DATA_PATH, 'val.txt')
        test_file = os.path.join(self.env.DATA_PATH, 'test.txt')

        trainUniqueUsers, trainItem, trainUser =[], [], []
        valUniqueUsers, valItem, valUser =[], [], []
        testUniqueUsers, testItem, testUser =[], [], []

        self.traindataSize = 0
        self.testDataSize = 0

        self.train_data = defaultdict(list)
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    if l[1]=='':
                        continue
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.train_data[uid].extend(items)
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainItem = set(self.trainItem)

        self.cold_item_index = set()
        self.val_data = defaultdict(list)
        with open(val_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    self.n_user = max(self.n_user, uid)
                    if l[1] == '':
                        continue
                    else:
                        items = [int(i) for i in l[1:]]

                        for item in items:
                            if item not in self.trainItem:
                                self.cold_item_index.add(item)
                    self.val_data[uid].extend(items)
                    valUniqueUsers.append(uid)
                    valUser.extend([uid] * len(items))
                    valItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    # self.valDataSize += len(items)
        self.val_user_list = np.array(valUniqueUsers)

        self.test_data = defaultdict(list)
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    uid = int(l[0])
                    self.n_user = max(self.n_user, uid)
                    if l[1] == '':
                        continue
                    else:
                        items = [int(i) for i in l[1:]]
                        for item in items:
                            if item not in self.trainItem:
                                self.cold_item_index.add(item)
                    self.test_data[uid].extend(items)
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.testDataSize += len(items)
        self.cold_item_index = list(self.cold_item_index)
        self.m_item += 1
        self.n_user += 1

        self.test_user_list = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)


        # pre-calculate
        self._allPos = self.getUserAllItems()

        all_item_index = np.array(list(range(self.m_item)))
        self.cold_item_index = np.load(os.path.join(self.env.DATA_PATH, 'cold_item_index.npy'))
        self.warm_item_index = np.setdiff1d(all_item_index, self.cold_item_index, assume_unique=False)

        self.image_feat, self.text_feat, self.audio_feat = self.load_mutimedia_feature()
        self.feature = np.concatenate([self.image_feat, self.text_feat], axis=1)
        if self.env.args.dataset == 'tiktok':
            self.feature = np.concatenate([self.feature, self.audio_feat], axis=1)
            


    @property
    def allPos(self):
        return self._allPos

    def load_mutimedia_feature(self):
 
        image_file = os.path.join(self.env.DATA_PATH, 'image_feat.npy')
        text_file = os.path.join(self.env.DATA_PATH, 'text_feat.npy')

        # low_dim_feature_file = os.path.join(self.env.DATA_PATH, 'low_dim_feature.npy')
        image_feat = np.load(image_file)
        text_feat = np.load(text_file)
        if self.env.args.dataset == 'tiktok':
            audio_feat = np.load(audio_file)
        # low_dim_feature = np.load(low_dim_feature_file)
        if self.env.args.dataset == 'tiktok':
            return image_feat, text_feat, audio_feat
        else:
            return image_feat, text_feat, None


    def getUserAllItems(self):
        """
        得到训练集和验证集中用户交互过的item
        用于负采样
        """
        posItems = defaultdict(list)

        for user in list(self.train_data.keys()):
            posItems[user].extend(self.train_data[user])
        return posItems

    def neg_sample(self):
        self.pair_data = []
        print('generate samples ...')
        for user_id in tqdm(self.train_data.keys()):
            positive_list = self.train_data[user_id]  # self.train_dict[user_id]
            for item_i in positive_list:
                item_j = np.random.randint(self.m_item)
                while item_j in positive_list:
                    item_j = np.random.randint(self.m_item)
                self.pair_data.append([user_id, item_i, item_j])

    def neg_uniform_sample(self):
        user_num = len(self.trainUser)
        users = np.random.randint(0, self.n_user, user_num)
        self.pair_data = []
        print('generate uniform samples ...')
        for user_id in tqdm(users):
            positive_list = self.train_data[user_id]  # self.train_dict[user_id]
            if len(positive_list) == 0:
                continue
            posindex = np.random.randint(0, len(positive_list))
            item_i = positive_list[posindex]
            item_j = np.random.randint(self.m_item)
            while item_j in positive_list:
                item_j = np.random.randint(self.m_item)
            self.pair_data.append([user_id, item_i, item_j])

    def __getitem__(self, index):
        user = self.pair_data[index][0]
        pos_item = self.pair_data[index][1]
        neg_item = self.pair_data[index][2]
        return user, pos_item, neg_item

    def __len__(self):
        return len(self.trainItem)

def Uniform_PairSample(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    user_num = len(dataset.trainItem)
    users = np.random.randint(0, dataset.n_user, user_num)
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_item)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)

def PairSample(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    allPos = dataset.allPos
    S = []
    for i, user in enumerate(dataset.train_data.keys()):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        for positem in dataset.train_data[user]:
            while True:
                negitem = np.random.randint(0, dataset.m_item)
                if negitem in posForUser:
                    continue
                else:
                    break
            random_user = np.random.randint(0, dataset.n_user)
            random_item = np.random.randint(0, dataset.m_item)
            S.append([user, positem, negitem, random_user, random_item])
    return np.array(S)

def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)