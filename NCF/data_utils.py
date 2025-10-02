import numpy as np
import pandas as pd
import scipy.sparse as sp
from copy import deepcopy
import random
import torch.utils.data as data


def load_all(dataset, data_path):

    train_rating = data_path + "{}.train.rating".format(dataset)
    valid_rating = data_path + "{}.valid.rating".format(dataset)
    test_negative = data_path + "{}.test.negative".format(dataset)

    ################# load training data #################
    train_data = pd.read_csv(
        train_rating,
        sep="\t",
        header=None,
        names=["user", "item", "noisy"],
        usecols=[0, 1, 2],
        dtype={0: np.int32, 1: np.int32, 2: np.int32},
    )

    if dataset == "adressa":
        users_num = 212231
        items_num = 6596
    else:
        users_num = train_data["user"].max() + 1
        items_num = train_data["item"].max() + 1
    print("user, item num")
    print(users_num, items_num)
    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((users_num, items_num), dtype=np.float32)
    train_valid_mat = sp.dok_matrix((users_num, items_num), dtype=np.float32)
    train_features_list = []
    train_data_noisy = []
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
        train_valid_mat[x[0], x[1]] = 1.0
        train_features_list.append([x[0], x[1]])
        train_data_noisy.append(x[2])

    ################# load validation data #################
    valid_data = pd.read_csv(
        valid_rating,
        sep="\t",
        header=None,
        names=["user", "item", "noisy"],
        usecols=[0, 1, 2],
        dtype={0: np.int32, 1: np.int32, 2: np.int32},
    )
    valid_data = valid_data.values.tolist()
    valid_features_list = []
    for x in valid_data:
        train_valid_mat[x[0], x[1]] = 1.0
        valid_features_list.append([x[0], x[1]])

    train_valid_pos = {}
    valid_pos = {}
    for x in train_features_list:
        if x[0] in train_valid_pos:
            train_valid_pos[x[0]].append(x[1])
        else:
            train_valid_pos[x[0]] = [x[1]]

    for x in valid_features_list:
        if x[0] in train_valid_pos:
            train_valid_pos[x[0]].append(x[1])
        else:
            train_valid_pos[x[0]] = [x[1]]
        if x[0] in valid_pos:
            valid_pos[x[0]].append(x[1])
        else:
            valid_pos[x[0]] = [x[1]]

    ################# load testing data #################
    test_mat = sp.dok_matrix((users_num, items_num), dtype=np.float32)

    test_pos = {}
    with open(test_negative, "r") as fd:
        line = fd.readline()
        while line != None and line != "":
            arr = line.split("\t")
            if dataset == "adressa":
                u = eval(arr[0])[0]
                i = eval(arr[0])[1]
            else:
                u = int(arr[0])
                i = int(arr[1])
            if u in test_pos:
                test_pos[u].append(i)
            else:
                test_pos[u] = [i]
            test_mat[u, i] = 1.0
            line = fd.readline()

    return (
        train_features_list,
        valid_features_list,
        test_pos,
        valid_pos,
        train_valid_pos,
        users_num,
        items_num,
        train_mat,
        train_valid_mat,
        train_data_noisy,
    )


class NCFData(data.Dataset):
    def __init__(
        self,
        features,
        items_num,
        train_mat=None,
        neg_num_per_pos=0,
        phase=0,
        is_not_noisy=None,
    ):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
			phase: 0 for training, 1 for validation, 2 for testing
		"""
        self.features_pos = features
        if phase == 0:
            self.is_not_noisy = is_not_noisy
        else:
            self.is_not_noisy = [0 for _ in range(len(features))]
        self.item_num = items_num
        self.train_mat = train_mat
        self.neg_num_per_pos = neg_num_per_pos
        self.phase = phase
        self.labels = [1 for _ in range(len(features))]

    def neg_sample(self):
        assert self.phase != 2, "no need to sampling when testing"

        self.features_neg = []
        for x in self.features_pos:
            u = x[0]
            for _ in range(self.neg_num_per_pos):
                j = np.random.randint(self.item_num)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.item_num)
                self.features_neg.append([u, j])

        labels_pos = [1 for _ in range(len(self.features_pos))]
        labels_neg = [0 for _ in range(len(self.features_neg))]
        self.is_not_noisy_fill = self.is_not_noisy + [
            1 for _ in range(len(self.features_neg))
        ]
        self.features_fill = self.features_pos + self.features_neg
        assert len(self.is_not_noisy_fill) == len(self.features_fill)
        self.labels_fill = labels_pos + labels_neg

    def __len__(self):
        return (self.neg_num_per_pos + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.phase != 2 else self.features_pos
        labels = self.labels_fill if self.phase != 2 else self.labels
        is_not_noises = self.is_not_noisy_fill if self.phase != 2 else self.is_not_noisy

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        is_not_noisy = is_not_noises[idx]

        return user, item, label, is_not_noisy
