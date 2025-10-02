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
        user_num = 212231
        item_num = 6596
    else:
        user_num = train_data["user"].max() + 1
        item_num = train_data["item"].max() + 1
    print("user, item num")
    print(user_num, item_num)
    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    train_features_list = []
    train_data_noisy = []
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0
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
    valid_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    valid_features_list = []
    for x in valid_data:
        valid_mat[x[0], x[1]] = 1.0
        valid_features_list.append([x[0], x[1]])

    train_valid_pos = {}
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

    train_pos = {}
    for x in train_features_list:
        if x[0] in train_pos:
            train_pos[x[0]].append(x[1])
        else:
            train_pos[x[0]] = [x[1]]

    valid_pos = {}
    for x in valid_features_list:
        if x[0] in valid_pos:
            valid_pos[x[0]].append(x[1])
        else:
            valid_pos[x[0]] = [x[1]]

    ################# load testing data #################
    test_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)

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
        train_pos,
        valid_pos,
        test_pos,
        train_valid_pos,
        user_num,
        item_num,
        train_mat,
        valid_mat,
        train_data_noisy,
    )


# construct the train and test datasets
class DenseMatrixUsers(data.Dataset):
    def __init__(self, user_index, dataset):
        super(DenseMatrixUsers, self).__init__()
        self.dataset = dataset
        self.user_index = user_index
        assert len(self.dataset) == len(self.user_index)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):

        user = self.user_index[index]
        user_interaction = self.dataset[index]

        return user, user_interaction
