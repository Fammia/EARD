import os
import time
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import copy

from tqdm import tqdm
import toolz
import model
import evaluate
import data_utils

import configargparse

parser = configargparse.ArgumentParser(
    description="Training script", default_config_files=["./config.conf"]
)

parser.add_argument(
    "-c",
    "--config",
    default="./config.conf",
    is_config_file=True,
    help="Path to config file",
)
parser.add_argument(
    "--dataset",
    type=str,
    help="dataset used for training, options: amazon_book, yelp, movielens",
    default="movielens",
)
parser.add_argument("--factor_lower", type=float, default=0.0, help="factor lower")
parser.add_argument("--factor_upper", type=float, default=0.5, help="factor upper")
parser.add_argument(
    "--seed", type=int, help="seed for reproducibility", default=123456789
)
parser.add_argument(
    "--epoch_eval", type=int, default=10, help="epoch to start evaluation"
)
parser.add_argument(
    "--batch_size", type=int, default=2048, help="epoch to start evaluation"
)
parser.add_argument(
    "--top_k", type=int, nargs="+", default=[50, 100], help="compute metrics @k"
)
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")

parser.add_argument(
    "--eval_interval", type=int, default=5, help="Interval for evaluation"
)
parser.add_argument("--patience", type=int, default=5, help="patience")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu_id")
args = parser.parse_args()

print(f"args: {args}")
print("In this program, 'factor_lower' and 'factor_upper' are ineffective because they are not used.")


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

torch.manual_seed(args.seed)  # cpu
torch.cuda.manual_seed(args.seed)  # gpu
np.random.seed(args.seed)  # numpy
random.seed(args.seed)  # random and transforms
cudnn.benchmark = True
torch.backends.cudnn.deterministic = True  # cudnn


def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


data_path = f"../data/{args.dataset}/"
model_path = f"./models/"

# Create the directory to save models
os.makedirs(model_path, exist_ok=True)

############################## PREPARE DATASET ##########################

(
    train_features_list,
    valid_features_list,
    train_pos,
    valid_pos,
    test_pos,
    train_valid_pos,
    users_num,
    items_num,
    train_mat,
    valid_mat,
    train_data_noisy,
) = data_utils.load_all(f"{args.dataset}", data_path)


train_mat_dense = train_mat.toarray()
users_list = np.array([i for i in range(users_num)])
train_dataset = data_utils.DenseMatrixUsers(users_list, train_mat_dense)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)

valid_mat_dense = valid_mat.toarray()
valid_dataset = data_utils.DenseMatrixUsers(users_list, valid_mat_dense)
valid_loader = data.DataLoader(
    valid_dataset,
    batch_size=args.batch_size,
    # batch_size=4096,
    shuffle=True,
    num_workers=args.num_workers,
)

########################### CREATE MODEL #################################

cdae_model = model.CDAE(
    users_num=users_num, items_num=items_num, hidden_dim=32, dropout=0.2
)
cdae_model.cuda()
BCE_loss = nn.BCEWithLogitsLoss(reduction="none")
num_ns = 1  # negative samples, means one negative sample for each positive sample
optimizer = optim.Adam(cdae_model.parameters(), lr=0.001)


########################### Eval #####################################
patience_count, best_hr = 0, 0  # Early stopping counter, best HR
best_recall = 0.0  # Best Recall
best_epoch = None  # The epoch corresponding to the best model
best_model_state = None  # Save the best model's state_dict in memory


def eval(model, valid_loader, valid_data_pos, train_mat, epoch):
    global best_recall, patience_count, best_epoch, best_model_state

    top_k = args.top_k
    model.eval()
    # model prediction can be more efficient instead of looping through each user, do it by batch
    predictedIndices_all = torch.empty(
        users_num, top_k[-1], dtype=torch.long
    )  # predictions
    GroundTruth = list(valid_data_pos.values())  # ground truth is exact item indices
    for user_valid, data_value_valid in valid_loader:
        with torch.no_grad():
            user_valid = user_valid.cuda()
            prediction_input_from_train = torch.tensor(
                train_mat[user_valid.cpu()]
            ).cuda()
            prediction = model(
                user_valid, prediction_input_from_train
            )  # prediction of the batch from train matrix
            valid_data_mask = (
                train_mat[user_valid.cpu()] * -9999
            )  # depends on the size of data

            prediction = prediction + torch.tensor(valid_data_mask).float().cuda()
            _, indices = torch.topk(prediction, top_k[-1])
            predictedIndices_all[user_valid.cpu()] = indices.cpu()

    predictedIndices = predictedIndices_all[list(valid_data_pos.keys())]
    precision, recall, NDCG, MRR = evaluate.compute_acc(
        GroundTruth, predictedIndices, top_k
    )

    print("################### EVAL ######################")
    print(f"Recall:{recall} NDCG: {NDCG}")

    if recall[0] > best_recall:
        best_recall = recall[0]
        best_epoch = epoch
        patience_count = 0
        # Save the best model's state_dict in memory to avoid frequent disk I/O
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"New best model found! Recall: {best_recall:.4f} at epoch {epoch}")
    else:
        patience_count += 1


########################### Test #####################################
def test(model, test_data_pos, train_mat, valid_mat):
    top_k = args.top_k
    model.eval()
    predictedIndices = []  # predictions
    GroundTruth = list(test_data_pos.values())

    for users in toolz.partition_all(
        1000, list(test_data_pos.keys())
    ):  # looping through users in test set
        user_id = torch.tensor(list(users)).cuda()
        data_value_test = torch.tensor(train_mat[list(users)]).cuda()
        predictions = model(user_id, data_value_test)  # model prediction for given data
        test_data_mask = (train_mat_dense[list(users)] + valid_mat[list(users)]) * -9999

        predictions = predictions + torch.tensor(test_data_mask).float().cuda()
        _, indices = torch.topk(
            predictions, top_k[-1]
        )  # returns sorted index based on highest probability
        indices = indices.cpu().numpy().tolist()
        predictedIndices += indices  # a list of top 100 predicted indices

    precision, recall, NDCG, MRR = evaluate.compute_acc(
        GroundTruth, predictedIndices, top_k
    )
    print("################### TEST ######################")
    print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
    print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))


########################### Training #####################################
startTime = time.time()
top_k = args.top_k
# Initialize user and item weight factors, size is user_num * item_num
ui_factor = torch.ones((users_num, items_num))
for epoch in range(1000):
    cdae_model.train()
    train_loss = 0

    for user, pos_interation_mat in train_loader:
        user = user.cuda()
        pos_interation_mat = pos_interation_mat.cuda()
        prediction = cdae_model(user, pos_interation_mat)
        # negative sampling
        with torch.no_grad():
            num_ns_per_user = pos_interation_mat.sum(1) * num_ns
            negative_samples = []
            idxs = []
            for idx in range(pos_interation_mat.size(0)):
                neg_item_list = torch.randint(
                    0, items_num, (int(num_ns_per_user[idx].item()),)
                )
                negative_samples.append(neg_item_list)
                idxs.extend([idx] * int(num_ns_per_user[idx].item()))

        # Create a mask to filter out valid samples; 1 indicates a valid sample, 0 indicates an invalid sample
        # This is equivalent to including negative samples in addition to the positives
        negative_samples = torch.cat(negative_samples, 0)
        idxs = torch.LongTensor(idxs)
        pos_neg_interaction_mat_mask = pos_interation_mat.clone()
        pos_neg_interaction_mat_mask[idxs, negative_samples] = 1

        # Get the labels and predictions for positive and negative samples
        groundtruth = pos_interation_mat[pos_neg_interaction_mat_mask > 0.0]
        pred = prediction[pos_neg_interaction_mat_mask > 0.0]

        loss = BCE_loss(pred, groundtruth)

        # calculating loss
        batch_loss = torch.mean(loss)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss

    print(f"Epoch: {epoch} Train loss: {train_loss}")

    if epoch >= args.epoch_eval and epoch % args.eval_interval == 0:
        # validation
        eval(cdae_model, valid_loader, valid_pos, train_mat_dense, epoch)

        if patience_count == args.patience:
            break

print("############################## Training End. ##############################")
print(f"Whole Cost time: {(time.time()-startTime) / 60:.2f} min")


########################### Test #####################################
cdae_model.load_state_dict(best_model_state)
cdae_model.cuda()

test(cdae_model, test_pos, train_mat_dense, valid_mat_dense)

# Save the best model

model_save_path = os.path.join(
    model_path,
    f"CDAE_{args.dataset}_vanilla.pth",
)

torch.save(best_model_state, model_save_path)
print(f"Best model saved to: {model_save_path}")
print(f"Best model found in epoch {best_epoch}")