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

from tqdm import tqdm
import toolz
import model
import evaluate
import data_utils

import configargparse

import matplotlib.pyplot as plt
import seaborn as sns

parser = configargparse.ArgumentParser(
    description="Training script", default_config_files=["./config_.conf"]
)

parser.add_argument(
    "-c",
    "--config",
    default="./config.conf",
    is_config_file=True,
    help="Path to config file",
)
parser.add_argument(
    "--model",
    type=str,
    help="model used for training. options: GMF, NeuMF-end",
    default="GMF",
)
parser.add_argument(
    "--dataset",
    type=str,
    help="dataset used for training, options: amazon_book, yelp, movielens",
    default="yelp",
)
parser.add_argument("--factor_lower", type=float, default=0.0, help="factor lower")
parser.add_argument("--factor_upper", type=float, default=1.0, help="factor upper")

parser.add_argument(
    "--seed", type=int, help="seed for reproducibility", default=123456789
)
parser.add_argument(
    "--epoch_eval", type=int, default=10, help="epoch to start evaluation"
)
parser.add_argument(
    "--top_k", type=int, nargs="+", default=[50, 100], help="compute metric @topk"
)
parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
parser.add_argument(
    "--eval_interval", type=int, default=1, help="Interval for evaluation"
)
parser.add_argument("--patience", type=int, default=10, help="patience")
parser.add_argument("--gpu_id", type=int, default=0, help="gpu_id")
args = parser.parse_args()

print(f"args: {args}")
print(
    "In this program, 'factor_lower' and 'factor_upper' are ineffective because they are not used."
)

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
    test_pos,
    valid_pos,
    train_valid_pos,
    users_num,
    items_num,
    train_mat,
    train_valid_mat,
    train_data_noisy,
) = data_utils.load_all(f"{args.dataset}", data_path)

# construct the train and test datasets
train_dataset = data_utils.NCFData(
    features=train_features_list,
    items_num=items_num,
    train_mat=train_mat,
    neg_num_per_pos=1,
    phase=0,
    is_not_noisy=train_data_noisy,
)

valid_dataset = data_utils.NCFData(
    features=valid_features_list,
    items_num=items_num,
    train_mat=train_mat,
    neg_num_per_pos=1,
    phase=1,
    is_not_noisy=None,
)

train_loader = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    worker_init_fn=worker_init_fn,
)

valid_loader = data.DataLoader(
    valid_dataset,
    batch_size=2048,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    worker_init_fn=worker_init_fn,
)

print(
    "data loaded! users_num:{}, items_num:{} train_data_len:{} test_user_num:{}".format(
        users_num, items_num, len(train_features_list), len(test_pos)
    )
)

########################### CREATE MODEL #################################

ncf_model = model.NCF(
    users_num=users_num,
    items_num=items_num,
    factors_num=32,
    layers_num=3,
    dropout=0.0,
    model_name=f"{args.model}",
    GMF_model=None,
    MLP_model=None,
)

ncf_model.cuda()
BCE_loss = nn.BCEWithLogitsLoss(reduction="none")
optimizer = optim.Adam(ncf_model.parameters(), lr=0.001)
train_mat_dense = torch.tensor(train_mat.toarray()).cpu()


########################### Eval Phase Definition #####################################
patience_count, best_hr = 0, 0  # Early stopping counter, best HR
best_recall = 0.0  # Best Recall
best_epoch = None  # The epoch corresponding to the best model
best_model_state = None  # Save the best model's state_dict in memory


def eval(model, valid_pos, mat, epoch):
    global best_recall, patience_count, best_epoch, best_model_state

    top_k = args.top_k
    model.eval()
    predictedIndices = []
    GroundTruth = []
    users_in_valid = list(valid_pos.keys())
    for users_valid in toolz.partition_all(15, users_in_valid):
        users_valid = list(users_valid)
        GroundTruth.extend([valid_pos[u] for u in users_valid])
        users_valid_torch = (
            torch.tensor(users_valid).repeat_interleave(items_num).cuda()
        )
        items_full = (
            torch.tensor([i for i in range(items_num)]).repeat(len(users_valid)).cuda()
        )
        prediction = model(users_valid_torch, items_full)
        _, indices = torch.topk(
            prediction.view(len(users_valid), -1) + mat[users_valid].cuda() * -9999,
            max(top_k),
        )
        indices = indices.cpu().numpy().tolist()
        predictedIndices.extend(indices)
    precision, recall, NDCG, MRR = evaluate.compute_acc(
        GroundTruth, predictedIndices, top_k
    )
    epoch_recall = recall[0]

    print("################### EVAL ######################")
    print(f"Recall:{recall} NDCG: {NDCG}")

    if epoch_recall > best_recall:
        best_recall = epoch_recall
        best_epoch = epoch
        patience_count = 0
        # Save the best model's state_dict in memory to avoid frequent disk I/O
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"New best model found! Recall: {best_recall:.4f} at epoch {epoch}")
    else:
        patience_count += 1


########################### Test Phase Definition #####################################
def test(model, test_data_pos, mat):
    top_k = args.top_k
    model.eval()
    predictedIndices = []
    GroundTruth = []
    users_in_test = list(test_data_pos.keys())
    for users_test in toolz.partition_all(15, users_in_test):
        users_test = list(users_test)
        GroundTruth.extend([test_data_pos[u] for u in users_test])
        users_test_torch = torch.tensor(users_test).repeat_interleave(items_num).cuda()
        items_full = (
            torch.tensor([i for i in range(items_num)]).repeat(len(users_test)).cuda()
        )
        prediction = model(users_test_torch, items_full)
        _, indices = torch.topk(
            prediction.view(len(users_test), -1) + mat[users_test].cuda() * -9999,
            max(top_k),
        )
        indices = indices.cpu().numpy().tolist()
        predictedIndices.extend(indices)
    precision, recall, NDCG, MRR = evaluate.compute_acc(
        GroundTruth, predictedIndices, top_k
    )

    print("################### TEST ######################")
    print("Recall {:.4f}-{:.4f}".format(recall[0], recall[1]))
    print("NDCG {:.4f}-{:.4f}".format(NDCG[0], NDCG[1]))


########################### Training Phase #####################################
startTime = time.time()
top_k = args.top_k
for epoch in range(1000):
    ncf_model.train()  # Enable dropout (if have).
    train_loss = 0

    train_loader.dataset.neg_sample()  # negative sampling is done here

    for user, item, label, _ in train_loader:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()

        ncf_model.zero_grad()
        prediction = ncf_model(user, item)
        loss = BCE_loss(prediction, label)

        batch_loss = torch.mean(loss)

        batch_loss.backward()
        optimizer.step()
        train_loss += batch_loss

    print("epoch: {}, loss:{}".format(epoch, train_loss))

    if epoch >= args.epoch_eval and epoch % args.eval_interval == 0:
        eval(
            model=ncf_model,
            valid_pos=valid_pos,
            mat=train_mat_dense,
            epoch=epoch,
        )

    if patience_count == args.patience:
        break


print("############################## Training End. ##############################")

print(f"Whole Cost time: {(time.time()-startTime) / 60:.2f} min")
print(f"args: {args}")

# Save the best model to disk once training is complete
model_save_path = os.path.join(
    model_path,
    f"{args.model}_{args.dataset}_vanilla.pth",
)

if best_model_state is not None:
    # Save the best model from memory
    torch.save(best_model_state, model_save_path)
    print(f"Best model found in epoch {best_epoch}")
    print(f"Best model saved to: {model_save_path}")

    # Create and load the best model for testing
    ncf_model.load_state_dict(best_model_state)
    ncf_model.cuda()
    test_model = ncf_model

########################### Logs Phase #####################################

train_mat_dense = torch.tensor(train_valid_mat.toarray()).cpu()
test(test_model, test_pos, train_mat_dense)