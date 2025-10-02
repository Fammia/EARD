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
parser.add_argument("--name", type=str, default="", help="name")
args = parser.parse_args()

print(f"args: {args}")


def get_gpu_id_str_sort_by_mem():
    # Sort GPUs by memory, from largest to smallest available memory
    import subprocess

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
    )
    output = result.stdout.decode("utf-8").strip()

    gpu_info = []
    for line in output.split("\n"):
        index, free_memory = line.split(",")
        gpu_info.append((int(index), int(free_memory)))

    gpu_info_sorted = sorted(gpu_info, key=lambda x: x[1], reverse=True)
    gpu_ids_sorted = [str(gpu[0]) for gpu in gpu_info_sorted]

    res = ",".join(gpu_ids_sorted)
    print("GPU IDs sorted by free memory:", res)
    return res


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = get_gpu_id_str_sort_by_mem()

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

# Create model save path
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
train_user_interaction_cnts = torch.zeros_like(train_mat_dense.sum(1))
train_item_interaction_cnts = torch.zeros_like(train_mat_dense.sum(0))


# Linear weight allocation - Principle: Linear weight allocation based on user loss ranking
def get_weight(avg_loss, startValue, endValue):
    # Sort average losses and get sorting indices
    sorted_indices = torch.argsort(avg_loss)

    # Create linearly distributed weights
    weights = torch.linspace(startValue, endValue, steps=len(avg_loss))

    # Reassign weights based on sorting results
    result = torch.zeros_like(avg_loss)
    result[sorted_indices] = weights

    return result


########################### Temp Scaled Weight Base #####################################
def get_based_weight(loss_values):
    """
    Weight calculation based on Cumulative Distribution Function (CDF) - Sparse implementation
    Theoretical basis: Weight allocation based on the position of loss values in the empirical distribution

    Args:
        loss_values: List of loss values, already in -loss form

    Returns:
        User index list, item index list, weight value list
    """
    if len(loss_values) == 0:
        return torch.ones_like(loss_values)

    # Sort loss values
    sorted_losses, _ = torch.sort(loss_values)

    # Calculate the position (rank) of each loss value in the sorted array
    ranks = torch.searchsorted(sorted_losses, loss_values, right=True).float()

    # Calculate CDF values using Weibull plotting position: (rank - 0.5) / n
    n_samples = len(loss_values)
    cdf_weights = (ranks - 0.5) / n_samples

    return cdf_weights


########################### Eval Phase Definition #####################################
patience_count, best_hr = 0, 0  # Early stopping counter, best HR
best_recall = 0.0  # Best Recall
best_epoch = None  # Epoch corresponding to the best model
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


ui_weight_dict = {}
loss_dict = {}

for epoch in range(1000):
    ncf_model.train()  # Enable dropout (if have).
    train_loss = 0

    train_loader.dataset.neg_sample()  # negative sampling is done here

    # Use sparse storage instead of dense matrices
    # Use dictionary to store user-item pairs and their loss values, ensuring no duplicates
    loss_dict = {}

    train_user_interaction_cnts.zero_()
    train_item_interaction_cnts.zero_()

    user_epoch_loss_sum = torch.zeros_like(train_user_interaction_cnts)
    item_epoch_loss_sum = torch.zeros_like(train_item_interaction_cnts)

    for user, item, label, is_not_noisy in train_loader:
        user = user.cuda()
        item = item.cuda()
        label = label.float().cuda()
        # is_not_noisy = is_not_noisy.float().cuda()

        user_cpu = user.cpu()
        item_cpu = item.cpu()

        train_user_interaction_cnts[user_cpu] += 1
        train_item_interaction_cnts[item_cpu] += 1

        ncf_model.zero_grad()
        prediction = ncf_model(user, item)
        loss = BCE_loss(prediction, label)

        # for user level
        user_epoch_loss_sum[user_cpu] += loss.cpu()
        # for item level
        item_epoch_loss_sum[item_cpu] += loss.cpu()

        with torch.no_grad():
            if epoch <= 1:
                mul_factor = torch.ones_like(loss)
            else:
                # Get weights from sparse storage - efficient batch processing
                batch_size = user.size(0)
                batch_mul_factor = torch.ones(batch_size, device=loss.device)

                # Get indices on CPU
                user_list = user_cpu.tolist()
                item_list = item_cpu.tolist()

                # Batch lookup weights
                for idx in range(batch_size):
                    key = (user_list[idx], item_list[idx])
                    if key in ui_weight_dict:
                        batch_mul_factor[idx] = ui_weight_dict[key]

                mul_factor = batch_mul_factor

        user_list = user_cpu.tolist()
        item_list = item_cpu.tolist()
        loss_list = loss.cpu().tolist()

        for u, i, l in zip(user_list, item_list, loss_list):
            loss_dict[(u, i)] = l

        batch_loss = torch.mean(mul_factor * loss)

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

    user_avg_loss = user_epoch_loss_sum / train_user_interaction_cnts
    item_avg_loss = item_epoch_loss_sum / train_item_interaction_cnts

    user_fact = get_weight(
        avg_loss=user_avg_loss,
        startValue=args.factor_upper,
        endValue=args.factor_lower,
    )
    item_fact = get_weight(
        avg_loss=item_avg_loss,
        startValue=args.factor_upper,
        endValue=args.factor_lower,
    )

    loss_users = []
    loss_items = []
    loss_values = []

    for (u, i), l in loss_dict.items():
        loss_users.append(u)
        loss_items.append(i)
        loss_values.append(l)

    loss_users_tensor = torch.tensor(loss_users)
    loss_items_tensor = torch.tensor(loss_items)
    loss_values_tensor = torch.tensor(loss_values)

    # Calculate user-item interaction level weights - using sparse implementation
    cdf_weights = get_based_weight(-loss_values_tensor)

    # Create new weight dictionary
    ui_weight_dict = {}

    for u, i, w in zip(
        loss_users_tensor.tolist(), loss_items_tensor.tolist(), cdf_weights.tolist()
    ):
        combined_weight = w * user_fact[u].item() * item_fact[i].item()
        ui_weight_dict[(u, i)] = combined_weight


print("############################## Training End. ##############################")

print(f"Whole Cost time: {(time.time()-startTime) / 60:.2f} min")
print(f"args: {args}")

model_save_path = os.path.join(
    model_path,
    f"{args.model}_{args.dataset}_lower-{args.factor_lower}_upper-{args.factor_upper}.pth",
)

if best_model_state is not None:
    torch.save(best_model_state, model_save_path)
    print(f"Best model found in epoch {best_epoch}")
    print(f"Best model saved to: {model_save_path}")

    ncf_model.load_state_dict(best_model_state)
    ncf_model.cuda()
    test_model = ncf_model

########################### Logs Phase #####################################

train_mat_dense = torch.tensor(train_valid_mat.toarray()).cpu()
test(test_model, test_pos, train_mat_dense)
