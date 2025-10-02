import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_


class NCF(nn.Module):
    def __init__(
        self,
        users_num,
        items_num,
        factors_num,
        layers_num,
        dropout,
        model_name,
        GMF_model=None,
        MLP_model=None,
    ):
        super(NCF, self).__init__()
        """
        users_num: number of users;
        items_num: number of items;
        factors_num: number of predictive factors;
        layers_num: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
        GMF_model: pre-trained GMF weights;
        MLP_model: pre-trained MLP weights.
        """
        self.dropout = dropout
        self.model_name = model_name
        self.GMF_model = GMF_model
        self.MLP_model = MLP_model

        self.embed_user_GMF = nn.Embedding(users_num, factors_num)
        self.embed_item_GMF = nn.Embedding(items_num, factors_num)
        self.embed_user_MLP = nn.Embedding(
            users_num, factors_num * (2 ** (layers_num - 1))
        )
        self.embed_item_MLP = nn.Embedding(
            items_num, factors_num * (2 ** (layers_num - 1))
        )

        MLP_modules = []
        for i in range(layers_num):
            input_size = factors_num * (2 ** (layers_num - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model_name in ["MLP", "GMF"]:
            predict_size = factors_num
        else:
            predict_size = factors_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)

        self._init_weight()

    def _init_weight(self):
        """We leave the weights initialization here."""
        if self.model_name != "NeuMF-pre":
            nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
            nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
            nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            nn.init.kaiming_uniform_(
                self.predict_layer.weight, a=1, nonlinearity="sigmoid"
            )

            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)

            # mlp layers
            for m1, m2 in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            # predict layers
            predict_weight = torch.cat(
                [
                    self.GMF_model.predict_layer.weight,
                    self.MLP_model.predict_layer.weight,
                ],
                dim=1,
            )
            precit_bias = (
                self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias
            )

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * precit_bias)

    def forward(self, user, item):
        if self.model_name != "MLP":
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if self.model_name != "GMF":
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat(
                (embed_user_MLP, embed_item_MLP), -1
            )  # stack by to back
            output_MLP = self.MLP_layers(interaction)

        if self.model_name == "GMF":
            concat = output_GMF
        elif self.model_name == "MLP":
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

