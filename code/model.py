import utils
import math
import torch
import dataloader
from dataloader import BasicDataset
from torch import nn
import torch.nn.functional as F
from typing import *
import numpy as np


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users, all_users, all_items):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class MMoE(nn.Module):
    def __init__(self, hidden_dim, num_tasks, num_experts, input_dim, expert_size=1,
                 use_expert_bias=True, use_gate_bias=True, alpha=0.2):
        super(MMoE, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.expert_size = expert_size
        self.alpha = alpha
        self.use_gate_bias = use_gate_bias
        self.expert_kernels = nn.Parameter(torch.FloatTensor(num_experts, input_dim, hidden_dim, ))
        self.use_expert_bias = use_expert_bias
        # std = 1.0 / math.sqrt(input_dim)
        std = 0.1
        nn.init.normal_(self.expert_kernels, std=std)
        if use_expert_bias:
            self.expert_bias = nn.Parameter(torch.FloatTensor(num_experts, hidden_dim, ))
            nn.init.uniform_(self.expert_bias.data, -std, std)
        self.norm = nn.BatchNorm1d(input_dim)
        for i in range(1, expert_size):
            lin_expert_kernel = nn.Parameter(torch.FloatTensor(self.num_experts, self.hidden_dim, self.hidden_dim))
            setattr(self, 'lin_expert_kernel_{}'.format(i), lin_expert_kernel)
            nn.init.normal_(lin_expert_kernel, std=std)
            if use_expert_bias:
                lin_expert_bias = nn.Parameter(torch.FloatTensor(self.num_experts, self.hidden_dim))
                setattr(self, 'lin_expert_bias_{}'.format(i), lin_expert_bias)
                nn.init.uniform_(lin_expert_bias.data, -std, std)
            #    self.lin_expert_biases.append(lin_expert_bias)

        for i in range(1, self.num_tasks + 1):
            kernel = nn.Parameter(torch.FloatTensor(input_dim, num_experts))
            nn.init.normal_(kernel, std=std)
            setattr(self, 'gate_kernel_{}'.format(i), kernel)
            # self.gate_kernels.append(kernel)
            if use_gate_bias:
                bias = nn.Parameter(torch.FloatTensor(num_experts, ))
                nn.init.uniform_(bias, -std, std)
                setattr(self, 'gate_bias_{}'.format(i), bias)
            # self.gate_bias.append(bias)

    def forward(self, input):
        assert input.shape[-1] == self.input_dim  # batch_size*input_dim
        # expert_output = torch.mm(input, self.expert_kernels)
        expert_output = torch.bmm(input.unsqueeze(0).repeat(self.num_experts, 1, 1), self.expert_kernels).permute(1, 0,
                                                                                                                  2)
        if self.use_expert_bias:
            expert_output = torch.add(expert_output, self.expert_bias)
        # expert_output = self.norm(expert_output)
        expert_output = F.leaky_relu(expert_output, self.alpha)  # batch_size*num_expert*hidden_dim
        for i in range(1, self.expert_size):
            lin_expert_kernel = getattr(self, 'lin_expert_kernel_{}'.format(i))  # (num_expert,hidden_dim,hidden_dim)
            expert_output = torch.bmm(expert_output.permute(1, 0, 2), lin_expert_kernel).permute(1, 0, 2)
            if self.use_expert_bias:
                lin_expert_bias = getattr(self, 'lin_expert_bias_{}'.format(i))  # (num_expert,hidden_dim)
                expert_output = torch.add(expert_output, lin_expert_bias)
            # expert_output=self.norm(expert_output)
            expert_output = F.leaky_relu(expert_output, self.alpha)

        gate_outputs = []
        final_outputs = []
        for i in range(1, self.num_tasks + 1):
            gate_kernel = getattr(self, 'gate_kernel_{}'.format(i))  # input_dim*num_expert
            gate_output = torch.mm(input, gate_kernel)  # batch_size*num_expert
            if self.use_gate_bias:
                gate_output = torch.add(gate_output, getattr(self, 'gate_bias_{}'.format(i)))
            gate_output = F.leaky_relu(gate_output, negative_slope=self.alpha)
            gate_output = F.softmax(gate_output, dim=1)  # batch_size*num_experts
            # print(gate_output[:10])
            gate_outputs.append(gate_output)

        for gate_output in gate_outputs:
            expended_gate_output = torch.bmm(gate_output.unsqueeze(1), expert_output).squeeze(1)
            # expended_gate_output = F.leaky_relu(expended_gate_output, negative_slope=self.alpha)
            final_outputs.append(expended_gate_output)
        return final_outputs


class CF_MO(PairWiseModel):
    def __init__(self, dataset: BasicDataset):
        super(CF_MO, self).__init__()
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.cnt = 0
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = utils.latent_dim_rec
        self.n_layers = utils.n_layers  # self.config['n_layers']
        self.keep_prob = utils.keep_prob  # self.config['keep_prob']
        self.num_experts = utils.num_experts  # self.config['num_experts']
        self.leaky_alpha = utils.leaky_alpha  # self.config['leaky_alpha']
        self.reg_alpha = utils.reg_alpha  # self.config['reg_alpha']  # 回归的部分次方
        self.w1 = utils.w1  # self.config['w1']
        self.w2 = utils.w2  # self.config['w2']
        self.attn_weight = utils.attn_weight  # self.config['attn_weight']
        print("attn_weight:{}".format(self.attn_weight))
        self.train_mmoe = False
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )

        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )
        self.stdv = 0.1
        nn.init.normal_(self.embedding_user.weight, std=self.stdv)
        nn.init.normal_(self.embedding_item.weight, std=self.stdv)
        self.user_mat = self.dataset.getUserGraph(dense=False)
        print('user mat loaded')
        self.item_mat = self.dataset.getItemGraph(dense=False)
        print('item mat loaded')
        self.graph, self.graph_self = self.dataset.getSparseGraph(add_self=True)
        self.unit_graph, self.diag_graph = self.dataset.get_diag_unit_graph()
        print('graph has already loaded')
        self.pna_layer = PNA_layer(self.latent_dim, self.graph, self.unit_graph, self.diag_graph, activation=False,
                                   non_linear=False)
        self.user_item_layer = Light_user_layer(latent_dim=self.latent_dim, user_mat=self.user_mat,
                                                item_mat=self.item_mat,
                                                activation=False, non_linear=False)
        self.light_GCN_layer = LightGCN_layer(self.latent_dim, self.graph)
        self.norm = nn.BatchNorm1d(self.latent_dim)
        self.user_transform = MMoE(hidden_dim=self.latent_dim, num_tasks=2, num_experts=self.num_experts,
                                   input_dim=self.latent_dim, expert_size=3, alpha=self.leaky_alpha,
                                   use_expert_bias=True, use_gate_bias=True)
        self.item_transform = MMoE(hidden_dim=self.latent_dim, num_tasks=2, num_experts=self.num_experts,
                                   input_dim=self.latent_dim, expert_size=3, alpha=self.leaky_alpha,
                                   use_expert_bias=True, use_gate_bias=True)
        if self.attn_weight:
            self.W_u = nn.Linear(self.latent_dim, self.latent_dim)
            self.q_u = nn.Parameter(torch.FloatTensor(self.latent_dim))
            nn.init.normal_(self.W_u.weight, std=0.01)
            nn.init.normal_(self.W_u.bias, std=0.01)
            # nn.init.normal_(self.q_u, std=self.stdv)
            nn.init.zeros_(self.q_u)
            self.W_i = nn.Linear(self.latent_dim, self.latent_dim)
            self.q_i = nn.Parameter(torch.FloatTensor(self.latent_dim))
            nn.init.normal_(self.W_i.weight, std=0.01)
            nn.init.normal_(self.W_i.bias, std=0.01)
            # nn.init.normal_(self.q_i, std=self.stdv)
            nn.init.zeros_(self.q_i)
            self.scorer = nn.Linear(self.latent_dim, 1)
            nn.init.normal_(self.scorer.weight, std=self.stdv)
            nn.init.normal_(self.scorer.bias, std=self.stdv)

    def train_MMoE(self):
        self.train_mmoe = True
        self.embedding_user.requires_grad_(False)
        self.embedding_item.requires_grad_(False)
        self.user_transform.requires_grad_(True)
        self.item_transform.requires_grad_(True)

    def computer(self):
        """
        获得用户和物品的嵌入向量
        :return:
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        # all_emb = torch.cat([users_emb, items_emb], dim=0)
        users = [users_emb]
        items = [items_emb]
        users_emb_even, items_emb_even = users_emb, items_emb
        for i in range(0, self.n_layers, 2):
            # pna_user, pna_item = self.pna_layer([users_emb, items_emb])
            user_emb_odd, item_emb_odd = self.light_GCN_layer([users_emb_even, items_emb_even])
            users.append(user_emb_odd)
            items.append(item_emb_odd)
            user_emb_even, item_emb_even = self.user_item_layer([users_emb_even, items_emb_even])
            users.append(user_emb_even)
            items.append(item_emb_even)
        # print(len(users))
            # users_emb,items_emb=user_emb1,item_emb1
        # user_emb3, item_emb3 = self.light_GCN_layer([user_emb1, item_emb1])
        # user_emb4, item_emb4 = self.user_item_layer([user_emb1, item_emb1])
        # users.append(user_emb3)
        # users.append(user_emb4)
        # items.append(item_emb3)
        # items.append(item_emb4)

        if self.attn_weight:
            # 利用attention 计算不同层聚合的权重大小
            # for i in range(1, len(users)):
            #     users[i] = torch.mul(users[i], users[0]) + users[i]
            #     items[i] = torch.mul(items[i], items[0]) + items[i]
            user_cat_emb = torch.cat(users, dim=1).reshape(num_user, -1, self.latent_dim)
            weights = torch.sum(torch.mul(self.q_u, torch.tanh(self.W_u(user_cat_emb))), dim=2) / self.latent_dim
            weights = torch.softmax(weights, dim=1)
            users_emb = torch.bmm(weights.unsqueeze(1), user_cat_emb).squeeze(1)
            items_cat_emb = torch.cat(items, dim=1).reshape(num_item, -1, self.latent_dim)
            weights = torch.sum(torch.mul(self.q_i, torch.tanh(self.W_i(items_cat_emb))), dim=2) / self.latent_dim
            weights = torch.softmax(weights, dim=1)  # (batch_size,n_layers)
            items_emb = torch.bmm(weights.unsqueeze(1), items_cat_emb).squeeze(1)
            # user_cat_emb = torch.cat(users, dim=1).reshape(num_user, -1,
            #                                                self.latent_dim)  # (batch_size,n_layers,embedding_dim)
            # weights = torch.bmm(user_cat_emb, user_cat_emb.permute(0, 2, 1)) / math.sqrt(self.latent_dim)
            # # weights = torch.softmax(weights, dim=2)  # (batch_size,n_layers,n_layers)
            # # users_emb = torch.sum(torch.bmm(weights, user_cat_emb), 1)
            # weights = weights.sum(2)
            # weights = torch.softmax(weights, dim=1)
            # users_emb = torch.bmm(weights.unsqueeze(1), user_cat_emb).squeeze(1)
            #
            # item_cat_emb = torch.cat(items, dim=1).reshape(num_item, -1,
            #                                                self.latent_dim)  # (batch_size,n_layers,embedding_dim)
            # weights = torch.bmm(item_cat_emb, item_cat_emb.permute(0, 2, 1)) / math.sqrt(self.latent_dim)
            # # weights = torch.softmax(weights, dim=2)  # (batch_size,n_layers,n_layers)
            # # items_emb = torch.sum(torch.bmm(weights, item_cat_emb), 1)
            # weights = weights.sum(2)
            # weights = torch.softmax(weights, dim=1)
            # items_emb = torch.bmm(weights.unsqueeze(1), item_cat_emb).squeeze(1)
        else:
            # pass
            # 加上不同的权重
            # for i in range(1, len(users)):
            # if i % 2 == 1:
            # users[i] = (pna_user+users[i])/2.
            # items[i] = (pna_item+items[i])/2.
            # users[i] = torch.mul(users[0], users[i]) + users[i]
            # items[i] = torch.mul(items[0], items[i]) + items[i]
            # users[0] = weights[0] * users[0]
            # items[0] = weights[0] * items[0]
            users_emb = torch.stack(users, dim=1)
            users_emb = torch.mean(users_emb, dim=1)
            items_emb = torch.stack(items, dim=1)
            items_emb = torch.mean(items_emb, dim=1)
        return users_emb, items_emb

    def getUsersRating(self, users, all_users, all_items):
        # 这里的all_users和all_items为computer的结果
        # 这里面会加入MMoE模块
        # all_users,all_items=self.computer()
        users_emb = all_users[users]
        items_emb = all_items
        # users_emb = F.leaky_relu(self.W_u(users_emb), negative_slope=self.leaky_alpha)
        # items_emb = F.leaky_relu(self.W_i(items_emb), negative_slope=self.leaky_alpha)
        if not self.train_mmoe:
            ratings = torch.matmul(users_emb, items_emb.t())
        else:
            user_emb_list = self.user_transform(users_emb)
            item_emb_list = self.item_transform(items_emb)
            ratings = torch.multiply(torch.sigmoid(torch.mm(user_emb_list[0], item_emb_list[0].t())),
                                     torch.pow(torch.abs(torch.mm(user_emb_list[1], item_emb_list[1].t())),
                                               self.reg_alpha))
            # ratings = torch.sigmoid(torch.mm(user_emb_list[0], item_emb_list[0].t()))
        # ratings = torch.pow(torch.mm(user_emb_list[1], item_emb_list[1].t()), self.alpha)
        return ratings

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def loss(self, users, pos, neg, score):
        users_emb, pos_emb, neg_emb, users_emb0, pos_emb0, neg_emb0 = self.getEmbedding(users, pos, neg)
        # users_emb = all_users[users]
        # pos_emb = all_items[pos]
        # neg_emb = all_items[neg]
        # pos_scores1 = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        # neg_scores1 = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        # score1 = torch.mean(F.softplus(neg_scores1 - pos_scores1))
        # users_emb = F.leaky_relu(self.W_u(users_emb), negative_slope=self.leaky_alpha)
        # pos_emb = F.leaky_relu(self.W_i(pos_emb), negative_slope=self.leaky_alpha)
        # neg_emb = F.leaky_relu(self.W_i(neg_emb), negative_slope=self.leaky_alpha)
        reg_loss = (1 / 2) * (users_emb0.norm(2).pow(2) +
                              pos_emb0.norm(2).pow(2) +
                              neg_emb0.norm(2).pow(2)
                              ) / float(len(users))
        pos_scores1 = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores1 = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        if not self.train_mmoe:
            score1 = torch.mean(F.softplus(neg_scores1 - pos_scores1))
            return self.w1 * score1, 0 * score1, reg_loss
        # # # print('loss:{}'.format(score1))
        # pred = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        # neg_score = torch.ones_like(score, device=score.device) * (-100)
        # neg_pred = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        # print(F.mse_loss(neg_pred,neg_score))
        # score2 = F.mse_loss(pred, score) + F.mse_loss(neg_pred, -score)
        # return 0 * score2, self.w2 * score2, reg_loss

        else:
            user_emb_list = self.user_transform(users_emb)
            pos_item_emb_list = self.item_transform(pos_emb)
            neg_item_emb_list = self.item_transform(neg_emb)
            pos_score1 = torch.sum(torch.mul(user_emb_list[0], pos_item_emb_list[0]), dim=1)
            neg_score1 = torch.sum(torch.mul(user_emb_list[0], neg_item_emb_list[0]), dim=1)
            score1 = torch.mean(F.softplus(neg_score1 - pos_score1))
            # # # print(score1)
            pred_pos = torch.abs(torch.sum(torch.mul(user_emb_list[1], pos_item_emb_list[1]), dim=1))
            pred_neg = torch.abs(torch.sum(torch.mul(user_emb_list[1], neg_item_emb_list[1]), dim=1))
            neg_score = torch.zeros_like(score, device=score.device)
            # print('pred:{}'.format(pred_pos[:10]))
            # print('score:{}'.format(score[:10]))
            # print(max(score))
            score2 = F.mse_loss(pred_pos, score) + F.mse_loss(pred_neg, neg_score)
            # print(score1)
            # print(score2)
            return self.w1 * score1, self.w2 * score2, reg_loss
        # # # print("pred:{}".format(pred[:10]))
        # # # print("score:{}".format(score[:10]))
        # # # print(score2)

    def forward(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma


class Layer(nn.Module):
    def __init__(self, latent_dim, user_mat, item_mat, graph, mode=0, alpha=0.2):
        super(Layer, self).__init__()
        self.latent_dim = latent_dim
        self.user_mat = user_mat
        self.item_mat = item_mat
        self.graph = graph
        self.alpha = alpha
        self.mode = mode
        self.norm = nn.BatchNorm1d(latent_dim)
        self.W_u = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim))
        self.W_i = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim))
        self.W_1 = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim))
        self.W_2 = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.W_u, gain=1)
        nn.init.xavier_uniform_(self.W_i, gain=1)
        nn.init.xavier_uniform_(self.W_1, gain=1)
        nn.init.xavier_uniform_(self.W_2, gain=1)
        # self.W_uc = nn.Linear(2 * latent_dim, latent_dim)
        # self.W_ic = nn.Linear(2 * latent_dim, latent_dim)
        self.W_uc = nn.Parameter(torch.FloatTensor(3 * latent_dim, latent_dim))
        self.W_ic = nn.Parameter(torch.FloatTensor(3 * latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.W_uc, gain=1)
        nn.init.xavier_uniform_(self.W_ic, gain=1)

    def forward(self, input):
        # all_emb = torch.sparse.mm(self.graph, input)
        # return all_emb
        users_emb, items_emb = input[0], input[1]
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        # print(self.user_mat.shape)
        # print(users_emb.shape)
        h_u1 = torch.sparse.mm(self.user_mat, users_emb)
        # h_u1 = torch.mm(h_u1, self.W_u)
        # h_u1 = self.norm(h_u1)
        # h_u1 = F.leaky_relu(h_u1, negative_slope=self.alpha)
        h_i1 = torch.sparse.mm(self.item_mat, items_emb)
        # h_i1 = torch.mm(h_i1, self.W_i)
        # h_i1 = self.norm(h_i1)
        # h_i1 = F.leaky_relu(h_i1, negative_slope=self.alpha)
        # users_emb, items_emb = h_u1, h_i1

        # 目前来说，除去最简单的LightGCN和加上内积的LightGCN，这个方法最好用
        # all_emb = torch.cat([users_emb, items_emb], dim=0)
        # all_emb_tran = torch.sparse.mm(self.graph, all_emb)
        # all_emb_tran = self.norm(all_emb_tran)
        # all_emb_tran = F.leaky_relu(all_emb_tran, negative_slope=0.2)
        # users_emb, items_emb = torch.split(all_emb_tran, [num_user, num_item])
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        all_emb_tran = torch.sparse.mm(self.graph, all_emb)
        # all_emb = torch.mm((torch.mul(all_emb, all_emb_tran)), self.W_1) + torch.mm(all_emb_tran, self.W_2)
        # all_emb = torch.mul(all_emb_tran, all_emb) + all_emb_tran
        all_emb = all_emb_tran
        # all_emb = self.norm(all_emb)
        # all_emb = F.leaky_relu(all_emb, negative_slope=self.alpha)
        h_u2, h_i2 = torch.split(all_emb, [num_user, num_item])
        if self.mode == 0:
            return h_u1, h_i1
        else:
            return h_u2, h_i2
        # users_emb = self.norm(torch.add(h_u1, h_u2))
        # items_emb = self.norm(torch.add(h_i1, h_i2))
        # users_emb = torch.mm(torch.cat([h_u1, h_u2, torch.mul(h_u1, h_u2)], dim=1), self.W_uc)
        # users_emb = self.norm(users_emb)
        # users_emb = F.leaky_relu(users_emb, negative_slope=self.alpha)
        #
        # items_emb = torch.mm(torch.cat([h_i1, h_i2, torch.mul(h_i1, h_i2)], dim=1), self.W_ic)
        # items_emb = self.norm(items_emb)
        # items_emb = F.leaky_relu(items_emb, negative_slope=self.alpha)
        # users_emb = h_u1
        # items_emb=h_i1
        # return users_emb, items_emb


class LightGCN_layer(nn.Module):
    def __init__(self, latent_dim, graph, activation=False, non_linear=False):
        super(LightGCN_layer, self).__init__()
        self.latent_dim = latent_dim
        self.graph = graph
        self.activation = activation
        self.non_linear = non_linear
        self.norm = nn.BatchNorm1d(latent_dim)
        if self.non_linear:
            self.w = nn.Parameter(torch.FloatTensor(self.latent_dim, self.latent_dim))
            nn.init.normal_(self.w.data, std=0.1)

    def forward(self, input):
        users_emb, items_emb = input[0], input[1]
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        all_emb = torch.sparse.mm(self.graph, all_emb)
        if self.non_linear:
            all_emb = torch.mm(all_emb, self.w)
        if self.activation:
            # all_emb = self.norm(all_emb)
            all_emb = F.leaky_relu(all_emb, negative_slope=0.2)
        h_u2, h_i2 = torch.split(all_emb, [num_user, num_item])
        return h_u2, h_i2


class Light_cross_layer(nn.Module):
    def __init__(self, latent_dim, graph, activation=False, non_linear=False):
        super(Light_cross_layer, self).__init__()
        self.latent_dim = latent_dim
        self.graph = graph
        self.activation = activation
        self.non_linear = non_linear
        self.norm = nn.BatchNorm1d(latent_dim)
        if self.non_linear:
            self.W_1 = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim))
            self.W_2 = nn.Parameter(torch.FloatTensor(latent_dim, latent_dim))
            nn.init.normal_(self.W_1.data, std=0.1)
            nn.init.normal_(self.W_2.data, std=0.1)

    def forward(self, input):
        users_emb, items_emb = input[0], input[1]
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        all_emb_tran = torch.sparse.mm(self.graph, all_emb)
        if self.non_linear:
            all_emb = torch.mm((torch.mul(all_emb, all_emb_tran)), self.W_1) + torch.mm(all_emb_tran, self.W_2)
        else:
            all_emb = torch.mul(all_emb_tran, all_emb) + all_emb_tran
        if self.activation:
            # all_emb = self.norm(all_emb)
            all_emb = F.leaky_relu(all_emb, negative_slope=0.2)
        # all_emb = self.norm(all_emb)
        # all_emb = F.leaky_relu(all_emb, negative_slope=self.alpha)
        user_emb, item_emb = torch.split(all_emb, [num_user, num_item])
        return user_emb, item_emb


class Light_user_layer(nn.Module):
    def __init__(self, latent_dim, user_mat, item_mat, activation=False, non_linear=False):
        super(Light_user_layer, self).__init__()
        self.latent_dim = latent_dim
        self.user_mat = user_mat
        self.item_mat = item_mat
        self.activation = activation
        self.non_linear = non_linear

    def forward(self, input):
        users_emb, items_emb = input[0], input[1]
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        h_u1 = torch.sparse.mm(self.user_mat, users_emb)
        if self.non_linear:
            h_u1 = torch.mm(h_u1, self.W_u)
        if self.activation:
            h_u1 = self.norm(h_u1)
            h_u1 = F.leaky_relu(h_u1, negative_slope=self.alpha)
        h_i1 = torch.sparse.mm(self.item_mat, items_emb)
        if self.non_linear:
            h_i1 = torch.mm(h_i1, self.W_i)
        if self.activation:
            h_i1 = self.norm(h_i1)
            h_i1 = F.leaky_relu(h_i1, negative_slope=self.alpha)
        return h_u1, h_i1


class PNA_layer(nn.Module):
    def __init__(self, latent_dim, graph, unit_graph, diag_graph, activation=False, non_linear=False):
        super(PNA_layer, self).__init__()
        self.latent_dim = latent_dim
        self.graph = graph
        self.unit_graph = unit_graph
        self.diag_graph = diag_graph
        self.activation = activation
        self.non_linear = non_linear
        if self.non_linear:
            self.lin = nn.Linear(2 * self.latent_dim, latent_dim)

    def forward(self, input):
        """
        这里是PNA聚合和均值聚合的 加和，求一个均值
        :param input:
        :return:
        """
        users_emb, items_emb = input[0], input[1]
        num_user, num_item = users_emb.shape[0], items_emb.shape[0]
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        gcn_all_emb = torch.sparse.mm(self.graph, all_emb)
        # self_user, self_item = torch.split(self_all_emb, [num_user, num_item])
        sum_emb = torch.sparse.mm(self.unit_graph, all_emb)
        pow_sum_emb = torch.square(sum_emb)
        pow_emb = torch.square(all_emb)
        sum_pow_emb = torch.sparse.mm(self.unit_graph, pow_emb)
        pna_all_emb = 0.5 * torch.sparse.mm(self.diag_graph, pow_sum_emb - sum_pow_emb)
        # h_pna_u, h_pna_i = torch.split(pna_all_emb, [num_user, num_item])
        if self.non_linear:
            tran_all_emb = torch.cat([gcn_all_emb, pna_all_emb], dim=1)
            tran_all_emb = self.lin(tran_all_emb)
        else:
            tran_all_emb = 0.5 * (gcn_all_emb + pna_all_emb) + all_emb
            # users_emb = 0.5 * (self_user + h_pna_u)
            # items_emb = 0.5 * (self_item + h_pna_i)
        if self.activation:
            tran_all_emb = F.leaky_relu(tran_all_emb, negative_slope=0.2)
        # tran_all_emb=tran_all_emb+all_emb
        users_emb, items_emb = torch.split(tran_all_emb, [num_user, num_item])
        return users_emb, items_emb
        # return h_pna_u,h_pna_i
