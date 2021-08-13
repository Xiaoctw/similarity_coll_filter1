import os
from os.path import join, dirname
import torch
from enum import Enum
import sys
from os.path import join, dirname
from cppimport import imp_from_filepath
import multiprocessing
from pathlib import Path
import numpy as np
import argparse
import scipy.sparse as sp
from model import *
import time
from torch import nn, optim


def parse_args():
    parser = argparse.ArgumentParser(description="Go collaborative filter models")
    parser.add_argument('--bpr_batch', type=int, default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int, default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float, default=0.005,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-3,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='lastfm',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book,amazon-electronic]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20,40,60]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard")
    #  parser.add_argument('--comment', type=str, default="lgn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--multicore', type=int, default=1, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--model', type=str, default='cf_mo', help='rec-model, support [mf, lgn, ngcf,neumf,cmn,cf_mo]')
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--w1', type=float, default=1.)
    parser.add_argument('--w2', type=float, default=0.1)
    parser.add_argument('--attn_weight', type=int, default=0)
    parser.add_argument('--comment', type=str, default='None')
    parser.add_argument('--num_experts', type=int, default=10)
    parser.add_argument('--leaky_alpha', type=float, default=0.2)
    parser.add_argument('--reg_alpha', type=float, default=1.0)
    return parser.parse_args()


args = parse_args()
batch_size = args.bpr_batch
latent_dim_rec = args.recdim
n_layers = args.layer
dropout = args.dropout
keep_prob = args.keepprob
A_n_fold = args.a_fold
test_u_batch_size = args.testbatch
multicore = args.multicore
lr = args.lr
decay = args.decay
pretrain = args.pretrain
A_split = False
bigdata = False
num_experts = args.num_experts
leaky_alpha = args.leaky_alpha
attn_weight = args.attn_weight
reg_alpha = args.reg_alpha
w1 = args.w1
w2 = args.w2
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'amazon-electronic']
all_models = ['mf', 'lgn', 'ngcf', 'neumf', 'cmn', 'cf_mo']
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# args = parse_args()

ROOT_PATH = dirname(dirname(__file__))
CODE_PATH = ROOT_PATH + '/code'
DATA_PATH = ROOT_PATH + '/data'
BOARD_PATH = CODE_PATH + '/runs'
FILE_PATH = CODE_PATH + '/checkpoints'
GPU = torch.cuda.is_available()
device = torch.device('cuda:{}'.format(args.cuda) if GPU and args.cuda else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed
dataset_name = args.dataset
model_name = args.model
if dataset_name not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset_name} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")
TRAIN_epochs = args.epochs
LOAD = args.load
# PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard

comment = None
loss = None
if args.comment == 'None':
    comment = model_name
else:
    comment = args.comment

weight_file = FILE_PATH + '/' + f"cf_mo-{dataset_name}-{comment}-{n_layers}-{latent_dim_rec}.pth.tar"

if model_name in {'mf', 'lgn', 'ngcf', 'neumf', 'cmn'}:
    loss = 'bpr'
else:
    loss = 'mmoe_loss'


def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")


try:
    sys.path.append(CODE_PATH + '/sources')
    # print(CODE_PATH + '/sources')
    path = join(dirname(__file__), "sources", "sampling.cpp")
        # print(path)
    sampling = imp_from_filepath(path)
        # sampling=cppimport.imp(path)
    cprint('CPP extension loaded')
    sampling.seed(seed)
    sample_ext = True

except:
    cprint("Cpp extension not loaded")
    sample_ext = False


class BPRLoss:
    def __init__(self,
                 recmodel: PairWiseModel, ):
        self.model = recmodel
        # print(config['decay'])
        self.weight_decay = decay
        self.lr = lr
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    # 在这里进行梯度下降，传播运算
    def stageOne(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        # print(reg_loss)
        reg_loss = reg_loss * self.weight_decay
        loss = loss + reg_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.cpu().item(), loss.cpu().item() - reg_loss.cpu().item(), reg_loss.cpu().item()


class MMoELoss:
    def __init__(self,
                 recmodel: PairWiseModel, ):
        self.model = recmodel
        self.weight_decay = decay
        self.lr = lr
        # 这里不要在加上weight_Decay了
        # self.opt = optim.Adam(recmodel.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        print(recmodel.parameters())

    def stageOne(self, users, pos, neg, score):
        loss1, loss2, reg_loss = self.model.loss(users, pos, neg, score)
        # regulation_loss = 0
        # for param in self.model.parameters():
        #     regulation_loss += torch.sum(torch.pow(param, 2))
        regulation_loss = self.weight_decay * reg_loss
        # loss = loss1 + loss2 + regulation_loss
        loss = loss1 + loss2 + regulation_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.cpu().item(), loss1.cpu().item(), loss2.cpu().item(), regulation_loss.cpu().item()


def UniformSample_original(dataset, neg_ratio=1, score=False):
    dataset: BasicDataset
    allPos = dataset.allPos
    # print(allPos)
    start = time.time()
    try:

        # print('n_item:{}'.format(dataset.n_users))
        # print('m_item:{}'.format(dataset.m_items))
        # print('train_data_size:{}'.format(dataset.trainDataSize))
        if not score:
            S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                         dataset.trainDataSize, allPos, neg_ratio)
        else:
            allPosScore = dataset.allPosScores
            S = sampling.sample_negative_score(dataset.n_users, dataset.m_items,
                                               dataset.trainDataSize, allPos, allPosScore, neg_ratio)
    # print('sampling in cpp')
    # print('finish')
    except:
        print('sampling in python')
        S = UniformSample_original_python(dataset, score=score)
    return S


def UniformSample_original_python(dataset, score=False):
    """
    the original impliment of BPR Sampling in LightGCN
    进行采用的过程，通过采样，得到正类样本和负类样本
    :return:
        np.array
    """
    total_start = time.time()
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    if not score:
        for i, user in enumerate(users):
            start = time.time()
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue
            sample_time2 += time.time() - start
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            while True:
                negitem = np.random.randint(0, dataset.m_items)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
            end = time.time()
            sample_time1 += end - start
    else:
        allPosScores = dataset.allPosScores
        for i, user in enumerate(users):
            start = time.time()
            posForUser = allPos[user]
            posForUserScore = allPosScores[user]
            # print(len(posForUser))
            # print(len(posForUserScore))
            # if len(posForUser)!=len(posForUserScore):
            # print(user)
            # print(len(posForUser))
            # print(len(posForUserScore))
            if len(posForUser) == 0:
                continue
            sample_time2 += time.time() - start
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            posScore = posForUserScore[posindex]
            while True:
                negitem = np.random.randint(0, dataset.m_items)
                if negitem in posForUser:
                    continue
                else:
                    break
            S.append([user, positem, negitem, posScore])
            end = time.time()
            sample_time1 += end - start
    total = time.time() - total_start
    return np.array(S)


# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


# return os.path.join(world.FILE_PATH, file)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', utils.batch_size)
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def preprocess_graph(adj, c=1):
    """ process the graph
        * options:
        normalization of augmented adjacency matrix
        formulation from convolutional filter
        normalized graph laplacian
    Parameters
    ----------
    adj: a sparse matrix represents the adjacency matrix
    Returns
    -------
    adj_normalized: a sparse matrix represents the normalized laplacian
        matrix
    """
    _adj = adj + c * sp.eye(adj.shape[0])  # Sparse matrix with ones on diagonal 产生对角矩阵
    # _D = sp.diags(_dseq)
    _dseq = _adj.sum(1).A1  # 按行求和后拉直
    # _dseq[_dseq == 0] = 1
    _dseq = np.power(_dseq, -0.5)
    _dseq[np.isinf(_dseq)] = 0.
    _dseq[np.isnan(_dseq)] = 0.
    _D_half = sp.diags(_dseq)  # 开平方构成对角矩阵
    # _D_half[sp]
    adj_normalized = _D_half @ _adj @ _D_half  # 矩阵乘法
    return adj_normalized.tocsr()  # 转成稀疏矩阵存储


def preprocess_adj_graph(mat, num=8):
    # num = 8
    n = mat.shape[0]
    mat = sp.lil_matrix(mat)
    mat[np.arange(n), np.arange(n)] = 0
    mat = sp.csr_matrix(mat)
    idxs1, idxs2 = [], []
    res_data = []
    for i in range(n):
        idxes = mat[i].nonzero()[1]
        data = mat[i].data
        list1 = list(zip(idxes, data))
        list1.sort(key=lambda x: x[1])
        list1 = list1[-min(len(list1), num):]
        for j in range(len(list1)):
            idxs1.append(i)
            idxs2.append(list1[j][0])
            res_data.append(list1[j][1])
        if i % (n // 5) == 0:
            print('{}/5'.format(i // (n // 5)))
    res_data = np.array(res_data)
    index = [np.array(idxs1), np.array(idxs2)]
    mat = sp.coo_matrix((res_data, index),
                        shape=(n, n))

    # index = dense_mat.nonzero()
    # data = dense_mat[dense_mat >= 1e-9]
    # args = np.argsort(dense_mat, axis=1)[:, -num:]
    # idxes = np.concatenate(num * [np.arange(len(args))], axis=0)
    # idxes = idxes.reshape(num, len(args)).transpose()
    # user_mat = np.zeros((len(args), len(args)))
    # user_mat[idxes, args] = dense_mat[idxes, args]
    # D = np.sum(user_mat, axis=1)
    # D[D == 0.] = 1.
    # D_sqrt = np.sqrt(D)
    # user_mat = user_mat / D_sqrt
    # user_mat = user_mat / D_sqrt.t()
    # _dseq = user_mat.sum(1)  # 按行求和后拉直
    # _dseq[_dseq == 0] = 1
    # _D_half = np.diag(np.power(_dseq, -0.5))  # 开平方构成对角矩阵
    # user_mat = _D_half.dot(user_mat).dot(_D_half)  # 矩阵乘法
    # user_mat[np.isnan(user_mat)] = 0
    # index = user_mat.nonzero()
    # # data = user_mat[user_mat > 0]
    # data = user_mat[index]
    # # print(index[0].shape)
    # # print(data.shape)
    # user_mat = sp.coo_matrix((data, index),
    #                          shape=(len(args), len(args)))
    mat = preprocess_graph(mat, 0)

    return mat


def transform_score(score: np.ndarray):
    import pandas as pd
    data = pd.DataFrame({'score': score})
    scores1 = score.copy()
    np.sort(scores1)
    num_item = len(scores1)
    bin = num_item // 5

    def cut(val):
        if val <= scores1[bin]:
            return 1
        elif val <= scores1[bin * 2]:
            return 2
        elif val <= scores1[bin * 3]:
            return 3
        elif val <= scores1[bin * 4]:
            return 4
        return 5

    data['new_score'] = data['score'].apply(cut)
    return data['new_score']


def construct_dataset():
    """
    不能直接设置dataset变量，因为dataloader调用了utils，utils就不要再调用dataloader了
    :return:
    """
    dataset = None
    if dataset_name in ['gowalla', 'yelp2018']:
        # path = Path(__file__).parent.parent / 'data' / world.dataset
        # path = join(dirname(os.path.dirname(__file__)), 'data', dataset_name)
        # 导入数据集
        dataset = dataloader.Loader()
    elif dataset_name == 'lastfm':
        #  path = Path(__file__).parent.parent / 'data' / 'lastfm'
        # path = join(dirname(os.path.dirname(__file__)), 'data', 'lastfm')
        dataset = dataloader.LastFM()
    elif dataset_name in ['amazon-electronic', 'amazon-book']:
        # path = join(dirname(os.path.dirname(__file__)), 'data', dataset_name)
        dataset = dataloader.Amazon()
    return dataset


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                # TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset: BasicDataset
    r_all = np.zeros((dataset.m_items,))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(r, test_item_scores)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
