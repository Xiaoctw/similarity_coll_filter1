import os
from os.path import join
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

import utils


class BasicDataset(Dataset):
    def __init__(self):
        self.path = join(os.path.dirname(os.path.dirname(__file__)), 'data', utils.dataset_name)
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    @property
    def allPosScores(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self, add_self=False):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

    def get_diag_unit_graph(self):
        raise NotImplementedError

    def getUserGraph(self, dense=False):
        pass

    def getItemGraph(self, dense=False):
        pass


class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """

    def __init__(self, ):
        super(LastFM, self).__init__()
        # train or test
        utils.cprint("loading [last fm]")
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        # trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        trainData = pd.read_table(self.path + '/data1.txt', header=None)
        # print(trainData.head())
        testData = pd.read_table(self.path + '/test1.txt', header=None)
        # print(testData.head())
        trustNet = pd.read_table(self.path + '/trustnetwork.txt', header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData -= 1
        testData -= 1
        # 这里分数减少1后，会把原来为1的值变为0。
        trainData[:][2] += 1
        testData[:][2] += 1
        # self.trustNet = trustNet
        self.trainData = trainData
        self.testData = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        self.trainScore = np.array(trainData[:][2])
        # print('trainUser:{}'.format(len(self.trainUser)))
        # print('tainScore:{}'.format(len(self.trainScore)))
        self.trainUniqueItems = np.unique(self.trainItem)
        # self.trainDataSize = len(self.trainUser)
        self.testUser = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem = np.array(testData[:][1])
        self.testScore = np.array(testData[:][2])
        # 进行归一化
        self.score = np.concatenate([self.trainScore, self.testScore], axis=0)
        # self.score = preprocessing.MinMaxScaler((1,5)).fit_transform(self.score.reshape(-1, 1)).reshape(-1, )
        self.score = utils.transform_score(self.score)
        self.trainScore, self.testScore = np.split(self.score, [self.trainScore.shape[0]])
        self.unique_user = np.unique(np.concatenate([self.trainUniqueUsers, self.testUniqueUsers]))
        self.unique_item = np.unique(np.concatenate([self.trainItem, self.testItem]))
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser)) / self.n_users / self.m_items}")

        # (users,users)
        # self.socialNet = csr_matrix((np.ones(len(trustNet)), (trustNet[:, 0], trustNet[:, 1])),
        #                             shape=(self.n_users, self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.m_items))

        self.UserItemScoreNet = csr_matrix((self.trainScore, (self.trainUser, self.trainItem)),
                                           shape=(self.n_users, self.m_items))
        # print(self.UserItemScoreNet)

        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self._allPosScores = self.getUserPosItemsScore(list(range(self.n_users)))
        # self._allPosScores = list(self._allPosScores)
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()
        # self.user_mat = sp.load_npz(self.path + '/user_mat.npz')
        # self.item_mat = sp.load_npz(self.path + '/item_mat.npz')
        # self.user_mat=self.UserItemNet.dot(self.UserItemNet.transpose())
        # self.item_mat=self.UserItemNet.transpose().dot(self.UserItemNet)

    @property
    def n_users(self):
        return int(np.max(self.unique_user) + 1)

    @property
    def m_items(self):
        return int(np.max(self.unique_item) + 1)

    @property
    def trainDataSize(self):
        return len(self.trainUser)

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def allPosScores(self):
        return self._allPosScores

    def getSparseGraph(self, add_self=False):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data,
                                                torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]))
            row = torch.arange(0, self.n_users + self.m_items)
            col = torch.arange(0, self.n_users + self.m_items)
            index = torch.stack([row, col])
            I = torch.sparse.FloatTensor(index, torch.ones(self.n_users + self.m_items),
                                         torch.Size((self.n_users + self.m_items, self.m_items + self.n_users)))
            self.Graph_self = self.Graph + I
            self.Graph = self.graph_helper(self.Graph)
            self.Graph = self.Graph.coalesce().to(utils.device)
            self.Graph_self = self.graph_helper(self.Graph_self)
            self.Graph_self = self.Graph_self.coalesce().to(utils.device)
            #     m = g1.shape[0]
            #     row = torch.arange(0, m)
            #     col = torch.arange(0, m)
            #     index = torch.stack([row, col])
            #     data = torch.ones(m)
            #     I = torch.sparse.FloatTensor(index, data,torch.Size((m,m)))
        #     if add_self:
        #         row = torch.arange(0, self.n_users + self.m_items)
        #         col = torch.arange(0, self.n_users + self.m_items)
        #         index = torch.stack([row, col])
        #         data = torch.ones(self.n_users + self.m_items)
        #         I = torch.sparse.FloatTensor(index, data,
        #                                      torch.Size((self.n_users + self.m_items, self.m_items + self.n_users)))
        #         self.Graph_self = self.Graph + I
        #         self.Graph_self = self.Graph_self.coalesce().to(world_config['device'])
        #     self.Graph = self.Graph.coalesce().to(world_config['device'])
        # if add_self:
        #     # print('graph shape:{}'.format(self.Graph.shape))
        #     # print('graph_self shape:{}'.format(self.Graph_self.shape))
        #     return self.Graph, self.Graph_self
        if add_self:
            return self.Graph, self.Graph_self
        return self.Graph

    def get_diag_unit_graph(self):
        user_dim = torch.LongTensor(self.trainUser)
        item_dim = torch.LongTensor(self.trainItem)
        first_sub = torch.stack([user_dim, item_dim + self.n_users])
        second_sub = torch.stack([item_dim + self.n_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1))
        graph = torch.sparse.FloatTensor(index, data,
                                         torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]))
        dense = graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D = torch.pow(D, -1)
        D[torch.isinf(D)] = 0
        index = torch.stack([torch.arange(self.n_users + self.m_items), torch.arange(self.n_users + self.m_items)])
        D = torch.sparse.FloatTensor(index, D, torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]))
        graph = graph.coalesce().to(utils.device)
        D = D.coalesce().to(utils.device)
        return graph, D

    def graph_helper(self, graph):
        # 相当于进行计算D-1AD操作
        dense = graph.to_dense()
        D = torch.sum(dense, dim=1).float()
        D[D == 0.] = 1.
        D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        dense = dense / D_sqrt
        dense = dense / D_sqrt.t()
        index = dense.nonzero()
        data = dense[dense >= 1e-9]
        assert len(index) == len(data)
        graph = torch.sparse.FloatTensor(index.t(), data, torch.Size(
            [self.n_users + self.m_items, self.n_users + self.m_items]))
        graph = graph.coalesce().to(utils.device)
        return graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            # print(self.UserItemNet[user].nonzero())
            posItems.append(self.UserItemNet[user].nonzero()[1])
            # print(posItems[-1])
        return posItems

    def getUserPosItemsScore(self, users):
        scores = []
        for user in users:
            # if user == 1209:
            #     print(self.UserItemScoreNet[user].nonzero()[1])
            # print(self.UserItemScoreNet[user].nonzero())
            # print(len(self.UserItemScoreNet[user].data))
            scores.append(self.UserItemScoreNet[user].data)
            # print(scores[-1])
        return scores

    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems

    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        return len(self.trainUniqueUsers)

    def getUserGraph(self, dense=False):
        num = 8
        if os.path.exists(self.path + '/user_mat.npz'):
            user_mat = sp.load_npz(self.path + '/user_mat.npz')
        else:
            user_mat = self.UserItemNet.dot(self.UserItemNet.transpose())
            dense_mat = user_mat.todense()
            dense_mat[np.arange(len(dense_mat)), np.arange(len(dense_mat))] = 0
            # index = dense_mat.nonzero()
            # data = dense_mat[dense_mat >= 1e-9]
            args = np.argsort(dense_mat, axis=1)[:, -num:]
            idxes = np.concatenate(num * [np.arange(len(args))], axis=0)
            idxes = idxes.reshape(num, len(args)).transpose()
            user_mat = np.zeros((len(args), len(args)))
            user_mat[idxes, args] = dense_mat[idxes, args]
            # D = np.sum(user_mat, axis=1)
            # D[D == 0.] = 1.
            # D_sqrt = np.sqrt(D)
            # user_mat = user_mat / D_sqrt
            # user_mat = user_mat / D_sqrt.t()
            _dseq = user_mat.sum(1)  # 按行求和后拉直
            _dseq[_dseq == 0] = 1
            _D_half = np.diag(np.power(_dseq, -0.5))  # 开平方构成对角矩阵
            user_mat = _D_half.dot(user_mat).dot(_D_half)  # 矩阵乘法
            user_mat[np.isnan(user_mat)] = 0
            index = user_mat.nonzero()
            # data = user_mat[user_mat > 0]
            data = user_mat[index]
            # print(index[0].shape)
            # print(data.shape)
            user_mat = sp.coo_matrix((data, index),
                                     shape=(len(args), len(args)))
            sp.save_npz(self.path + '/user_mat.npz', user_mat)
        if dense:
            user_mat = user_mat.todense()
            user_mat = torch.Tensor(user_mat)
            return user_mat
        else:
            user_mat = self._convert_sp_mat_to_sp_tensor(user_mat)
        # print(user_mat)
        # self.Graph = self.Graph.coalesce().to(world_config['device'])
        user_mat = user_mat.coalesce().to(utils.device)  # device'])
        # print(user_mat.shape)

        return user_mat

    def getItemGraph(self, dense=False):
        num = 8
        if os.path.exists(self.path + '/item_mat.npz'):
            item_mat = sp.load_npz(self.path + '/item_mat.npz')
        else:
            item_mat = (self.UserItemNet.transpose()).dot(self.UserItemNet)
            dense_mat = item_mat.todense()
            dense_mat[np.arange(len(dense_mat)), np.arange(len(dense_mat))] = 0
            # index = dense_mat.nonzero()
            # data = dense_mat[dense_mat >= 1e-9]
            args = np.argsort(dense_mat, axis=1)[:, -num:]
            idxes = np.concatenate(num * [np.arange(len(args))], axis=0)
            idxes = idxes.reshape(num, len(args)).transpose()
            item_mat = np.zeros((len(args), len(args)))
            item_mat[idxes, args] = dense_mat[idxes, args]
            # D = np.sum(user_mat, axis=1)
            # D[D == 0.] = 1.
            # D_sqrt = np.sqrt(D)
            # user_mat = user_mat / D_sqrt
            # user_mat = user_mat / D_sqrt.t()
            _dseq = item_mat.sum(1)  # 按行求和后拉直
            _dseq[_dseq == 0] = 1
            _D_half = np.diag(np.power(_dseq, -0.5))  # 开平方构成对角矩阵
            item_mat = _D_half.dot(item_mat).dot(_D_half)  # 矩阵乘法
            item_mat[np.isnan(item_mat)] = 0
            index = item_mat.nonzero()
            # data = item_mat[item_mat > 0]
            data = item_mat[index]
            item_mat = sp.coo_matrix((data, index),
                                     shape=(len(args), len(args)))
            sp.save_npz(self.path + '/item_mat.npz', item_mat)
        if dense:
            item_mat = item_mat.todense()
            item_mat = torch.Tensor(item_mat)
            return item_mat
        else:
            item_mat = self._convert_sp_mat_to_sp_tensor(item_mat)
        # self.Graph = self.Graph.coalesce().to(world_config['device'])
        item_mat = item_mat.coalesce().to(utils.device)
        # print(item_mat.shape)
        return item_mat

    def _convert_sp_mat_to_sp_tensor(self, X):
        '''
        转化为
        tensor形式
        '''
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        mat = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        return mat


class Amazon(BasicDataset):
    def __init__(self, ):
        # train or test
        super().__init__()
        # self.path=join(os.path.dirname(os.path.dirname(__file__)), 'data', self.config['dataset'])
        utils.cprint(f'loading [{self.path}]')
        # print(config)
        self.split = utils.A_split
        self.folds = utils.A_n_fold
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        # self.path = path
        train_file = self.path + '/train.txt'
        train_score_file = self.path + '/train_score.txt'
        test_file = self.path + '/test.txt'
        test_score_file = self.path + '/test_score.txt'
        trainUniqueUsers, trainItem, trainUser, trainScore = [], [], [], []
        testUniqueUsers, testItem, testUser, testScore = [], [], [], []
        self.train_data_size = 0
        self.testDataSize = 0
        cnt = 0
        with open(train_file) as f:
            for l in f.readlines():
                cnt += 1
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    # if len(items)==0:
                    #     print(uid)
                    #     print(cnt)
                    if len(items) > 0:
                        self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.train_data_size += len(items)
        with open(train_score_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    scores = [float(i) for i in l[1:]]
                    trainScore.extend(scores)

        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainScores = np.array(trainScore)

        with open(test_file) as f:
            for l in f.readlines():
                l = l.strip().split(' ')
                if len(l) > 1:  # amazon-book有个例外
                    items = [int(i) for i in l[1:]]
                    # print(items)
                    uid = int(l[0])
                    #  print(uid)
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    if len(items) > 0:
                        self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        with open(test_score_file) as f:
            for l in f.readlines():
                l = l.strip('\n').split(' ')
                scores = [float(i) for i in l[1:]]
                testScore.extend(scores)

        self.m_item += 1
        self.n_user += 1
        print('number of user:{}'.format(self.n_user))
        print('number of item:{}'.format(self.m_item))
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.testScore = np.array(testScore)
        self.Graph = None
        print(f"{self.train_data_size} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{utils.dataset_name} Sparsity : {(self.train_data_size + self.testDataSize) / self.n_users / self.m_items}")
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))

        self.UserItemScoreNet = csr_matrix((self.trainScores, (self.trainUser, self.trainItem)),
                                           shape=(self.n_user, self.m_item))

        # 计算度矩阵
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self._allPosScore = self.getUserPosItemsScore(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{utils.dataset_name} is ready to go")
        # if os.path.isfile(self.path + '/user_mat.npz'):
        #     self.user_mat = sp.load_npz(self.path + '/user_mat.npz')
        # else:
        #     self.user_mat = None
        #
        # if os.path.isfile(self.path + '/item_mat.npz'):
        #     self.item_mat = sp.load_npz(self.path + '/item_mat.npz')
        # else:
        #     self.item_mat = None

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.train_data_size

    @property
    def testDict(self):
        """
        获取的就是字典，键为用户，值为对应的物品
        :return:
        """
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def allPosScores(self):
        return self._allPosScore

    def _split_A_hat(self, A):
        # 意思是把图给分割了，画成了很多等份进行训练
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(utils.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        '''
        转化为
        tensor形式
        '''
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        mat = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        return mat

    def getSparseGraph(self, add_self=False):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                norm_adj = pre_adj_mat
                if add_self:
                    pre_adj_mat_self = sp.load_npz(self.path + '/s_pre_adj_mat_self.npz')
                    norm_adj_self = pre_adj_mat_self
                print("successfully loaded...")
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print("costing {:.4f} s, saved norm_mat...".format(end - s))
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
                if add_self:
                    adj_mat_self = adj_mat + sp.eye(adj_mat.shape[0])
                    rowsum_self = np.array(adj_mat_self.sum(axis=1))
                    d_inv_self = np.power(rowsum_self, -0.5).flatten()
                    d_inv_self[np.isinf(d_inv_self)] = 0.
                    d_mat_self = sp.diags(d_inv_self)
                    norm_adj_self = d_mat_self.dot(adj_mat_self)
                    norm_adj_self = norm_adj_self.dot(d_mat_self)
                    norm_adj_self = norm_adj_self.tocsr()
                    sp.save_npz(self.path + '/s_pre_adj_mat_self.npz', norm_adj_self)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                if add_self:
                    self.Graph_self = self._split_A_hat(norm_adj_self)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(utils.device)
                if add_self:
                    self.Graph_self = self._convert_sp_mat_to_sp_tensor(norm_adj_self)
                    self.Graph_self = self.Graph.coalesce().to(utils.device)
                print("don't split the matrix")
        if add_self:
            return self.Graph, self.Graph_self
        return self.Graph

    def get_diag_unit_graph(self):
        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
            norm_adj = pre_adj_mat
            print("successfully loaded...")
        except:
            print("generating adjacency matrix")
            s = time()
            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todok()
            # d_inv = np.power(rowsum, -0.5).flatten()
            # d_inv[np.isinf(d_inv)] = 0.
            # d_mat = sp.diags(d_inv)
            norm_adj = adj_mat
            # norm_adj = d_mat.dot(adj_mat)
            # norm_adj = norm_adj.dot(d_mat)
            # norm_adj = norm_adj.tocsr()
            end = time()
            print("costing {:.4f} s, saved norm_mat...".format(end - s))
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
        rowsum = np.array(norm_adj.sum(axis=1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat = self._convert_sp_mat_to_sp_tensor(d_mat)
        d_mat = d_mat.coalesce().to(utils.device)
        graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        graph = graph.coalesce().to(utils.device)
        return graph, d_mat

    # def getSparseGraph1(self):
    #     g1 = self.getSparseGraph()
    #     m = g1.shape[0]
    #     row = torch.arange(0, m)
    #     col = torch.arange(0, m)
    #     index = torch.stack([row, col])
    #     data = torch.ones(m)
    #     I = torch.sparse.FloatTensor(index, data,torch.Size((m,m)))
    #     return g1, g1 + I

    # index = torch.stack([row, col])
    # data = torch.FloatTensor(coo.data)
    # return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def getUserPosItemsScore(self, users):
        scores = []
        for user in users:
            # if user == 1209:
            #     print(self.UserItemScoreNet[user].nonzero()[1])
            scores.append(self.UserItemScoreNet[user].nonzero()[1])
        return scores

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
    @trainDataSize.setter
    def trainDataSize(self, value):
        self._trainDataSize = value

    # def getUserGraph(self, dense=False):
    #     num = 8
    #     if os.path.exists(self.path + '/user_mat.npz'):
    #         user_mat = sp.load_npz(self.path + '/user_mat.npz')
    #     else:
    #         user_mat = self.UserItemNet.dot(self.UserItemNet.transpose())
    #         dense_mat = user_mat.todense()
    #         dense_mat[np.arange(len(dense_mat)), np.arange(len(dense_mat))] = 0
    #         # index = dense_mat.nonzero()
    #         # data = dense_mat[dense_mat >= 1e-9]
    #         args = np.argsort(dense_mat, axis=1)[:, -num:]
    #         idxes = np.concatenate(num * [np.arange(len(args))], axis=0)
    #         idxes = idxes.reshape(num, len(args)).transpose()
    #         user_mat = np.zeros((len(args), len(args)))
    #         user_mat[idxes, args] = dense_mat[idxes, args]
    #         # D = np.sum(user_mat, axis=1)
    #         # D[D == 0.] = 1.
    #         # D_sqrt = np.sqrt(D)
    #         # user_mat = user_mat / D_sqrt
    #         # user_mat = user_mat / D_sqrt.t()
    #         _dseq = user_mat.sum(1)  # 按行求和后拉直
    #         _dseq[_dseq == 0] = 1
    #         _D_half = np.diag(np.power(_dseq, -0.5))  # 开平方构成对角矩阵
    #         user_mat = _D_half.dot(user_mat).dot(_D_half)  # 矩阵乘法
    #         user_mat[np.isnan(user_mat)] = 0
    #         index = user_mat.nonzero()
    #         # data = user_mat[user_mat > 0]
    #         data = user_mat[index]
    #         # print(index[0].shape)
    #         # print(data.shape)
    #         user_mat = sp.coo_matrix((data, index),
    #                                  shape=(len(args), len(args)))
    #         sp.save_npz(self.path + '/user_mat.npz', user_mat)
    #     if dense:
    #         user_mat = user_mat.todense()
    #         user_mat = torch.Tensor(user_mat)
    #         return user_mat
    #     else:
    #         user_mat = self._convert_sp_mat_to_sp_tensor(user_mat)
    #     # self.Graph = self.Graph.coalesce().to(world_config['device'])
    #     user_mat = user_mat.coalesce().to(world_config['device'])
    #     # print(user_mat.shape)
    #     return user_mat

    # def getItemGraph(self, dense=False):
    #     num = 8
    #     if os.path.exists(self.path + '/item_mat.npz'):
    #         item_mat = sp.load_npz(self.path + '/item_mat.npz')
    #     else:
    #         item_mat = (self.UserItemNet.transpose()).dot(self.UserItemNet)
    #         dense_mat = item_mat.todense()
    #         dense_mat[np.arange(len(dense_mat)), np.arange(len(dense_mat))] = 0
    #         # index = dense_mat.nonzero()
    #         # data = dense_mat[dense_mat >= 1e-9]
    #         args = np.argsort(dense_mat, axis=1)[:, -num:]
    #         idxes = np.concatenate(num * [np.arange(len(args))], axis=0)
    #         idxes = idxes.reshape(num, len(args)).transpose()
    #         item_mat = np.zeros((len(args), len(args)))
    #         item_mat[idxes, args] = dense_mat[idxes, args]
    #         # D = np.sum(user_mat, axis=1)
    #         # D[D == 0.] = 1.
    #         # D_sqrt = np.sqrt(D)
    #         # user_mat = user_mat / D_sqrt
    #         # user_mat = user_mat / D_sqrt.t()
    #         _dseq = item_mat.sum(1)  # 按行求和后拉直
    #         _dseq[_dseq == 0] = 1
    #         _D_half = np.diag(np.power(_dseq, -0.5))  # 开平方构成对角矩阵
    #         item_mat = _D_half.dot(item_mat).dot(_D_half)  # 矩阵乘法
    #         item_mat[np.isnan(item_mat)] = 0
    #         index = item_mat.nonzero()
    #         # data = item_mat[item_mat > 0]
    #         data = item_mat[index]
    #         item_mat = sp.coo_matrix((data, index),
    #                                  shape=(len(args), len(args)))
    #         sp.save_npz(self.path + '/item_mat.npz', item_mat)
    #     if dense:
    #         item_mat = item_mat.todense()
    #         item_mat = torch.Tensor(item_mat)
    #         return item_mat
    #     else:
    #         item_mat = self._convert_sp_mat_to_sp_tensor(item_mat)
    #     # self.Graph = self.Graph.coalesce().to(world_config['device'])
    #     item_mat = item_mat.coalesce().to(world_config['device'])
    #     # print(item_mat.shape)
    #     return item_mat

    def getUserGraph(self, dense=False):
        if os.path.exists(self.path + '/user_mat.npz'):
            user_mat = sp.load_npz(self.path + '/user_mat.npz')
        else:
            user_mat = self.UserItemNet.dot(self.UserItemNet.transpose())
            user_mat = utils.preprocess_adj_graph(user_mat)
            sp.save_npz(self.path + '/user_mat.npz', user_mat)
        if dense:
            user_mat = user_mat.todense()
            user_mat = torch.Tensor(user_mat)
            return user_mat
        else:
            user_mat = self._convert_sp_mat_to_sp_tensor(user_mat)
        user_mat = user_mat.coalesce().to(utils.device)
        return user_mat

    def getItemGraph(self, dense=False):
        if os.path.exists(self.path + '/item_mat.npz'):
            item_mat = sp.load_npz(self.path + '/item_mat.npz')
        else:
            item_mat = (self.UserItemNet.transpose()).dot(self.UserItemNet)
            item_mat = utils.preprocess_adj_graph(item_mat)
            sp.save_npz(self.path + '/item_mat.npz', item_mat)
        if dense:
            item_mat = item_mat.todense()
            item_mat = torch.Tensor(item_mat)
            return item_mat
        else:
            item_mat = self._convert_sp_mat_to_sp_tensor(item_mat)
        item_mat = item_mat.coalesce().to(utils.device)
        return item_mat


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    导入相关数据集
    """

    def __init__(self):
        # train or test
        super().__init__()
        utils.cprint(f'loading [{self.path}]')
        # print(config)
        self.split = utils.A_split
        self.folds = utils.A_n_fold
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.m_item = 0
        train_file = self.path + '/train.txt'
        test_file = self.path + '/test.txt'
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.train_data_size = 0
        self.testDataSize = 0
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.train_data_size += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        with open(test_file) as f:
            for l in f.readlines():
                l = l.strip().split(' ')
                if len(l) > 1:  # amazon-book有个例外
                    items = [int(i) for i in l[1:]]
                    # print(items)
                    uid = int(l[0])
                    #  print(uid)
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        print('number of user:{}'.format(self.n_user))
        print('number of item:{}'.format(self.m_item))
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        self.Graph = None
        print(f"{self.train_data_size} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{utils.dataset_name} Sparsity : {(self.train_data_size + self.testDataSize) / self.n_users / self.m_items}")
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        # 计算度矩阵
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self._allPosScores = None
        self.__testDict = self.__build_test()
        print(f"{utils.dataset_name} is ready to go")
        # if os.path.isfile(self.path + '/user_mat.npz'):
        #     self.user_mat = sp.load_npz(self.path + '/user_mat.npz')
        # else:
        #     self.user_mat = None
        #
        # if os.path.isfile(self.path + '/item_mat.npz'):
        #     self.item_mat = sp.load_npz(self.path + '/item_mat.npz')
        # else:
        #     self.item_mat = None

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.train_data_size

    @property
    def testDict(self):
        """
        获取的就是字典，键为用户，值为对应的物品
        :return:
        """
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def allPosScores(self):
        return self._allPos

    def _split_A_hat(self, A):
        # 意思是把图给分割了，画成了很多等份进行训练
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(utils.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        '''
        转化为
        tensor形式
        '''
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        mat = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        return mat

    def getSparseGraph(self, add_self=False):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                norm_adj = pre_adj_mat
                if add_self:
                    pre_adj_mat_self = sp.load_npz(self.path + '/s_pre_adj_mat_self.npz')
                    norm_adj_self = pre_adj_mat_self
                print("successfully loaded...")
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print("costing {:.4f} s, saved norm_mat...".format(end - s))
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
                if add_self:
                    adj_mat_self = adj_mat + sp.eye(adj_mat.shape[0])
                    rowsum_self = np.array(adj_mat_self.sum(axis=1))
                    d_inv_self = np.power(rowsum_self, -0.5).flatten()
                    d_inv_self[np.isinf(d_inv_self)] = 0.
                    d_mat_self = sp.diags(d_inv_self)
                    norm_adj_self = d_mat_self.dot(adj_mat_self)
                    norm_adj_self = norm_adj_self.dot(d_mat_self)
                    norm_adj_self = norm_adj_self.tocsr()
                    sp.save_npz(self.path + '/s_pre_adj_mat_self.npz', norm_adj_self)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                if add_self:
                    self.Graph_self = self._split_A_hat(norm_adj_self)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(utils.device)
                if add_self:
                    self.Graph_self = self._convert_sp_mat_to_sp_tensor(norm_adj_self)
                    self.Graph_self = self.Graph.coalesce().to(utils.device)
                print("don't split the matrix")
        if add_self:
            return self.Graph, self.Graph_self
        return self.Graph

    def get_diag_unit_graph(self):
        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
            norm_adj = pre_adj_mat
            print("successfully loaded...")
        except:
            print("generating adjacency matrix")
            s = time()
            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todok()
            # d_inv = np.power(rowsum, -0.5).flatten()
            # d_inv[np.isinf(d_inv)] = 0.
            # d_mat = sp.diags(d_inv)
            norm_adj = adj_mat
            # norm_adj = d_mat.dot(adj_mat)
            # norm_adj = norm_adj.dot(d_mat)
            # norm_adj = norm_adj.tocsr()
            end = time()
            print("costing {:.4f} s, saved norm_mat...".format(end - s))
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
        rowsum = np.array(norm_adj.sum(axis=1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat = self._convert_sp_mat_to_sp_tensor(d_mat)
        d_mat = d_mat.coalesce().to(utils.device)
        graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        graph = graph.coalesce().to(utils.device)
        return graph, d_mat

    # def getSparseGraph1(self):
    #     g1 = self.getSparseGraph()
    #     m = g1.shape[0]
    #     row = torch.arange(0, m)
    #     col = torch.arange(0, m)
    #     index = torch.stack([row, col])
    #     data = torch.ones(m)
    #     I = torch.sparse.FloatTensor(index, data,torch.Size((m,m)))
    #     return g1, g1 + I

    # index = torch.stack([row, col])
    # data = torch.FloatTensor(coo.data)
    # return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
    @trainDataSize.setter
    def trainDataSize(self, value):
        self._trainDataSize = value

    def getUserGraph(self, dense=False):
        if os.path.exists(self.path + '/user_mat.npz'):
            user_mat = sp.load_npz(self.path + '/user_mat.npz')
        else:
            user_mat = self.UserItemNet.dot(self.UserItemNet.transpose())
            user_mat = utils.preprocess_adj_graph(user_mat)
            sp.save_npz(self.path + '/user_mat.npz', user_mat)
        if dense:
            user_mat = user_mat.todense()
            user_mat = torch.Tensor(user_mat)
            return user_mat
        else:
            user_mat = self._convert_sp_mat_to_sp_tensor(user_mat)
        user_mat = user_mat.coalesce().to(utils.device)
        return user_mat

    def getItemGraph(self, dense=False):
        if os.path.exists(self.path + '/item_mat.npz'):
            item_mat = sp.load_npz(self.path + '/item_mat.npz')
        else:
            item_mat = (self.UserItemNet.transpose()).dot(self.UserItemNet)
            item_mat = utils.preprocess_adj_graph(item_mat)
            sp.save_npz(self.path + '/item_mat.npz', item_mat)
        if dense:
            item_mat = item_mat.todense()
            item_mat = torch.Tensor(item_mat)
            return item_mat
        else:
            item_mat = self._convert_sp_mat_to_sp_tensor(item_mat)
        item_mat = item_mat.coalesce().to(utils.device)
        return item_mat
