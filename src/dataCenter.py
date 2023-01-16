import sys
import os

from collections import defaultdict
from scipy.spatial.distance import pdist,squareform
import scipy.sparse as sp
import json

import random
#random.seed(1)
import numpy as np
#np.random.seed(1)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
IMBALANCE_THRESH = 101


def refine_label_order(labels):
    max_label = labels.max()
    j = 0

    for i in range(labels.max(),0,-1):
        if sum(labels==i) >= IMBALANCE_THRESH and i>j:
            while sum(labels==j) >= IMBALANCE_THRESH and i>j:
                j = j+1
            if i > j:
                head_ind = labels == j
                tail_ind = labels == i
                labels[head_ind] = i
                labels[tail_ind] = j
                j = j+1
            else:
                break
        elif i <= j:
            break

    return labels

class DataCenter(object):
	"""docstring for DataCenter"""
	def __init__(self, config):
		super(DataCenter, self).__init__()
		self.config = config
		self.indexMax = 0


	def src_smote(self, dataSet='cora', targetLabel = 1, portion=1.0, smoteorover = 'SMOTE'):
		#c_largest = self.labels.max().item()
		#adj_back = self.adj_lists.to_dense()
		#adj_back = self.adj_lists # 转换成 dense
		#adj_back = []
		indexMax = self.indexMax
		chosen = None
		new_features = None
		self.labels = np.array(self.labels)
		self.idx_train = np.array(self.idx_train)
		self.features = np.array(self.features)
		# ipdb.set_trace()
		#avg_number = int(self.dataSet+'_train'.shape[0] / (c_largest + 1))
		i = targetLabel
		new_chosen = self.idx_train[(self.labels == targetLabel)[self.idx_train]]
		self.smote_before_len = new_chosen
		c_portion = int(portion)
		portion_rest = portion - c_portion
		c_portion = 1

		random.seed(0)

		chonsenIndex = []
		for j in range(c_portion):
			num = int(new_chosen.shape[0])
			new_chosen = new_chosen[:num]

			chosen_embed = self.features[new_chosen, :]
			distance = squareform(pdist(chosen_embed))
			np.fill_diagonal(distance, distance.max() + 100)

			idx_neighbor = distance.argmin(axis=-1)
			if smoteorover == 'SMOTE':
				interp_place = random.random()
				embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place
			else:
				embed = chosen_embed
			if chosen is None:
				chosen = new_chosen
				new_features = embed
			else:
				#chosen = torch.cat((chosen, new_chosen), 0)
				#new_features = torch.cat((new_features, embed), 0)
				chosen.append(new_chosen) # todo 注意修改
				new_features.append(embed)

			count = 0
			for cc in range(0, len(embed)):
				chonsenIndex.append(count)
				count += 1
		adj = self.adj_lists
		random.shuffle(chonsenIndex)
		self.features = list(self.features)
		self.labels = list(self.labels)
		self.idx_train = list(self.idx_train)

		self.smote_idx = set()
		for i in range(0, ( int( len(chonsenIndex) * portion))  ):
			choseEmb = embed[chonsenIndex[i]] # SMOTE
			self.features.append(choseEmb) # 加入新生成的数据
			self.labels.append(targetLabel)  # 加入标签
			new_target_chosen = indexMax + i + 1 # 加入连接矩阵
			adj[new_target_chosen] = self.adj_lists[ new_chosen[chonsenIndex[i]] ]
			self.idx_train.append(new_target_chosen)
			self.smote_idx.add(new_target_chosen)
			#indexMax += 1
		# 修改连接矩阵

		#setattr(self, dataSet + '_test', test_indexs)
		#setattr(self, dataSet + '_val', val_indexs)
		setattr(self, dataSet + '_train', np.array(self.idx_train))

		setattr(self, dataSet + '_feats', np.array(self.features))
		setattr(self, dataSet + '_labels', np.array(self.labels))
		setattr(self, dataSet + '_adj_lists', adj)

	def mul_src_smote(self, dataSet='cora', targetLabel = 1, portion=1.0, smoteorover = 'SMOTE'):
		#c_largest = self.labels.max().item()
		#adj_back = self.adj_lists.to_dense()
		#adj_back = self.adj_lists # 转换成 dense
		#adj_back = []

		indexMax = self.indexMax
		chosen = None
		new_features = None
		self.labels = np.array(self.labels)
		self.idx_train = np.array(self.idx_train)
		self.features = np.array(self.features)
		# ipdb.set_trace()
		#avg_number = int(self.dataSet+'_train'.shape[0] / (c_largest + 1))
		i = targetLabel

		new_chosen = self.new_chosed


		self.smote_before_len = new_chosen
		c_portion = int(portion)
		portion_rest = portion - c_portion
		c_portion = 1

		random.seed(0)

		chonsenIndex = []
		for j in range(c_portion):
			num = int(new_chosen.shape[0])
			new_chosen = new_chosen[:num]

			chosen_embed = self.features[new_chosen, :]
			distance = squareform(pdist(chosen_embed))
			np.fill_diagonal(distance, distance.max() + 100)

			idx_neighbor = distance.argmin(axis=-1)
			if smoteorover == 'SMOTE':
				interp_place = random.random()
				embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place
			else:
				embed = chosen_embed
			if chosen is None:
				chosen = new_chosen
				new_features = embed
			else:
				#chosen = torch.cat((chosen, new_chosen), 0)
				#new_features = torch.cat((new_features, embed), 0)
				chosen.append(new_chosen) # todo 注意修改
				new_features.append(embed)

			count = 0
			for cc in range(0, len(embed)):
				chonsenIndex.append(count)
				count += 1
		adj = self.adj_lists
		random.shuffle(chonsenIndex)
		self.features = list(self.features)
		self.labels = list(self.labels)
		self.idx_train = list(self.idx_train)

		self.smote_idx = set()
		for i in range(0, ( int( len(chonsenIndex) * portion))  ):
			choseEmb = embed[chonsenIndex[i]] # SMOTE
			self.features.append(choseEmb) # 加入新生成的数据
			self.labels.append(self.labels[chonsenIndex[i]])  # 加入标签
			new_target_chosen = indexMax + i + 1 # 加入连接矩阵
			adj[new_target_chosen] = self.adj_lists[ new_chosen[chonsenIndex[i]] ]
			self.idx_train.append(new_target_chosen)
			self.smote_idx.add(new_target_chosen)
			#indexMax += 1
		# 修改连接矩阵

		#setattr(self, dataSet + '_test', test_indexs)
		#setattr(self, dataSet + '_val', val_indexs)
		setattr(self, dataSet + '_train', np.array(self.idx_train))

		setattr(self, dataSet + '_feats', np.array(self.features))
		setattr(self, dataSet + '_labels', np.array(self.labels))
		setattr(self, dataSet + '_adj_lists', adj)

	def mul_transfer(self, dataSet):

		setattr(self, dataSet + '_labels', np.array(list(np.array(self.labels))))



	def src_over(self, dataSet='cora', targetLabel = 1, portion=1.0):
		c_largest = self.labels.max().item()
		#adj_back = self.adj_lists.to_dense()
		adj_back = self.adj_lists # 转换成 dense
		#adj_back = []
		indexMax = self.indexMax
		chosen = None
		new_features = None

		# ipdb.set_trace()
		#avg_number = int(self.dataSet+'_train'.shape[0] / (c_largest + 1))
		i = targetLabel
		new_chosen = self.idx_train[(self.labels == targetLabel)[self.idx_train]]
		self.smote_before_len = new_chosen
		c_portion = int(portion)
		portion_rest = portion - c_portion
		c_portion = 1

		random.seed(0)

		chonsenIndex = []
		for j in range(c_portion):
			num = int(new_chosen.shape[0])
			new_chosen = new_chosen[:num]

			chosen_embed = self.features[new_chosen, :]
			distance = squareform(pdist(chosen_embed))
			np.fill_diagonal(distance, distance.max() + 100)

			idx_neighbor = distance.argmin(axis=-1)

			interp_place = random.random()
			embed = chosen_embed + (chosen_embed[idx_neighbor, :] - chosen_embed) * interp_place

			if chosen is None:
				chosen = new_chosen
				new_features = embed
			else:
				#chosen = torch.cat((chosen, new_chosen), 0)
				#new_features = torch.cat((new_features, embed), 0)
				chosen.append(new_chosen) # todo 注意修改
				new_features.append(embed)

			count = 0
			for cc in range(0, len(embed)):
				chonsenIndex.append(count)
				count += 1
		adj = self.adj_lists
		random.shuffle(chonsenIndex)
		self.features = list(self.features)
		self.labels = list(self.labels)
		self.idx_train = list(self.idx_train)

		self.smote_idx = set()
		for i in range(0, ( int( len(chonsenIndex) * portion))  ):
			choseEmb = embed[chonsenIndex[i]] # SMOTE
			self.features.append(choseEmb) # 加入新生成的数据
			self.labels.append(targetLabel)  # 加入标签
			new_target_chosen = indexMax + i + 1 # 加入连接矩阵
			adj[new_target_chosen] = self.adj_lists[ new_chosen[chonsenIndex[i]] ]
			self.idx_train.append(new_target_chosen)
			self.smote_idx.add(new_target_chosen)
			#indexMax += 1
		# 修改连接矩阵

		#setattr(self, dataSet + '_test', test_indexs)
		#setattr(self, dataSet + '_val', val_indexs)
		setattr(self, dataSet + '_train', np.array(self.idx_train))

		setattr(self, dataSet + '_feats', np.array(self.features))
		setattr(self, dataSet + '_labels', np.array(self.labels))
		setattr(self, dataSet + '_adj_lists', adj)



	def load_dataSet(self, dataSet='cora', targetLabel = 1, imb_ratio = .4):
		if dataSet == 'cora':
			cora_content_file = self.config['file_path.cora_content']
			cora_cite_file = self.config['file_path.cora_cite']

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			label_map = {} # map label to Label_ID
			with open(cora_content_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					feat_data.append([float(x) for x in info[1:-1]])
					node_map[info[0]] = i
					if not info[-1] in label_map:
						label_map[info[-1]] = len(label_map)
					labels.append(label_map[info[-1]])
			print (label_map)
			feat_data = np.asarray(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			with open(cora_cite_file) as fp:
				for i,line in enumerate(fp):
					info = line.strip().split()
					assert len(info) == 2
					paper1 = node_map[info[0]]
					paper2 = node_map[info[1]]
					adj_lists[paper1].add(paper2)
					adj_lists[paper2].add(paper1)
					if int(paper1) > self.indexMax:
						self.indexMax = int(paper1)
					if int(paper2) > self.indexMax:
						self.indexMax = int(paper2)

			assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._imb_split_data(feat_data.shape[0], labels, targetLabel, imb_ratio) # todo 在dataloader 进行imb

			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			setattr(self, dataSet+'_feats', feat_data)
			setattr(self, dataSet+'_labels', labels)
			setattr(self, dataSet+'_adj_lists', adj_lists)
			self.idx_train = train_indexs
			self.labels = labels
			self.features = feat_data
			self.adj_lists = adj_lists

			self.adj_dense = np.zeros((self.indexMax + 1, self.indexMax + 1))
			for k, vs in self.adj_lists.items():
				for v in vs:
					self.adj_dense[k][v] = 1
			'''
			for i in self.adj_dense[0]:
				print (i)
			print (self.adj_lists[0])
			exit()
			'''

		elif dataSet == 'pubmed':
			pubmed_content_file = self.config['file_path.pubmed_paper']
			pubmed_cite_file = self.config['file_path.pubmed_cites']

			feat_data = []
			labels = [] # label sequence of node
			node_map = {} # map node to Node_ID
			with open(pubmed_content_file) as fp:
				fp.readline()
				feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
				for i, line in enumerate(fp):
					info = line.split("\t")
					node_map[info[0]] = i
					labels.append(int(info[1].split("=")[1])-1)
					tmp_list = np.zeros(len(feat_map)-2)
					for word_info in info[2:-1]:
						word_info = word_info.split("=")
						tmp_list[feat_map[word_info[0]]] = float(word_info[1])
					feat_data.append(tmp_list)
			
			feat_data = np.asarray(feat_data)
			labels = np.asarray(labels, dtype=np.int64)
			
			adj_lists = defaultdict(set)
			with open(pubmed_cite_file) as fp:
				fp.readline()
				fp.readline()
				for line in fp:
					info = line.strip().split("\t")
					paper1 = node_map[info[1].split(":")[1]]
					paper2 = node_map[info[-1].split(":")[1]]
					adj_lists[paper1].add(paper2)
					adj_lists[paper2].add(paper1)
					if int(paper1) > self.indexMax:
						self.indexMax = int(paper1)
					if int(paper2) > self.indexMax:
						self.indexMax = int(paper2)

			
			assert len(feat_data) == len(labels) == len(adj_lists)
			test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])
			from collections import Counter
			colors = ['red', 'blue', 'red', 'green', 'blue', 'blue']
			c = Counter(labels)
			print(dict(c))
			#exit()
			setattr(self, dataSet+'_test', test_indexs)
			setattr(self, dataSet+'_val', val_indexs)
			setattr(self, dataSet+'_train', train_indexs)

			setattr(self, dataSet+'_feats', feat_data)
			setattr(self, dataSet+'_labels', labels)
			setattr(self, dataSet+'_adj_lists', adj_lists)
			self.idx_train = train_indexs
			self.labels = labels
			self.features = feat_data
			self.adj_lists = adj_lists

			self.adj_dense = np.zeros((self.indexMax + 1, self.indexMax + 1))
			for k, vs in self.adj_lists.items():
				for v in vs:
					self.adj_dense[k][v] = 1

		elif dataSet == 'blog':
			from scipy.io import loadmat

			mat = loadmat(self.config['file_path.blogmat'])
			adj = mat['network']
			label = mat['group']
			cx = adj.tocoo()
			adj_lists = defaultdict(set)

			for paper1, paper2, d in zip(cx.row, cx.col, cx.data):
				adj_lists[paper1].add(paper2)
				#adj_lists[paper2].add(paper1)
				if paper1 > self.indexMax:
					self.indexMax = paper1
				if paper2 > self.indexMax:
					self.indexMax = paper2

			embed = np.loadtxt(self.config['file_path.blogemb'])
			feature = np.zeros((embed.shape[0], embed.shape[1] - 1))
			feature[embed[:, 0].astype(int), :] = embed[:, 1:]

			features = normalize(feature)
			labels = np.array(label.todense().argmax(axis=1)).squeeze()

			labels[labels > 16] = labels[labels > 16] - 1

			print("change labels order, imbalanced classes to the end.")
			labels = refine_label_order(labels)
			feat_data = np.asarray(features)
			labels = np.asarray(labels, dtype=np.int64)
			'''
			from sklearn.preprocessing import MinMaxScaler
			scaler = MinMaxScaler()
			scaler.fit(feat_data)
			feat_data = np.asarray(scaler.transform(feat_data))
			'''
			self.adj_dense = adj.toarray()

			test_indexs, val_indexs, train_indexs = self._split_data(feat_data.shape[0])

			from collections import Counter
			c = Counter(labels)
			print(dict(c))
			#exit()
			setattr(self, dataSet + '_test', test_indexs)
			setattr(self, dataSet + '_val', val_indexs)
			setattr(self, dataSet + '_train', train_indexs)

			setattr(self, dataSet + '_feats', feat_data)
			setattr(self, dataSet + '_labels', labels)
			setattr(self, dataSet + '_adj_lists', adj_lists)
			self.idx_train = train_indexs
			self.labels = labels
			self.features = feat_data
			self.adj_lists = adj_lists

		elif dataSet == 'ppi':
			from scipy.io import loadmat
			dirDir = self.config['file_path.ppi_data']
			x0 = np.load(dirDir + 'train_feats.npy')
			y0 = np.load(dirDir + 'train_labels.npy')
			idx0 =np.load(dirDir + 'train_graph_id.npy')
			with open(dirDir + 'train_graph.json', 'r') as f:

				G1 = json.load(f)

			# 训练集 稍微拆分一下
			subTrain = [1,2]
			x1 = []
			y1 = []
			idx1 = []
			subIndx = []
			for i in range(0, len(idx0)):
				if idx0[i] in subTrain:
				#if subTrain:

					x1.append(x0[i])
					y1.append(y0[i])
					idx1.append(idx0[i])
					subIndx.append(i)

			from sklearn.preprocessing import MinMaxScaler
			scaler = MinMaxScaler()
			scaler.fit(x1)
			x1 = list(scaler.transform(x1))

			z =1

			x2 = np.load(dirDir + 'test_feats.npy')
			y2 = np.load(dirDir + 'test_labels.npy')
			idx2 =np.load(dirDir + 'test_graph_id.npy')
			with open(dirDir + 'test_graph.json', 'r') as f:

				G2 = json.load(f)

			x3 = np.load(dirDir + 'valid_feats.npy')
			y3 = np.load(dirDir + 'valid_labels.npy')
			idx3 =np.load(dirDir + 'valid_graph_id.npy')
			with open(dirDir + 'valid_graph.json', 'r') as f:

				G3 = json.load(f)
			# For trandustive
			x2 = list(scaler.transform(x2))
			x3 = list(scaler.transform(x3))

			z =1
			'''	
			t = np.zeros(121)
			for ii in y1:
				t += ii
				z = 1
			for iii in t:
				print (iii)
			exit()
			'''
			feat_data = list(x1) + list(x2) + list(x3)
			labels = list(y1) + list(y2) + list(y3)

			train_indexs = []
			test_indexs = []
			val_indexs = []
			target = 86
			new_chosed = []
			for i in range(0, len(idx1)):
				train_indexs.append(i)
				if labels[i][target] !=0:
					new_chosed.append(i)
			for i in range(len(idx1), len(idx1) + len(idx2)):
				test_indexs.append(i)
			for i in range(len(idx1) + len(idx2), len(idx1) + len(idx2) + len(idx3)):
				val_indexs.append(i)

			z = 1
			adj_lists = defaultdict(set)
			train_link = G1['links']
			for ll in train_link:
				paper1 = ll['source']
				if paper1 not in subIndx:
					continue
				paper2 = ll['target']
				adj_lists[paper1].add(paper2)

			test_link = G2['links']
			for ll in test_link:
				paper1 = ll['source'] + len(idx1)
				paper2 = ll['target'] + len(idx1)
				adj_lists[paper1].add(paper2)

			valid_link = G3['links']
			for ll in valid_link:
				paper1 = ll['source'] + len(idx1) + len(idx2)
				paper2 = ll['target'] + len(idx1) + len(idx2)
				adj_lists[paper1].add(paper2)
			feat_data = np.asarray(feat_data)

			tmp = np.array(labels)
			# 数据清洗一下 扫除孤立点
			delentNode = []
			for k, v in adj_lists.items():
				if len(v) < 2:
					delentNode.append(k)
			z = 1
			#delentNode = []  # 在二层的时候  会有 报错问题


			#test_indexs = [x for x in test_indexs if x not in delentNode]
			val_indexs = [x for x in val_indexs if x not in delentNode]
			train_indexs = [x for x in train_indexs if x not in delentNode]


			setattr(self, dataSet + '_test', test_indexs)
			setattr(self, dataSet + '_val', val_indexs)
			setattr(self, dataSet + '_train', train_indexs)

			setattr(self, dataSet + '_feats', feat_data)
			setattr(self, dataSet + '_labels', labels)
			setattr(self, dataSet + '_adj_lists', adj_lists)

			self.idx_train = train_indexs
			self.labels = labels
			self.features = feat_data
			self.adj_lists = adj_lists
			self.new_chosed = np.array(new_chosed)

		elif dataSet == 'wiki':
			dirDir = self.config['file_path.wiki_data']
			data = json.load(open(dirDir + 'data.json'))
			feat_data = list(np.array(data['features']))
			labels = np.asarray(data['labels'], dtype=np.int64)

			links = data['links']
			adj_lists = defaultdict(set)

			for i in range(0, len(links)):
				target = links[i]
				if i == 7625:
					z = 1
				for t in target:
					if i != t:
						adj_lists[i].add(t)
						#adj_lists[t].add(i) # todo

			train_indexs = [i for i in range(len(data['train_masks'][1])) if data['train_masks'][1][i] == True] # todo 这里有十等分交叉验证
			val_indexs = [i for i in range(len(data['val_masks'][1])) if data['val_masks'][1][i] == True]
			test_indexs = [i for i in range(len(data['test_mask'])) if data['test_mask'][i] == True]

			#train_indexs1 = [i for i in range(len(data['train_masks'][1])) if data['train_masks'][1][i] == True] # todo 这里有十等分交叉验证
			#val_indexs1 = [i for i in range(len(data['val_masks'][1])) if data['val_masks'][1][i] == True]

			#train_indexs.extend(train_indexs1)
			#val_indexs.extend(val_indexs1)

			#train_indexs =  [i for i in range(len(data['test_mask'])) if data['test_mask'][i] == True]# todo 这里有十等分交叉验证
			#val_indexs =  [i for i in range(len(data['test_mask'])) if data['test_mask'][i] == True]

			from sklearn.preprocessing import MinMaxScaler
			scaler = MinMaxScaler()
			scaler.fit(feat_data)
			feat_data = list(scaler.transform(feat_data))

			# 数据清洗一下 扫除孤立点
			delentNode = []

			for k, v in adj_lists.items():
				if len(v) < 2:
					delentNode.append(k)
			delentNode = []

			z = 1

			test_indexs = [x for x in test_indexs if x not in delentNode]
			val_indexs = [x for x in val_indexs if x not in delentNode]
			train_indexs = [x for x in train_indexs if x not in delentNode]

			test_indexs = [x for x in test_indexs if x  in adj_lists.keys()]
			val_indexs = [x for x in val_indexs if x  in adj_lists.keys()]
			train_indexs = [x for x in train_indexs if x in adj_lists.keys()]

			from collections import Counter
			print (Counter(labels))
			#exit()
			setattr(self, dataSet + '_test', test_indexs)
			setattr(self, dataSet + '_val', val_indexs)
			setattr(self, dataSet + '_train', train_indexs)
			setattr(self, dataSet + '_feats', feat_data)
			setattr(self, dataSet + '_labels', labels)
			setattr(self, dataSet + '_adj_lists', adj_lists)

			self.idx_train = train_indexs
			self.labels = labels
			self.features = feat_data
			self.adj_lists = adj_lists


	def _split_data(self, num_nodes, test_split = 3, val_split = 6):
		from collections import Counter

		#np.random.seed(0) # 设置随机种子 防止震荡
		rand_indices = np.random.permutation(num_nodes)

		test_size = num_nodes // test_split
		val_size = num_nodes // val_split
		train_size = num_nodes - (test_size + val_size)

		test_indexs = rand_indices[:test_size]
		val_indexs = rand_indices[test_size:(test_size+val_size)]
		train_indexs = rand_indices[(test_size+val_size):]

		print (len(train_indexs))
		print (len(val_indexs))
		print (len(test_indexs))
		#print (val_indexs)
		listResult = {'trainIndex': train_indexs,
					  'val_indexs': val_indexs, 'test_indexs': test_indexs}
		#np.save('blog.npy', listResult)
		#exit()

		return test_indexs, val_indexs, train_indexs

	def _imb_split_data(self, num_nodes, labels, targetLabel = 1, imb_ratio = .5, test_split=3, val_split=6):
		from collections import Counter

		np.random.seed(1)  # 设置随机种子 防止震荡
		rand_indices = np.random.permutation(num_nodes)

		test_size = num_nodes // test_split
		val_size = num_nodes // val_split
		train_size = num_nodes - (test_size + val_size)

		test_indexs = rand_indices[:test_size]
		val_indexs = rand_indices[test_size:(test_size + val_size)]
		train_indexs = rand_indices[(test_size + val_size):]
		# 在这里存


		# 简单点
		train_imb_index = []
		train_imb_target_tmp = []
		for i in train_indexs:
			label = labels[i]
			if label != targetLabel:
				train_imb_index.append(i)
			else:
				train_imb_target_tmp.append(i)

		train_imb_target = random.sample(train_imb_target_tmp, int(len(train_imb_target_tmp) * imb_ratio) )
		print (len(train_imb_target_tmp) - int(len(train_imb_target_tmp) * imb_ratio))
		train_imb_index.extend(train_imb_target)
		random.shuffle(train_imb_index)
		train_imb_index = np.array(train_imb_index)

		print(len(train_imb_index))
		print(len(val_indexs))
		print(len(test_indexs))
		# print (val_indexs)
		# save
		listResult = {'trainIndex': train_imb_index,
					  'val_indexs':val_indexs,'test_indexs':test_indexs}
		np.save('listResult_0.5.npy', listResult)
		#exit()
		return test_indexs, val_indexs, train_imb_index



