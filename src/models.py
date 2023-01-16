import sys, os
import torch
import random

import torch.nn as nn
import torch.nn.functional as F
import copy

class Classification(nn.Module):

	def __init__(self, emb_size, num_classes):
		super(Classification, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.layer = nn.Sequential(
								nn.Linear(emb_size, num_classes)	  
								#nn.ReLU()
							)
		self.init_params()

	def init_params(self):
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, embeds):
		logists = torch.log_softmax(self.layer(embeds), 1)
		return logists

class MultiLabelClassification(nn.Module):

	def __init__(self, emb_size, num_classes):
		super(MultiLabelClassification, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.layer = nn.Sequential(
								nn.Linear(emb_size, num_classes)
								#nn.ReLU()
							)
		self.init_params()

	def init_params(self):
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, embeds):
		logists = torch.sigmoid(self.layer(embeds))
		#logists = torch.where(torch.isnan(logists), torch.full_like(logists, 0), logists)
		return logists

# class Classification(nn.Module):

# 	def __init__(self, emb_size, num_classes):
# 		super(Classification, self).__init__()

# 		self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
# 		self.init_params()

# 	def init_params(self):
# 		for param in self.parameters():
# 			nn.init.xavier_uniform_(param)

# 	def forward(self, embeds):
# 		logists = torch.log_softmax(torch.mm(embeds,self.weight), 1)
# 		return logists

class UnsupervisedLoss(object):
	"""docstring for UnsupervisedLoss"""
	def __init__(self, adj_lists, train_nodes, device):
		super(UnsupervisedLoss, self).__init__()
		self.Q = 10
		self.N_WALKS = 6  # 在blog 要多一些？ 原本是 6
		self.WALK_LEN = 1
		self.N_WALK_LEN = 5
		self.MARGIN = 3
		self.adj_lists = adj_lists
		self.train_nodes = train_nodes
		self.device = device

		self.target_nodes = None
		self.positive_pairs = []
		self.negtive_pairs = []
		self.node_positive_pairs = {}
		self.node_negtive_pairs = {}
		self.unique_nodes_batch = []

	def get_loss_sage(self, embeddings, nodes):
		assert len(embeddings) == len(self.unique_nodes_batch)
		assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
		node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			if len(pps) == 0 or len(nps) == 0:
				continue

			# Q * Exception(negative score)
			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
			#print(neg_score)

			# multiple positive score
			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			pos_score = torch.log(torch.sigmoid(pos_score))
			#print(pos_score)

			nodes_score.append(torch.mean(- pos_score - neg_score).view(1,-1))
				
		loss = torch.mean(torch.cat(nodes_score, 0))
		
		return loss

	def get_loss_margin(self, embeddings, nodes):
		assert len(embeddings) == len(self.unique_nodes_batch)
		assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
		node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

		nodes_score = []
		assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
		for node in self.node_positive_pairs:
			pps = self.node_positive_pairs[node]
			nps = self.node_negtive_pairs[node]
			if len(pps) == 0 or len(nps) == 0:
				continue

			indexs = [list(x) for x in zip(*pps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)

			indexs = [list(x) for x in zip(*nps)]
			node_indexs = [node2index[x] for x in indexs[0]]
			neighb_indexs = [node2index[x] for x in indexs[1]]
			neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
			neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)

			nodes_score.append(torch.max(torch.tensor(0.0).to(self.device), neg_score-pos_score+self.MARGIN).view(1,-1))
			# nodes_score.append((-pos_score - neg_score).view(1,-1))

		loss = torch.mean(torch.cat(nodes_score, 0),0)

		# loss = -torch.log(torch.sigmoid(pos_score))-4*torch.log(torch.sigmoid(-neg_score))
		
		return loss


	def extend_nodes(self, nodes, num_neg=6):
		self.positive_pairs = []
		self.node_positive_pairs = {}
		self.negtive_pairs = []
		self.node_negtive_pairfs = {}

		self.target_nodes = nodes
		self.get_positive_nodes(nodes)
		# print(self.positive_pairs)
		self.get_negtive_nodes(nodes, num_neg)
		# print(self.negtive_pairs)
		self.unique_nodes_batch = list(set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
		tmp1 = set(self.target_nodes)
		tmp2 = set(self.unique_nodes_batch)

		#assert set(self.target_nodes) < set(self.unique_nodes_batch)  # blog
		return self.unique_nodes_batch

	def get_positive_nodes(self, nodes):
		return self._run_random_walks(nodes)

	def get_negtive_nodes(self, nodes, num_neg):
		for node in nodes:
			neighbors = set([node])
			frontier = set([node])
			for i in range(self.N_WALK_LEN):
				current = set()
				for outer in frontier:
					current |= self.adj_lists[int(outer)]
				frontier = current - neighbors
				neighbors |= current
			far_nodes = set(self.train_nodes) - neighbors
			neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
			self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
			self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
		return self.negtive_pairs

	def _run_random_walks(self, nodes):
		for node in nodes:
			if len(self.adj_lists[int(node)]) == 0:
				continue
			cur_pairs = []
			for i in range(self.N_WALKS):
				curr_node = node
				for j in range(self.WALK_LEN):
					neighs = self.adj_lists[int(curr_node)]
					next_node = random.choice(list(neighs))
					# self co-occurrences are useless
					if next_node != node and next_node in self.train_nodes:
						self.positive_pairs.append((node,next_node))
						cur_pairs.append((node,next_node))
					curr_node = next_node

			self.node_positive_pairs[node] = cur_pairs
		return self.positive_pairs
		

class SageLayer(nn.Module):
	"""
	Encodes a node's using 'convolutional' GraphSage approach
	"""
	def __init__(self, input_size, out_size, gcn=False): 
		super(SageLayer, self).__init__()

		self.input_size = input_size
		self.out_size = out_size


		self.gcn = gcn
		self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))

		self.init_params()

	def init_params(self):
		for param in self.parameters():
			nn.init.xavier_uniform_(param)

	def forward(self, self_feats, aggregate_feats, neighs=None):
		"""
		Generates embeddings for a batch of nodes.

		nodes	 -- list of nodes
		"""
		if not self.gcn:
			combined = torch.cat([self_feats, aggregate_feats], dim=1)
		else:
			combined = aggregate_feats
		t1 = combined.t()
		t2 = self.weight.mm(t1)
		t3 = F.relu(t2)
		t4 = t3.t()
		combined = F.relu6(self.weight.mm(combined.t())).t()
		return combined

class GraphSage(nn.Module):
	"""docstring for GraphSage"""
	def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, gcn=False, agg_func='MEAN'):
		super(GraphSage, self).__init__()

		self.input_size = input_size
		self.out_size = out_size
		self.num_layers = num_layers
		self.gcn = gcn
		self.device = device
		self.agg_func = agg_func

		self.raw_features = raw_features
		self.adj_lists = adj_lists

		for index in range(1, num_layers+1):
			layer_size = out_size if index != 1 else input_size
			setattr(self, 'sage_layer'+str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

	def forward(self, nodes_batch, need_previous = False):
		"""
		Generates embeddings for a batch of nodes.
		nodes_batch	-- batch of nodes to learn the embeddings
		"""
		lower_layer_nodes = list(nodes_batch)
		nodes_batch_layers = [(lower_layer_nodes,)] # 第0个 是 近的，后面的是远的
		# self.dc.logger.info('get_unique_neighs.')
		for i in range(self.num_layers):
			lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes= self._get_unique_neighs_list(lower_layer_nodes)
			nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict)) # nodes_batch_layer 是所有的信息
			# 注意！这个insert是插入最前 所以 是从后往前搞

		assert len(nodes_batch_layers) == self.num_layers + 1

		pre_hidden_embs = self.raw_features # 第一个是 feature .. 感觉不大对啊？哦？
		record_embs = []
		PRE_HIDDEN_EMBS = []
		PRE_FEATURES = []
		SAMP_NEIGHS = []
		MASK = []
		sage_pre_feature = []
		for index in range(1, self.num_layers+1): #
			nb = nodes_batch_layers[index][0]  # 节点及其 邻居  从 1 开始
			pre_neighs = nodes_batch_layers[index-1] # 之前 的
			# self.dc.logger.info('aggregate_feats.')
			aggregate_feats, pre_feature, samp_neighs, mask = self.aggregate(nb, pre_hidden_embs, pre_neighs)

			aggregate_feats = torch.where(torch.isnan(aggregate_feats), torch.full_like(aggregate_feats, 0.5), aggregate_feats) # todo 物理操作

			_, pre_feature2, _, _ = self.aggregate(nb, self.raw_features, pre_neighs)
			PRE_FEATURES.append(pre_feature2) # 如果这个只要原始特征的话 在pre-feature改变前定义

			# 如果层数是1 aggregate_feats 是特征 cur_hidden_embs 是聚合特征
			# 好吧 如果层数为1 只有一个 samp_neighs 是邻居信息 可以参考 agg里面怎么操作的
			sage_layer = getattr(self, 'sage_layer'+str(index))
			if index > 1:
				nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
			# self.dc.logger.info('sage_layer.')

			cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
										aggregate_feats=aggregate_feats)
			sage_pre_feature.append(pre_hidden_embs[nb])
			pre_hidden_embs = cur_hidden_embs
			record_embs.append(cur_hidden_embs)
			# 多层进行记录

			PRE_HIDDEN_EMBS.append(pre_hidden_embs)


			SAMP_NEIGHS.append(samp_neighs)
			MASK.append(mask)
		# 所以只要 pre_hidden_embs 和 pre_feature
		if not need_previous:
			return pre_hidden_embs, pre_feature, samp_neighs, mask, sage_pre_feature
		else:
			return PRE_HIDDEN_EMBS, PRE_FEATURES, SAMP_NEIGHS, MASK, sage_pre_feature # 多层 PRE_FEATURES 是最外层EMBDING



	def _nodes_map(self, nodes, hidden_embs, neighs):
		layer_nodes, samp_neighs, layer_nodes_dict = neighs
		assert len(samp_neighs) == len(nodes)
		index = [layer_nodes_dict[x] for x in nodes]
		return index

	def _get_unique_neighs_list(self, nodes, num_sample=10):
		'''

		:param nodes: 查询的节点
		:param num_sample:
		:return:
		samp_neighs 节点邻居信息
		unique_nodes 邻居的唯一标识 数量通常大于 nodes
		'''
		_set = set
		to_neighs = [self.adj_lists[int(node)] for node in nodes]  # 这个应该就是 邻居 的 索引
		if not num_sample is None:
			_sample = random.sample
			samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
		else:
			samp_neighs = to_neighs
		samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
		_unique_nodes_list = list(set.union(*samp_neighs)) # 这里对邻居做了一个去重 所以大于目前节点数目
		i = list(range(len(_unique_nodes_list))) # 然后做了一个 排序所以
		unique_nodes = dict(list(zip(_unique_nodes_list, i)))
		return samp_neighs, unique_nodes, _unique_nodes_list

	def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
		'''

		:param nodes:  当前节点及其 邻居信息
		:param pre_hidden_embs:  之前的 emb
		:param pre_neighs:  之前的邻居 [] 就是那个三元组
		:param num_sample:
		:return:
		'''
		unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

		assert len(nodes) == len(samp_neighs) # L-1层邻居节点 和 当前节点 不同的话 报警
		indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
		assert (False not in indicator)
		if not self.gcn:
			samp_neighs = [(samp_neighs[i]-set([nodes[i]])) for i in range(len(samp_neighs))]
		# self.dc.logger.info('2')
		if len(pre_hidden_embs) == len(unique_nodes):
			embed_matrix = pre_hidden_embs
		else:
			embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
		# self.dc.logger.info('3')
		mask = torch.zeros(len(samp_neighs), len(unique_nodes))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1 # 看来需要这个 mask 啊！
		maskUn = copy.deepcopy(mask)
		# self.dc.logger.info('4')

		if self.agg_func == 'MEAN':
			num_neigh = mask.sum(1, keepdim=True) # 目标节点有多少邻居
			mask = mask.div(num_neigh).to(embed_matrix.device) # torch.div()方法将输入的每个元素除以一个常量，然后返回一个新的修改过的张量。
			aggregate_feats = mask.mm(embed_matrix) # x = x.mm(self.w) #x与w相乘

		elif self.agg_func == 'MAX':
			# print(mask)
			indexs = [x.nonzero() for x in mask==1]
			aggregate_feats = []
			# self.dc.logger.info('5')
			for feat in [embed_matrix[x.squeeze()] for x in indexs]:
				if len(feat.size()) == 1:
					aggregate_feats.append(feat.view(1, -1))
				else:
					aggregate_feats.append(torch.max(feat,0)[0].view(1, -1))
			aggregate_feats = torch.cat(aggregate_feats, 0)

		# self.dc.logger.info('6')
		
		return aggregate_feats, embed_matrix, samp_neighs, maskUn

'''
Below is edge decoder
'''
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import math


class EdgeDecoder(Module):
	"""
    Simple Graphsage layer
    """

	def __init__(self, nembed, dropout=0.1):
		super(EdgeDecoder, self).__init__()
		self.dropout = dropout

		self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))

		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.de_weight.size(1))
		self.de_weight.data.uniform_(-stdv, stdv)

	def forward(self, node_embed):
		combine = F.linear(node_embed, self.de_weight)
		adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1, -2)))

		return adj_out

# GraphSMOTE 专用

class SageConv(Module):
	"""
    Simple Graphsage layer
    """

	def __init__(self, in_features, out_features, bias=False):
		super(SageConv, self).__init__()

		self.proj = nn.Linear(in_features * 2, out_features, bias=bias)

		self.reset_parameters()

	# print("note: for dense graph in graphsage, require it normalized.")

	def reset_parameters(self):

		nn.init.normal_(self.proj.weight)

		if self.proj.bias is not None:
			nn.init.constant_(self.proj.bias, 0.)

	def forward(self, features, adj):
		"""
        Args:
            adj: can be sparse or dense matrix.
        """

		# fuse info from neighbors. to be added:
		if not isinstance(adj, torch.sparse.FloatTensor):
			if len(adj.shape) == 3:
				neigh_feature = torch.bmm(adj, features) / (
							adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
			else:
				neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
		else:
			# print("spmm not implemented for batch training. Note!")

			neigh_feature = torch.spmm(adj, features) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)

		# perform conv
		data = torch.cat([features, neigh_feature], dim=-1)
		combined = self.proj(data)

		return combined


class Sage_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Sage_Classifier, self).__init__()

        self.sage1 = SageConv(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.log_softmax(self.mlp(x), 1)

        return x

'''
class Classification(nn.Module):

	def __init__(self, emb_size, num_classes):
		super(Classification, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.layer = nn.Sequential(
								nn.Linear(emb_size, num_classes)	  
								#nn.ReLU()
							)
		self.init_params()

	def init_params(self):
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, embeds):
		logists = torch.log_softmax(self.layer(embeds), 1)
		return logists
'''