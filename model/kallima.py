#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2017 by Caspar. All rights reserved.
# File Name: kallima.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-03-01 19:56:33
###########################################################################

import os
from itertools import product, combinations
from collections import Counter

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, coo_matrix, find
from scipy.sparse import linalg as spla
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.cluster import AgglomerativeClustering

from .. import dstclc
from ..util import io, func, math, bintree
from ..util.oo import iprofile


class Kallima(BaseEstimator, ClusterMixin, TransformerMixin):
	''' Kallima Algorithm, a multi-label clustering method
	Parameters
	----------
	metric : string, or callable
		The metric to use when calculating distance between instances in a feature array
	method : {'mstcut', 'cutree'}, optional
		The graph clustering method, Minimum Spanning Tree Cut or Cut Binary Tree
	cut_method : {'normcut', 'ratiocut', 'mincut'}, optional
		The graph cutting method, Normalized Cut, Ratio Cut or Minimum Cut
	cut_step : float, optional (between 0 and 0.5, default 0.1)
		The interval of edge cutting threshold in mstcut method
	cns_ratio : float, optional (between 0 and 1, default 0.5)
		The coefficient of the constraint distance when embedding it into the instance distance
	nn_method : {'rnn', 'knn'}, optional
		The nearest neighbor graph method, Radius Nearest Neighbors or K Nearest Neighbors
	nn_param : float, or int
		The parameter for the nearest neighbor graph method, radius for RNN or n_neighbors for KNN
	max_cltnum : int, optional
		The maximum number of the expected clusters
	coarse : float, optional (between 0 and 1, default 0.4)
		The percentage threshold of distance difference when removing the inconsistent edges in mstcut method
	rcexp : int, optional
		The exponential coefficient e when calculating the rirc = ri * rc^e.	
	cond : float, optional (default 0.3)
		The conductance threshold used to filter the merged clusters in mstcut method
	cross_merge : bool, default=False
		Whether or not to do cross-merge step in mstcut method
	merge_all : bool, default=False
		Whether or not to merge all the nodes besides leaves in cross-merge step in mstcut method
	save_g : bool, default=False
		Whether or not to save the graph model
	n_jobs : int, optional (default = 1)
		The number of parallel jobs to run for neighbors search. If -1, then the number of jobs is set to the number of CPU cores
	'''
	def __init__(self, metric='euclidean', method='mstcut', cut_method='normcut', cut_step=0.1, cns_ratio=0.5, nn_method='rnn', nn_param=0.5, max_cltnum=100, coarse=0.4, rcexp=1, cond=0.3, cross_merge=False, merge_all=False, save_g=False, n_jobs=1):
		self.metric = metric
		self.method = method
		self.cut_method = cut_method
		self.cut_step = cut_step
		self.cns_ratio = cns_ratio
		self.nn_method = nn_method
		self.nn_param = nn_param
		self.max_cltnum = max_cltnum
		self.coarse = coarse
		self.rcexp = rcexp
		self.cond = cond
		self.cross_merge = cross_merge
		self.merge_all = merge_all
		self.save_g = save_g
		self.n_jobs = n_jobs

	def fit(self, X, y=None, constraint=None):
		'''Compute Constraint Kallima clustering.
		Parameters
		----------
		X : array-like or sparse matrix, shape = [n_samples, n_features]
			Training instances to cluster.
		'''
		X = check_array(X, accept_sparse="csr", order='C', dtype=[np.float64, np.float32, np.float16, np.int64, np.int32, np.int16, np.int8])
		self.mdl_name = '%s%s%s' % ('' if constraint is None else 'cns', self.nn_method, self.nn_param)
		## Calculate the distance
		if (os.path.exists('../distance_matrix.npz')):
			X = io.read_npz('../distance_matrix.npz')['data']
			self.metric = 'precomputed'
		if (self.metric == 'precomputed'):
			D = X
		else:
			D = dstclc.cns_dist(X, C=constraint, metric=self.metric, a=self.cns_ratio, n_jobs=self.n_jobs)
			io.write_npz(D, fpath='distance_matrix', compress=True)
		## Build the nearest neighbor graph
		if (self.nn_method == 'rnn'):
			self.NNG_ = NNG = radius_neighbors_graph(D, self.nn_param, mode='distance', metric='precomputed', n_jobs=self.n_jobs)
		else:
			self.NNG_ = NNG = kneighbors_graph(D, self.nn_param, mode='distance', metric='precomputed', n_jobs=self.n_jobs)
		## Cluster based on the nearest neighbor graph
		if (self.method == 'cutree'):
			clusters = self._cut_tree(NNG)
		else:
			clusters = self._mst_cut(NNG)
		self.clusters_ = clusters
		self.labels_ = labels = np.zeros((X.shape[0], len(clusters)), dtype=np.int64)
		for i, clt in enumerate(clusters.keys()):
			labels[clt, i] = 1
		## Save the NNG
		if (self.save_g):
			coo_NNG = NNG.tocoo()
			edges = list(zip(coo_NNG.row, coo_NNG.col))
			G = nx.Graph()
			G.add_weighted_edges_from([(i, j, coo_NNG.data[k]) for k, (i, j) in enumerate(edges)])
			try:
				nx.write_gml(G, '%s.gml' % self.mdl_name)
			except Exception as e:
				print 'Cannot save the NNG graph models %s!' % self.mdl_name
				print e
		return self

	def predict(self, X, constraint=None):
		'''Predict the closest cluster each sample in X belongs to.
		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			New data to predict.
		Returns
		-------
		u : array, shape [n_samples, n_clusters]
			Predicted multiple cluster labels.
		'''
		check_is_fitted(self, 'clusters_')
		X = self._check_test_data(X)
		
		return self.labels_
		
	def fit_predict(self, X, y=None, constraint=None):
		'''Compute cluster centers and predict cluster index for each sample.
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			New data to predict.
		Returns
		-------
		u : array, shape [n_samples, n_clusters]
			Predicted multiple cluster labels.
		'''
		self.fit(X, y=y, constraint=constraint)
		return self.labels_

	def transform(self, X, y=None, constraint=None):
		'''Transform X to a cluster-distance space.
		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			New data to transform.
		Returns
		-------
		X_new : array, shape [n_samples, k]
			X transformed in the new space.
		'''
		check_is_fitted(self, 'clusters_')
		X = self._check_test_data(X)
		return self.labels_

	def _transform(self, X, y=None, constraint=None):
		'''guts of transform method; no input validation'''
		return self.labels_
		
	def _check_test_data(self, X):
		X = check_array(X, accept_sparse="csr", order='C', dtype=[np.float64, np.float32, np.float16, np.int64, np.int32, np.int16, np.int8])
		n_samples, n_features = X.shape
		expected_n_features = self.NNG_.shape[1]
		if not n_features == expected_n_features:
			raise ValueError("Incorrect number of features. Got %d features, expected %d" % (n_features, expected_n_features))
		return X
		
	def _bicut_val(self, g, a, b):
		return sum([g[i, j] for i, j in product(a, b)])
		
	def _cut_val(self, g, clusters, method='mincut'):
		vset = set(range(g.shape[0]))
		if (all(isinstance(clt, set) for clt in clusters)):
			clt_sets = clusters
		else:
			clt_sets = [set(clt) for clt in clusters]
		cmp_clt_sets = [vset - clt for clt in clt_sets]
		# print clt_sets
		# print cmp_clt_sets
		if (method == 'normcut'):
			return 0.5 * sum([self._bicut_val(g, clt, cmpclt) / (g[list(clt),:].sum() - g.diagonal()[list(clt)].sum()) for clt, cmpclt in zip(clt_sets, cmp_clt_sets)])
		elif (method == 'ratiocut'):
			return 0.5 * sum([self._bicut_val(g, clt, cmpclt) / len(clt) for clt, cmpclt in zip(clt_sets, cmp_clt_sets)])
		else:
			return 0.5 * sum([self._bicut_val(g, clt, cmpclt) for clt, cmpclt in zip(clt_sets, cmp_clt_sets)])
			
	def _rirc(self, g, clusters):
		clt_num = len(clusters)
		clt_arrays = [np.array(list(clt)) for clt in clusters]
		clt_sets = [set(clt) for clt in clusters]
		clt_size = [len(clt) for clt in clt_sets]
		# Calculate edge cut
		edge_cut = [self._cut_val(g, [clt], method='mincut') for clt in clt_sets]
		mutual_edge_cut = {}
		ri, rc = np.zeros((clt_num, clt_num), dtype=np.float16), np.zeros((clt_num, clt_num), dtype=np.float16)
		for i, j in combinations(range(clt_num), 2):
			# print clt_sets[i], clt_sets[j]
			union_clt = np.array(list(clt_sets[i]) + list(clt_sets[j]))
			union_clt_size = clt_size[i] + clt_size[j]
			sub_graph = g[union_clt,:][:,union_clt]
			mutual_edge_cut[(i,j)] = self._cut_val(sub_graph, [range(clt_size[i])], method='mincut')
			# Relative inter-connectivity
			if (edge_cut[i] == 0 or edge_cut[j] == 0):
				ri[i, j] = ri[j, i] = rc[i, j] = rc[j, i] = 0
				continue
			ri[i, j] = ri[j, i] = 2.0 * abs(mutual_edge_cut[(i,j)]) / (abs(edge_cut[i]) + abs(edge_cut[j]))
			# Relative closeness
			mutual_edge_num = find(sub_graph[range(clt_size[i]),:][:,range(clt_size[i],union_clt_size)])[0].shape[0]
			bicut_edge_num = find(g[clt_arrays[i],:])[0].shape[0], find(g[clt_arrays[j],:])[0].shape[0]
			if (mutual_edge_num == 0 or bicut_edge_num[0] == 0 or bicut_edge_num[1] == 0):
				rc[i, j] = rc[j, i] = 0
			else:
				rc[i, j] = rc[j, i] = (mutual_edge_cut[(i,j)] / mutual_edge_num) / (1.0 * clt_size[i] / union_clt_size * edge_cut[i] / bicut_edge_num[0] + 1.0 * clt_size[j] / union_clt_size * edge_cut[j] / bicut_edge_num[1])
		return ri * rc ** self.rcexp, edge_cut, mutual_edge_cut
		
	def _node_pair_gen(self, leaf_sets, g=[], leaf_clt=[]):
		leaf_ids = [x['id'] for x in leaf_sets]
		clt_sets = [func.flatten_list([leaf_clt[id] for id in ids]) for ids in leaf_ids]
		clt_arrays = [np.array(x) for x in clt_sets]
		vset = set(range(g.shape[0]))
		cmp_clt_arrays = [np.array(list(vset - set(x))) for x in clt_sets]
		# Calculate the relative inter-connectivity (RI) and relative closeness (RC)
		rirc, ec, mec = self._rirc(g, clt_sets)
		ec = [self._cut_val(g, [clt], method='mincut') for clt in clt_sets]
		# Calculate the conductance
		for i in xrange(len(leaf_sets)):
			leaf_sets[i]['data']['cond'] = 2.0 * ec[i] / min(g[clt_arrays[i],:][:,clt_arrays[i]].sum(), g[cmp_clt_arrays[i],:][:,cmp_clt_arrays[i]].sum())
		# Generate pairwise cluster sets
		generated = set([])
		while (rirc.max() > 0 and len(generated) < len(clt_sets)):
			max_idx = rirc.argmax()
			max_pos = (max_idx / rirc.shape[1], max_idx % rirc.shape[1])
			if (max_pos[0] == max_pos[1]):
				# Odd number of clusters
				yield (max_pos[0],), dict(cond=0)
				break
			yield max_pos, dict(cond=0)
			generated |= set(max_pos)
			rirc[max_pos,:] = 0
			rirc[:,max_pos] = 0
		if (len(generated) < len(clt_sets)):
			remain_cltsets = list(set(range(len(clt_sets))) - generated)
			for cltidx_pair in bintree._npg(remain_cltsets):
				if (len(cltidx_pair[0]) > 1):
					yield (remain_cltsets[cltidx_pair[0][0]], remain_cltsets[cltidx_pair[0][1]]), dict(cond=0)
				else:
					yield (remain_cltsets[cltidx_pair[0][0]],), dict(cond=0)
			
	def _tree_clt(self, node):
		if (node == None):
			return []
		clusters = []
		clusters.extend(self._tree_clt(node.left))
		clusters.extend(self._tree_clt(node.right))
		clusters.extend([node.value])
		return clusters
		
	def _conductance(self, g, vset, clt):
		clt_array, cmp_clt_array = np.array(clt), np.array(list(vset - set(clt)))
		edge_cut = self._cut_val(g, [clt], method='mincut')
		return 2.0 * edge_cut / min(g[clt_array,:][:,clt_array].sum(), g[cmp_clt_array,:][:,cmp_clt_array].sum())
		
	def _cross_merge(self, node, g, vset):
		merged_nodes = []
		if (node.left != None):
			merged_nodes.extend(self._cross_merge(node.left, g, vset))
		if (node.right != None):
			merged_nodes.extend(self._cross_merge(node.right, g, vset))
		if (not self.merge_all and (node.left != None or node.right != None)):
			return merged_nodes
		parent = node.parent
		if (parent == None or parent.parent == None or parent.parent.parent == None):
			return merged_nodes
		id_order = 'bottom_up' if node.data['pid'] < node.data['id'] else 'top_down'
		while (parent.parent.parent != None):
			uncles = list(set([parent.parent.left, parent.parent.right]) - set([parent]))
			for cand_node in bintree.preorder_getnode(uncles[0]):
				if (not self.merge_all and (cand_node.left != None or cand_node.right != None)):
					continue
				merged_clt = node.data['clt'] + cand_node.data['clt']
				cond = self._conductance(g, vset, merged_clt)
				rirc = 0.5 * self._rirc(g, [node.data['clt'], cand_node.data['clt']])[0].sum()
				insert_pnode = cand_node.parent if ((id_order == 'bottom_up' and cand_node.data['id'] > node.data['id']) or (id_order == 'top_down' and cand_node.data['id'] < node.data['id'])) else node.parent
				if (cond > insert_pnode.data['cond'] and rirc > 0.1):
					# print cond, rirc, merged_clt
					pid = insert_pnode.data['id']
					merged_nodes.append((merged_clt, dict(pid=pid, clt=merged_clt, cond=cond)))
			if (not self.merge_all): break
			parent = parent.parent
		return merged_nodes

	@iprofile
	def _mst_cut(self, NNG):
		## Build the minimum spanning tree
		self.MST_ = MST = minimum_spanning_tree(NNG)
		coo_MST = MST.tocoo()
		## Sort the edge weight/distance to perform hierarchical edge cut
		sorted_idx = coo_MST.data.argsort()
		sorted_row, sorted_col, sorted_data = coo_MST.row[sorted_idx], coo_MST.col[sorted_idx], coo_MST.data[sorted_idx]
		# from ..util import plot
		# plot.plot_hist(sorted_data, 'Edge Weight (Distance)', 'Number', fit_line=True, title='Histogram of Edge Weight (Distance)', fname='ew_hist')
		# plot.plot_hist(sorted_data, 'Edge Weight (Distance)', 'Number', cumulative=True, fit_line=True, title='Histogram of Edge Weight (Distance)', fname='cew_hist')
		## Find the cut threshold according to the distribution of the weight/distance, linspace: uniform distribution
		cut_idx, g, presrv_e, hrc_clusters = sorted_data.shape[0], coo_MST, dict(row=np.array([], dtype=np.int64), col=np.array([], dtype=np.int64), data=np.array([], dtype=np.float64)), {}
		hist, bin_edges = np.histogram(sorted_data, bins='rice')
		weird_val_idx = len(hist) - 1 - np.abs(hist[-1:0:-1] - hist[-2::-1]).argmax()
		cut_val = sorted_data[np.searchsorted(sorted_data, (bin_edges[weird_val_idx] + bin_edges[weird_val_idx + 1]) / 2)]
		for cut_thrshd in np.linspace(cut_val, bin_edges[weird_val_idx], num=(cut_val-bin_edges[weird_val_idx])/self.cut_step, endpoint=False):
			# print cut_thrshd
			# Find out the edges that connect to the leaves
			out_degree, in_degree = Counter(np.append(sorted_row[:cut_idx], presrv_e['row'])), Counter(np.append(sorted_col[:cut_idx], presrv_e['col']))
			degree_sum = out_degree + in_degree
			leaves = [x for x in degree_sum.keys() if degree_sum[x] == 1]
			# Search the cut index
			new_cut_idx = np.searchsorted(sorted_data[:cut_idx], cut_thrshd)
			# Update the preserved edges
			preserved_mask = np.array([True if u in leaves or v in leaves else False for u, v in zip(sorted_row[new_cut_idx:cut_idx], sorted_col[new_cut_idx:cut_idx])])
			if (preserved_mask.shape[0] > 0):
				presrv_e = dict(row=np.append(presrv_e['row'], sorted_row[new_cut_idx:cut_idx][preserved_mask]), col=np.append(presrv_e['col'], sorted_col[new_cut_idx:cut_idx][preserved_mask]), data=np.append(presrv_e['data'], sorted_data[new_cut_idx:cut_idx][preserved_mask]))
			cut_idx = new_cut_idx
			# Build a new graph
			g = coo_matrix((np.append(sorted_data[:cut_idx], presrv_e['data']), (np.append(sorted_row[:cut_idx], presrv_e['row']), np.append(sorted_col[:cut_idx], presrv_e['col']))), shape=g.shape)
			# Find out the clusters
			clt_num, clt_lbs = connected_components(g)
			clts = [tuple(np.where(clt_lbs==x)[0]) for x in np.unique(clt_lbs)]
			# Update the clusters
			hrc_clusters.update(dict([(clt, cut_thrshd) for clt in clts if len(clt) > 1 and (clt, cut_thrshd) not in hrc_clusters]))
			if (len(hrc_clusters) >= 2 * self.max_cltnum):
				break
		# minor_clusters = hrc_clusters
		## Find the inconsistent edge weight/distance
		dok_MST, csr_MST, csc_MST = g.todok(), g.tocsr(), g.tocsc()
		rowcol_ew = [np.append(dok_MST.getrow(i).toarray().reshape((dok_MST.shape[0],)), dok_MST.getcol(i).toarray().reshape((dok_MST.shape[0],))) for i in range(dok_MST.shape[0])]
		rowcol_ewp = [x[x>0] for x in rowcol_ew]
		minimum_ew = [x.min() if x.shape[0] > 0 else 0 for x in rowcol_ewp]
		# print minimum_ew
		for i in xrange(dok_MST.shape[0]):
			for j in xrange(csr_MST.indptr[i], csr_MST.indptr[i+1]):
				col = csr_MST.indices[j]
				if ((dok_MST[i,col] - minimum_ew[i]) / minimum_ew[i] > self.coarse and minimum_ew[col] < dok_MST[i,col]):
					dok_MST[i, col] = 0
			for j in xrange(csc_MST.indptr[i], csc_MST.indptr[i+1]):
				row = csc_MST.indices[j]
				if ((dok_MST[row,i] - minimum_ew[i]) / minimum_ew[i] > self.coarse and minimum_ew[row] < dok_MST[row,i]):
					dok_MST[row,i] = 0
		# Find out the clusters
		clt_num, clt_lbs = connected_components(dok_MST)
		clts = [tuple(np.where(clt_lbs==x)[0]) for x in np.unique(clt_lbs)]
		minor_clusters = dict([(clt, 0) for clt in clts if len(clt) > 1])
		minor_clts = minor_clusters.keys()
		# print minor_clts
		## Convert the distance matrix into similarity matrix
		SIM_NNG = NNG.copy()
		SIM_NNG.data = 1 - SIM_NNG.data
		## Calculate the similarity and distance matrix of minor clusters
		sim_rirc, ec, mec = self._rirc(SIM_NNG, minor_clts)
		sim_rirc = dstclc.normdist(sim_rirc)
		dist_rirc = 1 - sim_rirc
		## Merge the clusters by hierarchical clustering
		conn = (sim_rirc > 0.5).view(dtype='int8')
		aggl_clt = AgglomerativeClustering(connectivity=conn, affinity='precomputed', memory='/dev/shm', linkage='complete')
		aggl_clt.fit(dist_rirc)
		vset, cand_clts, conds = set(range(SIM_NNG.shape[0])), minor_clts[:], [10]*len(minor_clts)
		for l, r in aggl_clt.children_:
			merged_clt = cand_clts[l] + cand_clts[r]
			clt_array, cmp_clt_array = np.array(merged_clt), np.array(list(vset - set(merged_clt)))
			edge_cut = self._cut_val(SIM_NNG, [merged_clt], method='mincut')
			# Calculate the conductance of each merged cluster to help determine whether it should be kept
			conductance = self._conductance(SIM_NNG, vset, merged_clt)
			if (conductance > self.cond):
				cand_clts.append(merged_clt)
				conds.append(conductance)
		## Cross merge clusters over the hierarchical tree
		if (self.cross_merge):
			children_list = [[]] * len(minor_clts) + aggl_clt.children_.tolist()
			parent_list = [-1 for x in range(len(children_list))]
			for i, children in enumerate(children_list):
				for child in children:
					parent_list[child] = i
			# Construct binary tree
			node_list = [dict(children=children, data=dict(id=i, pid=pid, clt=clt, cond=cond)) for i, pid, clt, children, cond in zip(range(len(cand_clts)), parent_list, cand_clts, children_list, conds)]
			clt_tree = bintree.from_childlist(node_list, order='bottom_up')
			merged_nodes = self._cross_merge(clt_tree, SIM_NNG, vset)
			mn_dict = dict([(tuple(set(k)), v) for k, v in merged_nodes])
			if (len(mn_dict) > 0):
				merged_clts, merged_conds = zip(*[(mn['clt'], mn['cond']) for mn in mn_dict.values()])
				cand_clts.extend(merged_clts)
				conds.extend(merged_conds)
		## Final refined clusters
		clusters = dict([(cltid, cond) for cltid, cond in zip(cand_clts, conds) if cond > self.cond])
		# clt_leaves = [{} for x in range(len(minor_clts))]
		# clt_tree = bintree.bottom_up(clt_leaves, node_pair_gen=self._node_pair_gen, npg_params=dict(g=SIM_NNG, leaf_clt=minor_clts))
		# cluster_idx = self._tree_clt(clt_tree)
		# clusters = dict([(tuple(sorted(set(func.flatten_list([minor_clts[i] for i in cltidx['id']])))), cltidx['data']) for cltidx in cluster_idx])
		# print minor_clts
		# print clusters.keys()
		## Save the minimum spanning tree
		if (self.save_g):
			mst_edges = list(zip(coo_MST.row, coo_MST.col))
			G = nx.Graph()
			G.add_weighted_edges_from([(i, j, coo_MST.data[k]) for k, (i, j) in enumerate(mst_edges)])
			try:
				nx.write_gml(G, '%s_mst.gml' % self.mdl_name)
			except Exception as e:
				print 'Cannot save the MST graph model %s!' % self.mdl_name
				print e
		return clusters
		
	def _cut_tree(self, NNG):
		pass