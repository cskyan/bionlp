#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2017 by Caspar. All rights reserved.
# File Name: bintree.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-03-09 11:55:38
###########################################################################
#

from binarytree import convert, pprint

from . import func


class Node(object):
	def __init__(self, data={}, left=None, right=None, parent=None):
		self.data = data
		self.left = left
		self.right = right
		self.parent = parent

try:
	from ete3 import Tree
	class ETENode(Tree):
		def __init__(self, data={}, left=None, right=None, parent=None):
			self.data = data
			self.left = left
			self.right = right
			self.parent = parent
			super(ETENode, self).__init__()
		def post_build(self):
			self.name = str(self.data.setdefault('id', ''))
			self.children.extend([x for x in [self.left, self.right] if x is not None])
			self.up = self.parent
except Exception as e:
	print(e)


def from_childlist(node_list, order='bottom_up', node_type='normal'):
	if (order == 'bottom_up'):
		root = preorder_build(-1, None, node_list, node_type=node_type)
	elif (order == 'top_down'):
		root = preorder_build(0, None, node_list, node_type=node_type)
	return root


def preorder_build(node_id, parent_node, node_list, node_type='normal'):
	if (node_type == 'ete'):
		node = ETENode(data=node_list[node_id]['data'], parent=parent_node)
	else:
		node = Node(data=node_list[node_id]['data'], parent=parent_node)
	children = node_list[node_id]['children']
	if (len(children) == 0):
		node.left = None
		node.right = None
		if (node_type == 'ete'):
			node.post_build()
		return node
	if (children[0] != -1):
		node.left = preorder_build(children[0], node, node_list, node_type=node_type)
	else:
		node.left = None
	if (len(children) == 1):
		node.right = None
		return node
	if (children[1] != -1):
		node.right = preorder_build(children[1], node, node_list, node_type=node_type)
	else:
		node.right = None
	if (node_type == 'ete'):
		node.post_build()
	return node


def preorder_getnode(node):
	data_list = [node]
	if (node.left != None):
		data_list.extend(preorder_getnode(node.left))
	if (node.right != None):
		data_list.extend(preorder_getnode(node.right))
	return data_list


def preorder_search(node, cond_func, stop_found=False):
	results = []
	if (cond_func(node)):
		results.append(node)
		if (stop_found):
			return results
	if (node.left != None):
		results.extend(preorder_search(node.left, cond_func, stop_found=stop_found))
	if (node.right != None):
		results.extend(preorder_search(node.right, cond_func, stop_found=stop_found))
	return results


def preorder_modify(node, in_func, out_func, **kwargs):
	in_func(node, **kwargs)
	if (node.left != None):
		preorder_modify(node.left, in_func, out_func, **kwargs)
	if (node.right != None):
		preorder_modify(node.right, in_func, out_func, **kwargs)
	out_func(node, **kwargs)


def bottom_up(leaves, node_pair_gen=None, npg_params={}):
	if (node_pair_gen is None):
		node_pair_gen = _npg
	# Prepare leaf nodes
	levels = [[dict(id=(i,), left=None, right=None, data=leaves[i]) for i in range(len(leaves))]]
	# print levels
	# Merge nodes until the root is generated
	while (len(levels[-1]) > 1):
		merged_nodes = []
		# Generate a new node from a pair of nodes from the current top level
		for node_pair, data in node_pair_gen(levels[-1], **npg_params):
			# print node_pair
			if (len(node_pair) == 1):
				# Odd number of nodes
				merged_nodes.append(dict(id=levels[-1][node_pair[0]]['id'], left=node_pair[0], right=None, data=data))
			else:
				# Even number of nodes
				merged_nodes.append(dict(id=levels[-1][node_pair[0]]['id']+levels[-1][node_pair[1]]['id'], left=node_pair[0], right=node_pair[1], data=data))
		# print merged_nodes
		levels.append(merged_nodes)
	# print levels
	ordered_levels = [levels[-1]]
	# Reorder the nodes
	for lv in levels[-2::-1]:
		node_list = []
		for node in ordered_levels[-1]:
			node_list.extend([lv[node[x]] for x in ['left', 'right'] if node[x] != None])
		ordered_levels.append(node_list)
	# print ordered_levels
	# Construct the binary tree node list
	root = convert(func.flatten_list(ordered_levels))
	# pprint(root)
	return root

def _npg(leaf_sets, **kwargs):
	for i in range(int(len(leaf_sets) / 2)):
		yield (2 * i, 2 * i + 1), {}
	if (len(leaf_sets) % 2 != 0):
		yield (len(leaf_sets) - 1,), {}
