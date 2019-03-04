#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2017 by Caspar. All rights reserved.
# File Name: njobs.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-02-08 17:32:42
###########################################################################
#

import time
from multiprocessing import Process, Pool

import numpy as np


def split_1d(task_num, split_num=None, task_size=None, split_size=None, ret_idx=False):
	if (split_num is None):
		if (task_size is None or split_size is None):
			return range(task_num+1) if ret_idx else [1] * task_num
		group_size = max(1, split_size / task_size)
		split_num = task_num / group_size
		remainder = task_num % group_size
		results = [group_size] * split_num if (remainder == 0) else [group_size] * split_num + [remainder]
	else:
		group_size = task_num / split_num
		remainder = task_num % split_num
		results = [group_size] * split_num if (remainder == 0) else [group_size + 1] * remainder + [group_size] * (split_num - remainder)
	return np.cumsum([0]+results).tolist() if ret_idx else results
		
		
def split_2d(task_grid, split_num=None, task_size=None, split_size=None, ret_idx=False):
	if (split_num is None):
		if (task_size is None or split_size is None):
			return [range(task_grid[0]+1), range(task_grid[1])+1] if ret_idx else [[1] * task_grid[0], [1] * task_grid[1]]
		group_size = max(1, split_size / task_size)
		factor = (1.0 * group_size / np.product(task_grid))**0.5
		_grid = np.array(task_grid) * factor
		grid = _grid.round().astype('int')
		if np.product(grid) < np.product(_grid):
			grid[(_grid % 1).argmax()] += 1
		results = [split_1d(task_grid[0], task_size=1, split_size=grid[0]), split_1d(task_grid[1], task_size=1, split_size=grid[1])]
	else:
		factor = (1.0 * split_num / np.product(task_grid))**0.5
		_grid = np.array(task_grid) * factor
		grid = _grid.round().astype('int')
		if np.product(grid) < np.product(_grid):
			grid[(_grid % 1).argmax()] += 1
		results = [split_1d(task_grid[0], split_num=grid[0]), split_1d(task_grid[1], split_num=grid[1])]
	return [np.cumsum([0]+results[0]).tolist(), np.cumsum([0]+results[1]).tolist()] if ret_idx else results
		

def run(target, **kwargs):
	p = Process(target=target, kwargs=kwargs)
	p.start()
	p.join()


def run_pool(target, n_jobs=1, pool=None, ret_pool=False, dist_param=[], **kwargs):
	if (n_jobs < 1): return None
	res_list, fix_kwargs, iter_kwargs = [], {}, {}
	if (pool is None):
		pool = Pool(processes=n_jobs)
	# Separate iterable arguments and singular arguments
	for k, v in kwargs.iteritems():
		if (k in dist_param and hasattr(v, '__iter__')):
			iter_kwargs[k] = v
		else:
			fix_kwargs[k] = v
	if (len(iter_kwargs) == 0):
		res_list = [pool.apply_async(target, kwds=kwargs) for x in range(n_jobs)]
	else:
		# Construct arguments for each process
		for argv in zip(*iter_kwargs.values()):
			args = dict(zip(iter_kwargs.keys(), argv))
			args.update(fix_kwargs)
			r = pool.apply_async(target, kwds=args)
			res_list.append(r)
	res_list = [r.get() for r in res_list] if len(res_list) > 1 else res_list[0].get()
	time.sleep(0.01)
	if (ret_pool):
		return res_list, pool
	pool.close()
	pool.join()
	return res_list
	
	
def run_ipp(target, n_jobs=1, client=None, ret_client=False, dist_param=[], **kwargs):
	from subprocess import Popen
	import ipyparallel as ipp
	use_client = False
	if (client):
		if (type(client) is str and not client.isspace()):
			try:
				c = ipp.Client(profile=client, timeout=5)
				use_client = True
			except Exception as e:
				print 'Failed to connect to the ipcluster with profile_%s' % client
		elif (type(client) is ipp.Client):
			c = client
			use_client = True
	try:
		if (not use_client):
			pstart = Popen(['ipcluster', 'start', '-n', '%i' % n_jobs])
			time.sleep(8)
			c = ipp.Client(timeout=5)
		res_list, fix_kwargs, iter_kwargs = [], {}, {}
		# Separate iterable arguments and singular arguments
		for k, v in kwargs.iteritems():
			if (k in dist_param and hasattr(v, '__iter__')):
				iter_kwargs[k] = v
			else:
				fix_kwargs[k] = v
		# Construct arguments for each process
		for i, argv in enumerate(zip(*iter_kwargs.values())):
			args = dict(zip(iter_kwargs.keys(), argv))
			args.update(fix_kwargs)
			r = c[c.ids[i % len(c.ids)]].apply_async(target, **args)
			res_list.append(r)
		if c.wait(res_list):
			time.sleep(0.01)
			results = [r.get() for r in res_list]
		else:
			print 'No results return!'
			results = []
		if (not use_client):
			c.shutdown(hub=True)
			pstart.terminate()
			pend = Popen(['ipcluster', 'stop'])
			time.sleep(5)
			pend.terminate()
			return results
		elif (ret_client):
			return results, c
		else:
			c.close()
			return results
	except Exception as e:
		print e
		if (not use_client):
			pstart.terminate()
			pend = Popen(['ipcluster', 'stop'])
			time.sleep(5)
			pend.terminate()
		return []