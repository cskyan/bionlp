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


def run(target, **kwargs):
	p = Process(target=target, kwargs=kwargs)
	p.start()
	p.join()


def run_pool(target, n_jobs=1, dist_param=[], **kwargs):
	res_list, fix_kwargs, iter_kwargs = [], {}, {}
	def collect_res(res):
		res_list.append(res)
	pool = Pool(processes=n_jobs)
	# Separate iterable arguments and singular arguments
	for k, v in kwargs.iteritems():
		if (k in dist_param and hasattr(v, '__iter__')):
			iter_kwargs[k] = v
		else:
			fix_kwargs[k] = v
	# Construct arguments for each process
	for argv in zip(*iter_kwargs.values()):
		args = dict(zip(iter_kwargs.keys(), argv))
		args.update(fix_kwargs)
		pool.apply_async(target, kwds=args, callback=collect_res)
	pool.close()
	pool.join()
	return res_list
	
	
def run_ipp(target, n_jobs=1, dist_param=[], **kwargs):
	import ipyparallel as ipp
	from subprocess import Popen
	pstart = Popen(['ipcluster', 'start', '-n', '%i' % n_jobs])
	time.sleep(5)
	try:
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
			results = [r.get() for r in res_list]
		else:
			print 'No results return!'
			results = []
		c.shutdown(hub=True)
		pstart.terminate()
		pend = Popen(['ipcluster', 'stop'])
		time.sleep(5)
		pend.terminate()
		return results
	except Exception as e:
		print e
		pstart.terminate()
		pend = Popen(['ipcluster', 'stop'])
		time.sleep(5)
		pend.terminate()
		return []