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

from multiprocessing import Process, Pool


def run(target, **kwargs):
	p = Process(target=target, kwargs=kwargs)
	p.start()
	p.join()


def run_pool(target, **kwargs):
	res_list = []
	def collect_res(res):
		res_list.append(res)
	pool = Pool()
	fix_kwargs, iter_kwargs = {}, {}
	# Separate iterable arguments and singular arguments
	for k, v in kwargs.iteritems():
		if (hasattr(v, '__iter__')):
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