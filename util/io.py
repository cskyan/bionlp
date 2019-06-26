#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: io.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-04-14 10:56:04
###########################################################################
#

import os, sys, yaml, json, time, errno, pickle

import numpy as np
import pandas as pd
from scipy import sparse

from . import fs, io


def inst_print(text):
	print(text)
	sys.stdout.flush()

def write_json(data, fpath='data.json', code='ascii'):
	if (type(data) is not dict): data = dict(data=data)
	fs.write_file(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')), fpath=fpath, code=code)


def read_json(fpath):
	return parse_json('\n'.join(fs.read_file(fpath)))


def parse_json(json_str):
	fp = io.StringIO(json_str)
	return json.load(fp)


def write_obj(obj, fpath='obj'):
	fs.mkdir(os.path.dirname(fpath))
	with open(os.path.splitext(fpath)[0] + '.pkl', 'wb') as f:
		pickle.dump(obj, f)


def read_obj(fpath):
	fpath = os.path.splitext(fpath)[0] + '.pkl'
	if (os.path.exists(fpath)):
		with open(fpath, 'rb') as f:
			return pickle.load(f)
	else:
		print(('File %s does not exist!' % fpath))


def write_npz(ndarray, fpath='ndarray', compress=False):
#	fs.mkdir(os.path.dirname(fpath))
	if (compress):
		save_f = np.savez_compressed
	else:
		save_f = np.savez
	if (type(ndarray) == dict):
		save_f(os.path.splitext(fpath)[0] + '.npz', **ndarray)
	elif (type(ndarray) == np.ndarray):
		save_f(os.path.splitext(fpath)[0] + '.npz', data=ndarray)


def read_npz(fpath):
	fpath = os.path.splitext(fpath)[0] + '.npz'
	if (os.path.exists(fpath)):
		return np.load(fpath)
	else:
		print(('File %s does not exist!' % fpath))


def write_spmt(mt, fpath, sparse_fmt='csr', compress=False):
	fs.mkdir(os.path.dirname(fpath))
	fpath = os.path.splitext(fpath)[0] + '.npz'
	if (compress):
		save_f = np.savez_compressed
	else:
		save_f = np.savez
	if (not sparse.isspmatrix(mt)):
		mt = sparse.csc_matrix(mt) if sparse_fmt == 'csc' else sparse.csr_matrix(mt)
	if (sparse_fmt == 'csc'):
		save_f(fpath, data=mt.data, indices=mt.indices, indptr=mt.indptr, shape=mt.shape)
	elif (sparse_fmt == 'csr'):
		save_f(fpath, data=mt.data, indices=mt.indices, indptr=mt.indptr, shape=mt.shape)
	else:
		write_spmt(mt.tocsr(), fpath, sparse_fmt='csr', compress=compress)


def read_spmt(fpath, sparse_fmt='csr'):
	npzfile = read_npz(fpath)
	if (sparse_fmt == 'csc'):
		mt = sparse.csc_matrix((npzfile['data'], npzfile['indices'], npzfile['indptr']), shape=npzfile['shape'])
	else:
		mt = sparse.csr_matrix((npzfile['data'], npzfile['indices'], npzfile['indptr']), shape=npzfile['shape'])
	return mt


def write_df(df, fpath, with_col=True, with_idx=False, sparse_fmt=None, compress=False):
	fs.mkdir(os.path.dirname(fpath))
	fpath = os.path.splitext(fpath)[0] + '.npz'
	if (compress):
		save_f = np.savez_compressed
	else:
		save_f = np.savez
	if (sparse_fmt == None or (type(sparse_fmt) == str and sparse_fmt.lower() == 'none')):
		save_f(fpath, data=df.values, shape=df.shape, col=df.columns.values if with_col else None, idx=df.index.values if with_idx else None)
	elif (sparse_fmt == 'csc'):
		sp_mt = sparse.csc_matrix(df.values)
		save_f(fpath, data=sp_mt.data, indices=sp_mt.indices, indptr=sp_mt.indptr, shape=sp_mt.shape, col=df.columns.values if with_col else None, idx=df.index.values if with_idx else None)
	elif (sparse_fmt == 'csr'):
		sp_mt = sparse.csr_matrix(df.values)
		save_f(fpath, data=sp_mt.data, indices=sp_mt.indices, indptr=sp_mt.indptr, shape=sp_mt.shape, col=df.columns.values if with_col else None, idx=df.index.values if with_idx else None)


def read_df(fpath, with_col=True, with_idx=False, sparse_fmt=None):
	npzfile = read_npz(fpath)
	if (sparse_fmt == None or (type(sparse_fmt) == str and sparse_fmt.lower() == 'none')):
		mt = npzfile['data']
	elif (sparse_fmt == 'csc'):
		mt = sparse.csc_matrix((npzfile['data'], npzfile['indices'], npzfile['indptr']), shape=npzfile['shape']).todense()
	elif (sparse_fmt == 'csr'):
		mt = sparse.csr_matrix((npzfile['data'], npzfile['indices'], npzfile['indptr']), shape=npzfile['shape']).todense()
	return pd.DataFrame(data=mt, index=npzfile['idx'] if with_idx and len(npzfile['idx'].shape) == 1 else None, columns=npzfile['col'] if len(npzfile['col'].shape) == 1 else None)


def write_spdf(df, fpath, with_col=True, with_idx=False, sparse_fmt=None, compress=False):
	fs.mkdir(os.path.dirname(fpath))
	fpath = os.path.splitext(fpath)[0] + '.npz'
	if (compress):
		save_f = np.savez_compressed
	else:
		save_f = np.savez
	if (sparse_fmt == None or (type(sparse_fmt) == str and sparse_fmt.lower() == 'none')):
		save_f(fpath, data=df['values'], shape=df['shape'], col=df['columns'] if with_col else None, idx=df['index'] if with_idx else None)
	elif (sparse_fmt == 'csc'):
		sp_mt = sparse.csc_matrix(df['values'])
		save_f(fpath, data=sp_mt.data, indices=sp_mt.indices, indptr=sp_mt.indptr, shape=sp_mt.shape, col=df['columns'] if with_col else None, idx=df['index'] if with_idx else None)
	elif (sparse_fmt == 'csr'):
		sp_mt = sparse.csr_matrix(df['values'])
		save_f(fpath, data=sp_mt.data, indices=sp_mt.indices, indptr=sp_mt.indptr, shape=sp_mt.shape, col=df['columns'] if with_col else None, idx=df['index'] if with_idx else None)


def read_spdf(fpath, with_col=True, with_idx=False, sparse_fmt=None):
	npzfile = read_npz(fpath)
	if (sparse_fmt == None or (type(sparse_fmt) == str and sparse_fmt.lower() == 'none')):
		mt = npzfile['data']
	elif (sparse_fmt == 'csc'):
		mt = sparse.csc_matrix((npzfile['data'], npzfile['indices'], npzfile['indptr']), shape=npzfile['shape'])
	elif (sparse_fmt == 'csr'):
		mt = sparse.csr_matrix((npzfile['data'], npzfile['indices'], npzfile['indptr']), shape=npzfile['shape'])
	return dict(values=mt, shape=mt.shape, index=npzfile['idx'] if with_idx and len(npzfile['idx'].shape) == 1 else None, columns=npzfile['col'] if len(npzfile['col'].shape) == 1 else None)


def df2gen(df, batch_size=32, cache_fpath='tmp_data.h5', table_id=None):
	'''
	Transform Pandas DataFrame into Generator
	'''
	if (type(df) == pd.io.parsers.TextFileReader):
		is_iter = True
	elif (type(df) != pd.DataFrame):
		X = pd.DataFrame(df)
	if (is_iter):
		for sub_df in df:
			yield sub_df
	else:
		table_id = table_id if table_id else str(int(time.time()))
		df.to_hdf(cache_fpath, table_id, format='table', data_columns=True)
		for sub_df in pd.read_hdf(cache_fpath, key=table_id, iterator=True, chunksize=batch_size):
			yield sub_df
			del sub_df


def write_yaml(data, fpath, append=False, dfs=False):
#	fs.mkdir(os.path.dirname(fpath))
	fpath = os.path.splitext(fpath)[0] + '.yaml'
	with open(fpath, 'a' if append else 'w') as f:
		yaml.dump(data, f, default_flow_style=dfs)


def read_yaml(fpath):
	fpath = os.path.splitext(fpath)[0] + '.yaml'
	if (os.path.exists(fpath)):
		with open(fpath, 'r') as f:
			return yaml.load(f)
	else:
		print(('File %s does not exist!' % fpath))


def param_writer(fpath):
	data = {}
	def add_params(mdl_t, mdl_name, params, finished=False, data=data):
		if (mdl_t and mdl_name and params):
			mdl_list = data.setdefault(mdl_t, [])
			for mdl in mdl_list:
				if (mdl['name'] == mdl_name):
					mdl['params'] = params
					break
			else:
				mdl_list.append(dict(name=mdl_name, params=params))
		if (finished):
			try:
				write_yaml(data, fpath)
			except:
				print(('Cannot writer the data: %s' % data))
				return
	return add_params


def param_reader(fpath):
	data = read_yaml(fpath)
	def get_params(mdl_t, mdl_name, data=data):
		if (not data):
			print(('Cannot find the config file: %s' % fpath))
			return {}
		if (mdl_t in data):
			mdl_list = data[mdl_t]
		else:
			print(('Model type %s does not exist.' % mdl_t))
			return {}
		for mdl in mdl_list:
			if (mdl['name'] == mdl_name):
				return mdl['params']
		else:
			print(('Parameters of model %s does not exist.' % mdl_name))
			return {}
	return get_params


def cfg_reader(fpath):
	data = read_yaml(fpath)
	def get_params(module, function, data=data):
		if (not data):
			print(('Cannot find the config file: %s' % fpath))
			return {}
		if (module in data):
			func_list = data[module]
		else:
			print(('Module %s does not exist.' % module))
			return {}
		for func in func_list:
			if (func['function'] == function):
				return func['params']
		else:
			print(('Parameters of function %s does not exist.' % func))
			return {}
	return get_params


def lockf(flpath, wait_time=1):
	if sys.platform.startswith('linux2'):
		import fcntl
	else:
		return
	x = flpath if type(flpath) is file else open(flpath, 'w+')
	while True:
		try:
			fcntl.flock(x, fcntl.LOCK_EX | fcntl.LOCK_NB)
			break
		except IOError as e:
			if e.errno != errno.EAGAIN:
				raise
			else:
				time.sleep(wait_time)
	return x

def unlockf(flpath):
	if sys.platform.startswith('linux2'):
		import fcntl
	else:
		return
	x = flpath if type(flpath) is file else open(flpath, 'w+')
	fcntl.flock(x, fcntl.LOCK_UN)
	x.close()
