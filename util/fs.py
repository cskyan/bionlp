#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: fs.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-03-01 21:21:37
###########################################################################
#

import os
import codecs


def mkdir(path):
	if not os.path.exists(path):
		print "Creating folder: " + path
		os.makedirs(path)
		

def read_file(fpath, code='ascii'):
	try:
		data_str = []
		if (code.lower() == 'ascii'):
			with open(fpath, 'r') as fd:
				for line in fd.readlines():
					data_str.append(line.strip())
		else:
			with codecs.open(fpath, mode='r', encoding=code, errors='ignore') as fd:
				for line in fd.readlines():
					data_str.append(line.strip())
	except Exception as e:
		print e
		print 'Can not open the file \'%s\'!' % fpath
		raise
	return data_str
	
	
def read_files(fpaths, code='ascii'):
	for fpath in fpaths:
		try:
			yield read_file(fpath, code)
		except Exception as e:
			continue
	
	
def write_file(fpath, content, code='ascii'):
	try:
		if (code.lower() == 'ascii'):
			with open(fpath, mode='w') as fd:
				fd.write(content)
				fd.close()
		else:
			with codecs.open(fpath, mode='w', encoding=code, errors='ignore') as fd:
				fd.write(content)
				fd.close()
	except Exception as e:
		print e
		print 'Can not write to the file \'%s\'!' % fpath
		exit(-1)
		
		
def write_files(fpaths, contents, code='ascii'):
	for i in xrange(min(len(fpaths), len(contents))):
		try:
			write_file(fpaths[i], contents[i], code)
		except Exception as e:
			continue

		
def listf(path, full_path=False):
	if (full_path):
		return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
	else:
		return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def traverse(path):
	for root, dirs, files in os.walk(path):
		for file in files:
			fpath = os.path.join(root, file)
			yield fpath