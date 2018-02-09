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
import re
import sys
import codecs
import StringIO
import cStringIO


def mkdir(path):
	if path and not os.path.exists(path):
		print "Creating folder: " + path
		os.makedirs(path)
		

def read_file(fpath, code='ascii'):
	if (isinstance(fpath, StringIO.StringIO) or isinstance(fpath, cStringIO.InputType)):
		return fpath.readlines()
	try:
		data_str = []
		if (code.lower() == 'ascii'):
			with open(fpath, 'r') as fd:
				for line in fd.readlines():
					data_str.append(line)
		else:
			with codecs.open(fpath, mode='r', encoding=code, errors='ignore') as fd:
				for line in fd.readlines():
					data_str.append(line)
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
	
	
def write_file(content, fpath, code='ascii'):
	if (isinstance(fpath, StringIO.StringIO) or isinstance(fpath, cStringIO.OutputType)):
		fpath.write(content)
		return
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
		sys.exit(-1)
		
		
def write_files(contents, fpaths, code='ascii'):
	for i in xrange(min(len(fpaths), len(contents))):
		try:
			write_file(contents[i], fpaths[i], code)
		except Exception as e:
			continue

			
def pardir(path):
	return os.path.abspath(os.path.join(path, os.pardir))
			
		
def listf(path, pattern='.*', full_path=False):
	prog = re.compile(pattern)
	if (full_path):
		return [os.path.join(path, f) for f in os.listdir(path) if prog.match(f) and os.path.isfile(os.path.join(path, f))]
	else:
		return [f for f in os.listdir(path) if prog.match(f) and os.path.isfile(os.path.join(path, f))]


def traverse(path):
	for root, dirs, files in os.walk(path):
		for file in files:
			fpath = os.path.join(root, file)
			yield fpath