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

import os, io, re, sys, codecs


def mkdir(path):
	if path and not os.path.exists(path):
		print(("Creating folder: " + path))
		os.makedirs(path)


def read_file(fpath, code='ascii'):
	if (isinstance(fpath, io.StringIO)):
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
		print(e)
		print(('Can not open the file \'%s\'!' % fpath))
		raise
	return data_str


def read_files(fpaths, code='ascii'):
	for fpath in fpaths:
		try:
			yield read_file(fpath, code)
		except Exception as e:
			continue


def write_file(content, fpath, code='ascii'):
	if (isinstance(fpath, io.StringIO)):
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
		print(e)
		print(('Can not write to the file \'%s\'!' % fpath))
		sys.exit(-1)


def write_files(contents, fpaths, code='ascii'):
	for i in range(min(len(fpaths), len(contents))):
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

def count_lines(fpath):
	import subprocess, shlex
	wc = subprocess.Popen(shlex.split('wc -l %s' % fpath), stdout=subprocess.PIPE)
	num_lines = int(wc.communicate()[0].split()[0])
	return num_lines

def read_last_line(fpath):
	with open(fpath, 'rb') as fd:
	    offset = -100
	    while True:
	        fd.seek(offset, 2)
	        lines = fd.readlines()
	        if len(lines) > 1:
	            last = lines[-1]
	            break
	        offset *= 2
	return last.decode()

def read_firstlast_line(fpath):
	with open(fpath, 'rb') as fd:
	    first = next(fd)
	    offset = -100
	    while True:
	        fd.seek(offset, 2)
	        lines = fd.readlines()
	        if len(lines) > 1:
	            last = lines[-1]
	            break
	        offset *= 2
	return first.decode(), last.decode()
