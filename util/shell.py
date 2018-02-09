#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2017 by Caspar. All rights reserved.
# File Name: shell.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-06-12 10:53:48
###########################################################################
#

import time
import subprocess, shlex


def daemon(cmd, proc_name, interval=3):
	while True:
		numproc = check_numproc(proc_name)
		if (numproc == 0):
			subprocess.call(cmd, shell=True)
		time.sleep(interval)

	
def check_numproc(proc_name):
	ps = subprocess.Popen(shlex.split('ps -ef'), stdout=subprocess.PIPE)
	grep = subprocess.Popen(shlex.split('grep %s'%proc_name), stdin=ps.stdout, stdout=subprocess.PIPE)
	grepv = subprocess.Popen(shlex.split('grep -v grep --color=auto'), stdin=grep.stdout, stdout=subprocess.PIPE)
	wc = subprocess.Popen(shlex.split('wc -l'), stdin=grepv.stdout, stdout=subprocess.PIPE)
	numproc = int(wc.communicate()[0])
	return numproc