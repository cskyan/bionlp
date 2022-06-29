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
import subprocess, queue, threading, shlex


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


class InteractiveCMD(object):
    def __init__(self, cmd):
    	self.cmd = cmd
    	self.p = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
    	self.out_queue = queue.Queue()
    	self.err_queue = queue.Queue()

    def start(self, verbose=False):
        out_thread = threading.Thread(target=InteractiveCMD._enqueue_output, args=(self.p.stdout, self.out_queue))
        err_thread = threading.Thread(target=InteractiveCMD._enqueue_output, args=(self.p.stderr, self.err_queue))
        out_thread.daemon = True
        err_thread.daemon = True
        out_thread.start()
        err_thread.start()
        if verbose:
            print(InteractiveCMD._get_output(self.out_queue))
            print(InteractiveCMD._get_output(self.err_queue))

    def input(self, input_str, verbose=False):
        if verbose: print('Your input: %s' % input_str)
        self.p.stdin.write(input_str.strip('\n')+'\n')
        self.p.stdin.flush()
        if verbose:
            self.print_output_error()

    def inputs(self, inputs, intervel=0, verbose=False):
    	for input_str in inputs:
    		self.input(input_str, verbose=verbose)
    		time.sleep(intervel)

    def print_output_error(self):
        print(InteractiveCMD._get_output(self.out_queue))
        print(InteractiveCMD._get_output(self.err_queue))

    def _enqueue_output(out, tgt_queue):
    	for line in iter(out.readline, b''):
    		tgt_queue.put(line)
    	out.close()


    def _get_output(out_queue):
    	out_str = ''
    	try:
    		while True:
    			out_str += out_queue.get_nowait()
    	except queue.Empty:
    		return out_str
