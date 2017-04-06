#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: oo.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-11-21 21:57:35
###########################################################################
#

import os
import cProfile


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def iprofile(func):
	def target_func(*args, **kwargs):
		profile = cProfile.Profile()
		try:
			profile.enable()
			result = func(*args, **kwargs)
			profile.disable()
			return result
		finally:
			profile.print_stats()
	return target_func