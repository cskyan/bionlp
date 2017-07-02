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
import json


def mmc_json_srlz(key, value):
	if (type(value) == str):
		return value, 1
	return json.dumps(value), 2

def mmc_json_desrlz(key, value, flags):
	if (flags == 1):
		return value
	if (flags == 2):
		return json.loads(value)
	raise Exception("Unknown serialization format")