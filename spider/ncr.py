#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: ncr.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-08-28 06:46:35
###########################################################################
#

import os, sys, json, copy
from io import StringIO

import pandas as pd

import ftfy

# from ..util import ontology
# from bionlp.util import ontology

SRC_PATH = os.path.join(os.path.expanduser("~"), 'source', 'py', 'NeuralCR')
sys.path.append(SRC_PATH)
import ncrmodel

if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ANT_PATH = os.path.join(DATA_PATH, 'ncr')
SC=';;'
GLB_MDL = None


def annotext(text, ontos=[], param_dir='model_params', word_model_file='model_params/pmc_model_new.bin'):
	global GLB_MDL
	model = GLB_MDL = ncrmodel.NCR.loadfromfile(param_dir, word_model_file) if GLB_MDL is None else GLB_MDL
	res = model.annotate_text(ftfy.fix_text(text), 0.8)
	return [dict(id=r[2].replace(':', '_'), loc=r[:2], conf=r[3]) for r in res]


if __name__ == '__main__':
    text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
    print([a['id'] for a in annotext(text, param_dir=os.path.join(os.path.expanduser("~"), 'downloads/model_params'), word_model_file=os.path.join(os.path.expanduser("~"), 'downloads/model_params/pmc_model_new.bin'))])
