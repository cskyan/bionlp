#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: clinphen.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-07-01 01:51:12
###########################################################################
#

import os, sys, json, copy
from io import StringIO

import pandas as pd

import clinphen
import ftfy

# from ..util import ontology
# from bionlp.util import ontology


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ANT_PATH = os.path.join(DATA_PATH, 'clinphen')
SC=';;'


def annotext(text, ontos=[], umls=False, rare_pheno=False):
	custom_thesaurus = os.path.join(clinphen.srcDir, 'data', 'hpo_umls_thesaurus.txt') if umls else ''
	df = pd.read_table(StringIO(clinphen.main(StringIO(ftfy.fix_text(text)), custom_thesaurus, rare_pheno)))
	res = [dict(zip(df.columns, x)) for x in zip(*[df[col] for col in df.columns])]
	return [dict(id=r['HPO ID'].replace(':', '_'), name=r['Phenotype name'], occrns=r['No. occurrences'], text=r['Example sentence']) for r in res]


if __name__ == '__main__':
    text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
    print([a['id'] for a in annotext(text, ontos=['HP'])])
