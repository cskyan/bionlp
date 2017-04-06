#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: sparql.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-11-24 12:04:12
###########################################################################
#

import os
import re
import sys
import codecs
import json

from rdflib import URIRef, Variable
from rdflib.query import ResultRow
from SPARQLWrapper import SPARQLWrapper, JSON

from ..util import fs

if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp\\sparql'
elif sys.platform.startswith('linux2'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp', 'sparql')
SC=';;'
		
		
class SPARQL(object):
	def __init__(self, endpoint):
		self.sparql = SPARQLWrapper(endpoint)
		
	@classmethod
	def prepareQuery(cls, q_str, initNs={}):
		prefix = []
		for k, v in initNs.iteritems():
			prefix.append('PREFIX %s: <%s>' % (k, v))
		return '\n'.join(prefix) + '\n' + q_str
		
	def query(self, q):
		self.sparql.setQuery(q)
		self.sparql.setReturnFormat(JSON)
		results = self.sparql.query().convert()
		result_rows = []
		for result in results['results']['bindings']:
			rr_list, rr_dict = [], {}
			for k, v in result.iteritems():
				rr_dict[Variable(k)] = URIRef(v['value'])
				rr_list.append(Variable(k))
			result_rows.append(ResultRow(rr_dict, rr_list))
		return result_rows
		

class MeSHSPARQL(SPARQL):
	def __init__(self, endpoint='https://id.nlm.nih.gov/mesh/sparql'):
		super(MeSHSPARQL, self).__init__(endpoint)