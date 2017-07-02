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
import json
import codecs
import hashlib

from rdflib import URIRef, Variable
from rdflib.query import ResultRow
from SPARQLWrapper import SPARQLWrapper, JSON

from ..util import fs, cache

if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp\\sparql'
elif sys.platform.startswith('linux2'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp', 'sparql')
SC=';;'
		
		
class SPARQL(object):
	def __init__(self, endpoint, use_cache=False):
		self.sparql = SPARQLWrapper(endpoint)
		self.use_cache = use_cache
		if (use_cache):
			self.memcached = ('localhost', 11211)
		
	@classmethod
	def prepareQuery(cls, q_str, initNs={}):
		prefix = []
		for k, v in initNs.iteritems():
			prefix.append('PREFIX %s: <%s>' % (k, v))
		return '\n'.join(prefix) + '\n' + q_str
		
	def query(self, q):
		results = None
		if (self.use_cache):
			q_hash = hashlib.md5(q).hexdigest()
			from pymemcache.client.base import Client
			client = Client(self.memcached, serializer=cache.mmc_json_srlz, deserializer=cache.mmc_json_desrlz)
			try:
				cache_res = client.get(q_hash)
				if (cache_res is not None):
					q_str, results = cache_res
					if (q.strip().lower() != q_str.strip().lower()):
						results = None
			except Exception as e:
				print e
		if (results is None):
			self.sparql.setQuery(q)
			self.sparql.setReturnFormat(JSON)
			results = self.sparql.query().convert()
			if (self.use_cache):
				try:
					client.set(q_hash, (q, results))
				except Exception as e:
					print e
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