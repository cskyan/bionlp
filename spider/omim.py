#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: omim.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-10-10 15:57:59
###########################################################################
#

import os, sys, copy, time, json, urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from html.parser import HTMLParser
from apiclient import APIClient
import ftfy

# from ..util import ontology
# from bionlp.util import ontology
# from ..util import io
# import xmlextrc
from bionlp.util import io
from bionlp.spider import xmlextrc


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
OMIM_PATH = os.path.join(DATA_PATH, 'omim')
API_KEY = 'btpdnKMaTTaq87fvYPgl9A'
SC=';;'


def omim_refs(omim_ids):
	client = OMIMAPI(function='entref')
	unique_omim_ids = set(omim_ids)
	print('Querying publications for OMIM ids: %s' % ', '.join(unique_omim_ids))
	sys.stdout.flush()
	res = [client.call(mimNumber=omimid) for omimid in omim_ids]
	res_map = dict(zip(unique_omim_ids, [[ref['reference']['pubmedID'] for refs in r['omim']['referenceLists'] for ref in refs['referenceList'] if 'pubmedID' in ref['reference']] if r else [] for r in res]))
	return [res_map[omimid] for omimid in omim_ids]


class RefBuilder():
	def __init__(self):
		self._tag = ''
		self._tag_stack = []
		self.referenceLists = []
		self._referenceList = []
		self._ref = {}

	def start(self, tag, attrib):
		self._tag = tag
		self._tag_stack.append(self._tag)
		if (self._tag == 'referenceList'):
			self._referenceList = []
		if (self._tag == 'reference'):
			self._ref = {}

	def end(self, tag):
		self._tag = tag
		self._tag_stack.pop()
		if (self._tag == 'referenceList'):
			self.referenceLists.append(self._referenceList)
		if (self._tag == 'reference'):
			self._referenceList.append(self._ref)

	def data(self, data):
		if data.isspace(): return
		data = data.strip()
		# Process the text content
		if (self._tag == 'pubmedID' and self._tag_stack[-2] == 'reference'):
			self._ref['pmid'] = data

	def build(self):
		return {'omim':{'referenceLists':[{'referenceList':[{'reference':ref} for ref in refs]} for refs in self.referenceLists]}}


BUILDER_MAP = {'entref':RefBuilder}


class OMIMAPI(APIClient, object):

	BASE_URL = 'https://api.omim.org/api'
	_function_url = {'entref':'/entry/referenceList'}
	_default_param = {'entref':dict(apiKey=API_KEY, format='json', mimNumber='')}
	_func_restype = {'entref':'xml'}

	def __init__(self, function='entref'):
		if (function not in self._default_param):
			raise ValueError('The function %s is not supported!' % function)
		APIClient.__init__(self)
		self.function = function
		self.func_url = self._function_url[function]
		self.restype = self._func_restype.setdefault(function, 'json')

	def _handle_response(self, response):
		if (self.restype == 'json'):
			res = {}
			if (response.status != 200): raise ConnectionError('Server error! Please wait a second and try again.')
			res_str = ftfy.fix_text(response.data.decode('utf-8')).replace('\\', '')
			try:
				res = io.load_json(res_str)
			except json.JSONDecodeError as e:
				print(e)
				print('Cannot deserialize the json data:\n%s' % res_str)
			except Exception as e:
				print(e)
			return res
		elif (self.restype == 'xml'):
			Builder = BUILDER_MAP[self.function]
			builder = Builder()
			parser = xmlextrc.get_parser(builder)
			try:
				parser.feed(response.data.decode('utf-8'))
			except Exception as err:
				print('Can not parse the response of API call!')
				raise err
			parser.close()
			return builder.build()
		else:
			return {}

	def call(self, max_trail=-1, interval=3, **kwargs):
		args = copy.deepcopy(self._default_param[self.function])
		args.update((k, v) for k, v in kwargs.items() if k in args)
		trail = 0
		while max_trail <= 0 or trail < max_trail:
			try:
				res = APIClient.call(self, '%s' % self.func_url, **args)
				break
			except Exception as e:
				print(e)
				sys.stdout.flush()
				time.sleep(interval)
				trail += 1
		return res


if __name__ == '__main__':
	print(omim_refs(['100100']))
