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
from bionlp.util import io


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
OMIM_PATH = os.path.join(DATA_PATH, 'omim')
API_KEY = 'btpdnKMaTTaq87fvYPgl9A'
SC=';;'


def omim_refs(omim_ids):
	client = OMIMAPI(function='entref')
	res = [client.call(mimNumber=omimid) for omimid in omim_ids]
	return [[ref['reference']['pubmedID'] for refs in r['omim']['referenceLists'] for ref in refs['referenceList'] if 'pubmedID' in ref['reference']] for r in res]


class OMIMAPI(APIClient, object):

	BASE_URL = 'https://api.omim.org/api'
	_function_url = {'entref':'/entry/referenceList'}
	_default_param = {'entref':dict(apiKey=API_KEY, format='json', mimNumber='')}
	_func_restype = {'entref':'json'}

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
				time.sleep(interval)
				trail += 1
		return res


if __name__ == '__main__':
	print(omim_refs(['100100']))
