#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: ncbo.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-06-27 15:56:23
###########################################################################
#

import os, sys, copy, time, json

from apiclient import APIClient
import ftfy

# from ..util import ontology
# from bionlp.util import ontology


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ANT_PATH = os.path.join(DATA_PATH, 'ncbo')
API_KEY = '8e9d4dda-89e8-486b-811f-b96e4ef50c20'
SC=';;'


def annotext(text, ontos=[]):
	client = NCBOAPI(apikey=API_KEY, function='annotate')
	res = client.call(text=text, ontologies=','.join(ontos).upper())
	return [dict(id=annot['annotatedClass']['@id'].split('/')[-1], url=annot['annotatedClass']['@id'], antxt=[dict(loc=(antxt['from'], antxt['to']), text=antxt['text'], mtype=antxt['matchType']) for antxt in annot['annotations']]) for annot in res]


class NCBOAPI(APIClient, object):

	BASE_URL = 'http://data.bioontology.org'
	_function_url = {'annotate':'annotator'}
	_default_param = {'annotate':dict(text='', ontologies='')}
	_func_restype = {'annotate':'json'}
	_parm_options = {'annotate':{'ontologies':['HP', 'UBERON', 'MP', 'CL', 'NCIT', 'EFO']}}

	def __init__(self, apikey='', function='annotate'):
		if (function not in self._default_param):
			raise ValueError('The function %s is not supported!' % function)
		APIClient.__init__(self)
		self.apikey = apikey
		self.function = function
		self.func_url = self._function_url[function]
		self.restype = self._func_restype.setdefault(function, 'json')

	def _handle_response(self, response):
		if (self.restype == 'xml'):
			Builder = BUILDER_MAP[self.function]
			builder = Builder()
			parser = xmlextrc.get_parser(builder)
			try:
				parser.feed(ftfy.fix_text(response.data))
			except Exception as err:
				print('Can not parse the response of API call!')
				raise err
			parser.close()
			return builder.build()
		elif (self.restype == 'json'):
			# return json.loads(ftfy.fix_text(response.data.decode('utf-8')).replace('\\', ''))
			res = ftfy.fix_text(response.data.decode('utf-8', errors='replace')).replace('\\', '')
			try:
				res_json = json.loads(res)
			except Exception as e:
				print(e)
				print(res)
				res_json = {}
			return res_json

	def call(self, max_trail=-1, interval=3, **kwargs):
		args = copy.deepcopy(self._default_param[self.function])
		args['apikey'] = self.apikey
		args.update((k, v) for k, v in kwargs.items() if k in args)
		trail = 0
		while max_trail <= 0 or trail < max_trail:
			try:
				res = APIClient.call(self, '/%s' % self.func_url, **args)
				break
			except Exception as e:
				print(e)
				import time
				time.sleep(interval)
				trail += 1
		return res

if __name__ == '__main__':
	text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
	print([a['id'] for a in annotext(text, ontos=['HP'])])
