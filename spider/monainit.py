#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: monainit.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-06-27 15:56:23
###########################################################################
#

import os, sys, json, copy

from html.parser import HTMLParser
from apiclient import APIClient
import ftfy

# from ..util import ontology
# from bionlp.util import ontology


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ANT_PATH = os.path.join(DATA_PATH, 'monainit')
SC=';;'


def annotext(text, ontos=[]):
	client = MonaInitAPI(function='annotate')
	res = client.call(content=text)
	if len(ontos) > 0:
		res = [r for r in res if any([r['id'].startswith(onto.upper()) for onto in ontos])] if len(ontos) > 0 else res
	for r in res: r['id'] = r['id'].replace(':', '_')
	return res


class AnnotParser(HTMLParser):
	def __init__(self):
		super(AnnotParser, self).__init__()
		self.offset = 0
		self._tag = ''
		self._tag_stack = []
		self._annots = []
		self.annots = []

	def handle_starttag(self, tag, attrib):
		if tag in ['br', 'link', 'input']: return
		self._tag = tag
		self._tag_stack.append(self._tag)
		if (self._tag == 'span'):
			for att in attrib:
				att_txt = list(zip(att[0].split('\n'), att[1].split('\n')))[0]
				if att_txt[0] == 'data-scigraph':
					annot_txt = [annot.split(',') for annot in att_txt[1].split('|')]
					self._annots = [dict(id=annot[-2], loc=[self.offset, self.offset], text='', type=annot[-1]) for annot in annot_txt]

	def handle_endtag(self, tag):
		if tag in ['br', 'link', 'input']: return
		self._tag = tag
		self._tag_stack.pop()
		self._tag = self._tag_stack[-1] if len(self._tag_stack) > 0 else ''

	def handle_data(self, data):
		if data.isspace(): return
		# Process the text content
		len_txt = len(data)
		if (self._tag == 'span'):
			for i in range(len(self._annots)):
				self._annots[i]['loc'][1] += len_txt
				self._annots[i]['text'] = data
			self.annots.extend(self._annots)
			self._annots = []
		self.offset += len_txt

	def build(self):
		return self.annots


class MonaInitAPI(APIClient, object):

	BASE_URL = 'https://scigraph-ontology.monarchinitiative.org/scigraph'
	_function_url = {'annotate':'annotations'}
	_default_param = {'annotate':dict(content='', minLength=4, longestOnly='false', includeAbbrev='false', includeAcronym='false', includeNumbers='false')}
	_func_restype = {'annotate':'html'}
	_parm_options = {'annotate':{}}

	def __init__(self, function='annotate'):
		if (function not in self._default_param):
			raise ValueError('The function %s is not supported!' % function)
		APIClient.__init__(self)
		self.function = function
		self.func_url = self._function_url[function]
		self.restype = self._func_restype.setdefault(function, 'json')

	def _handle_response(self, response):
		if (self.restype == 'html'):
			parser = AnnotParser()
			try:
				parser.feed(ftfy.fix_text(response.data.decode('utf-8')))
			except Exception as err:
				print('Can not parse the response of API call!')
				raise err
			parser.close()
			return parser.build()
		elif (self.restype == 'json'):
			return json.loads(ftfy.fix_text(response.data.decode('utf-8', errors='replace')).replace('\\', ''))

	def call(self, **kwargs):
		args = copy.deepcopy(self._default_param[self.function])
		args.update((k, v) for k, v in kwargs.items() if k in args)
		return APIClient.call(self, '/%s' % self.func_url, **args)

if __name__ == '__main__':
	text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
	print([a['id'] for a in annotext(text, ontos=['HP'])])
