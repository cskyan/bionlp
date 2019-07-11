#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: mseqdr.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-07-08 11:25:32
###########################################################################
#

import os, re, sys, json, copy, requests
from html.parser import HTMLParser

import pandas as pd

from apiclient import APIClient
import ftfy

# from ..util import ontology
# from bionlp.util import ontology


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ANT_PATH = os.path.join(DATA_PATH, 'mseqdr')
SC=';;'


def annotext(text, ontos=[], func='gethpo', **kwargs):
	client = MSeqDRAPI(function=func)
	res = client.call(hponame=text)
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
		self._enter_table = False
		self._enter_form = False
		self._enter_tbcontent = False
		self._enter_tbcell = False
		self._enter_termid = False
		self._enter_term = False
		self._done_columns = False
		self._columns = []
		self._annot = []
		self._annots = []
		self.nomatch = re.compile('No .* match')

	def handle_starttag(self, tag, attrib):
		if tag in ['br', 'link', 'input']: return
		self._tag = tag
		self._tag_stack.append(self._tag)
		attrib_dict = dict(attrib)
		if (self._tag == 'table' and ('id', 'resultgrid') in attrib):
			self._enter_table = True
		elif (self._tag == 'form' and ('id', 'searchfrm') in attrib):
			self._enter_form = True
		elif (self._tag == 'tr' and ('style', 'background-color:#ffab0a') in attrib):
			self._enter_tbcontent = True
		elif (self._tag == 'td' and self._enter_tbcontent):
			self._enter_tbcell = True
		elif (self._tag == 'a' and self._enter_tbcell):
			if attrib_dict.setdefault('href', '').startswith('hpo_browser.php?'): self._enter_termid = True
			elif attrib_dict.setdefault('href', '').startswith('http://www.human-phenotype-ontology.org/hpoweb/showterm?'): self._enter_term = True

	def handle_endtag(self, tag):
		if tag in ['br', 'link', 'input']: return
		self._tag = tag
		starttag = self._tag_stack.pop()
		if (self._tag == 'table' and self._enter_table):
			self._enter_tbcontent = False
			self._enter_table = False
		elif (self._tag == 'form' and self._enter_form):
			self._enter_form = False
		elif (self._tag == 'tr' and self._enter_tbcontent):
			if (not self._done_columns):
				self._done_columns = True
			else:
				if (len(self._annot) and re.match(self.nomatch, self._annot[0]) is None) > 0: self._annots.append(self._annot)
				self._annot = []
		elif (self._tag == 'td' and self._enter_tbcontent):
			self._enter_tbcell = False
		elif (self._tag == 'a'):
			if self._enter_termid: self._enter_termid = False
			elif self._enter_term: self._enter_term = False
		self._tag = self._tag_stack[-1] if len(self._tag_stack) > 0 else ''


	def handle_data(self, data):
		if data.isspace(): return
		# Process the text content
		if (self._tag == 'td' and self._tag_stack[-2] == 'tr' and self._enter_tbcontent):
			if (not self._done_columns):
				self._columns.append(data.strip())
			else:
				self._annot.append(data.strip())
		elif (self._tag == 'a' and self._tag_stack[-2] == 'td' and (self._enter_termid or self._enter_term)):
			self._annot.append(data.strip())

	def build(self):
		# df = pandas.DataFrame(dict(zip(self._columns, zip(*self._annots))))
		# print(self._annots)
		return [dict(id=ann[2]) for ann in self._annots]


class MSeqDRAPI(APIClient, object):

	BASE_URL = 'https://mseqdr.org/clinical'
	_function_url = {'gethpo':'search_phenotype.php'}
	_default_param = {'gethpo':dict(hponame='', dbsource='HPO')}
	_func_restype = {'gethpo':'html'}
	_parm_options = {'gethpo':{}}

	def __init__(self, function='gethpo'):
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
				parser.feed(ftfy.fix_text(response.data.decode('utf-8', errors='replace')).replace('</TD><TR><TR><TD>', '</TD></TR><TR><TD>'))
			except Exception as err:
				print('Can not parse the response of API call!')
				raise err
			parser.close()
			return parser.build()

	def call(self, **kwargs):
		args = copy.deepcopy(self._default_param[self.function])
		args.update((k, v) for k, v in kwargs.items() if k in args)
		return APIClient.call(self, '/%s' % self.func_url, **args)


if __name__ == '__main__':
    text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
    print([a['id'] for a in annotext(text, ontos=['HP'])])
