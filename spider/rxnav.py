#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2017 by Caspar. All rights reserved.
# File Name: rxnav.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2017-03-30 21:01:43
###########################################################################
#

import os
import sys
import copy
import json
import xmlextrc

from apiclient import APIClient
# from abc import ABCMeta

from .. import nlp


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux2'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
RXNAV_PATH = os.path.join(DATA_PATH, 'rxnav')

		
class DrugBuilder():
#	__metaclass__ = ABCMeta

	def __init__(self):
		self._tag = ''
		self._tag_stack = []
		self.name = ''
		self.concept_group = []
		
	def start(self, tag, attrib):
		self._tag = tag
		self._tag_stack.append(self._tag)
		if (self._tag == 'conceptGroup'):
			self.concept_group.append({})
		if (self._tag == 'conceptProperties'):
			self.concept_group[-1].setdefault('property', []).append({})
	
	def end(self, tag):
		self._tag = tag
		self._tag_stack.pop()
		
	def data(self, data):
		if data.isspace(): return
		data = data.strip()
		# Process the text content
		if (self._tag == 'tty'):
			if (self._tag_stack[-2] == 'conceptGroup'):
				self.concept_group[-1]['tty'] = data
			elif (self._tag_stack[-2] == 'conceptProperties'):
				self.concept_group[-1]['property'][-1]['tty'] = data
		if (self._tag == 'rxcui'):
			self.concept_group[-1]['property'][-1]['rxcui'] = data
		if (self._tag == 'name'):
			if (self._tag_stack[-2] == 'drugGroup'):
				self.name = data
			elif (self._tag_stack[-2] == 'conceptProperties'):
				self.concept_group[-1]['property'][-1]['name'] = data
		if (self._tag == 'synonym'):
			self.concept_group[-1]['property'][-1]['synonym'] = data
		if (self._tag == 'language'):
			self.concept_group[-1]['property'][-1]['language'] = data
		if (self._tag == 'umlscui'):
			self.concept_group[-1]['property'][-1]['umlscui'] = data

	def close(self):
		pass

	def build(self):
		return {'name':self.name,'concept_group':self.concept_group}
		
		
BUILDER_MAP = {'drugs':DrugBuilder, 'interaction':DrugBuilder}


class RxNavAPI(APIClient, object):
	BASE_URL = 'https://rxnav.nlm.nih.gov/REST'
	_function_url = {'drugs':'drugs', 'interaction':'interaction/interaction.json'}
	_default_param = {'drugs':dict(name=''), 'interaction':dict(rxcui='')}
	_func_restype = {'drugs':'xml', 'interaction':'json'}
	def __init__(self, function='drugs'):
		if (not self._default_param.has_key(function)):
			raise ValueError('The function %s is not supported!' % function)
		# super(APIClient, self).__init__()
		# APIClient is old-stype class
		APIClient.__init__(self)
		self.function = function
		self.func_url = self._function_url[function]
		self.restype = self._func_restype[function]
	def _handle_response(self, response):
		if (self.restype == 'xml'):
			Builder = BUILDER_MAP[self.function]
			builder = Builder()
			parser = xmlextrc.get_parser(builder)
			try:
				parser.feed(nlp.clean_text(response.data, encoding='utf-8', replacement=None))
			except Exception as err:
				print 'Can not parse the response of API call!'
				raise err
			parser.close()
			return builder.build()
		elif (self.restype == 'json'):
			return json.dumps(nlp.clean_text(response.data, encoding='utf-8', replacement=None)).replace('\\', '')
	
	def call(self, **kwargs):
		args = copy.deepcopy(self._default_param[self.function])
		args.update((k, v) for k, v in kwargs.iteritems() if args.has_key(k))
		return APIClient.call(self, '/%s'% self.func_url, **args)