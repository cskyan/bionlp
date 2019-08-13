#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: umls.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2019-08-08 15:56:23
###########################################################################
#

import os, sys, time, copy, json, requests
requests.packages.urllib3.disable_warnings()
from collections import OrderedDict
from html.parser import HTMLParser

import ftfy
from apiclient import APIClient

# from . import xmlextrc
from bionlp.spider import xmlextrc


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
UMLS_PATH = os.path.join(DATA_PATH, 'umls')
API_KEY = '1fba7f61-e441-438d-b265-e000f7f18d4d'
TGT = 'TGT-376970-QYy5MSSJkmasElP4YXFjMzn0JUy2CGe73LB3dLI62MNlJX0y0e-cas'


def fetch_umls(ids=[], info=None, **kwargs):
	client = UMLSAPI(apikey=API_KEY, function='concept')
	res = [client.call(dict(cui=uid, info=info), **kwargs) for uid in ids]
	return [r['result'] for r in res]


def fetch_snomedct_us(ids=[], info=None, **kwargs):
	client = UMLSAPI(apikey=API_KEY, function='source')
	res = [client.call(dict(source='SNOMEDCT_US', id=uid, info=info), **kwargs) for uid in ids]
	return [r['result'] for r in res]


def fetch_msh(ids=[], info=None, **kwargs):
	client = UMLSAPI(apikey=API_KEY, function='source')
	res = [client.call(dict(source='MSH', id=uid, info=info), **kwargs) for uid in ids]
	return [r['result'] for r in res]


class TGTParser(HTMLParser):
	def __init__(self):
		super(TGTParser, self).__init__()
		self.tgt = ''
		self._tag = ''

	def handle_starttag(self, tag, attrib):
		self._tag = tag
		attrib_dict = dict(attrib)
		if (self._tag == 'form'):
			self.tgt = attrib_dict['action'].split('/')[-1]

	def build(self):
		return self.tgt


class UMLSAPI(APIClient, object):
	BASE_URL = 'https://uts-ws.nlm.nih.gov/rest/'
	APIKEY_URL = 'https://utslogin.nlm.nih.gov/cas/v1/api-key'
	TICKET_URL = 'https://utslogin.nlm.nih.gov/cas/v1/api-key/'
	SERVICE_URL = 'http://umlsks.nlm.nih.gov'
	_function_url = {'concept':'content/current/CUI/', 'source':'content/current/source/'}
	_url_param = {'concept':OrderedDict(cui='', info=None), 'source':OrderedDict(source='', id='', info=None)}
	_default_param = {'concept':dict(ticket=''), 'source':dict(ticket='')}
	_func_restype = {'concept':'json', 'source':'json'}
	def __init__(self, apikey='', function='concept'):
		if (function not in self._default_param):
			raise ValueError('The function %s is not supported!' % function)
		APIClient.__init__(self)
		self.apikey = apikey
		self.tgt = TGT
		try:
			self.authenticate()
		except Exception as e:
			self.request_tgt()
		self.function = function
		self.func_url = self._function_url[function]
		self.restype = self._func_restype[function]
	def _handle_response(self, response):
		if (self.restype == 'xml'):
			# Builder = BUILDER_MAP[self.function]
			# builder = Builder()
			# parser = xmlextrc.get_parser(builder)
			# try:
			# 	parser.feed(nlp.clean_text(response.data, encoding='utf-8', replacement=None))
			# except Exception as err:
			# 	print('Can not parse the response of API call!')
			# 	raise err
			# parser.close()
			# return builder.build()
			pass
		elif (self.restype == 'json'):
			if (response.status != 200): raise ConnectionError('Server error! Please wait a second and try again.')
			return json.loads(ftfy.fix_text(response.data.decode('utf-8')).replace('\\', ''))

	def request_tgt(self):
		res = requests.post(UMLSAPI.APIKEY_URL, data=dict(apikey=self.apikey))
		parser = TGTParser()
		try:
			parser.feed(ftfy.fix_text(res.text))
		except Exception as err:
			print('Can not parse the response of TGT authentication call!')
			raise err
		parser.close()
		self.tgt = parser.build()
		print('Please update your new TGT: %s' % self.tgt)

	def authenticate(self):
		res = requests.post(UMLSAPI.TICKET_URL+self.tgt, data=dict(service=UMLSAPI.SERVICE_URL), headers={'content-type': 'application/x-www-form-urlencoded'})
		if (res.status_code != 200): raise ConnectionError('Cannot request a service ticket!')
		return res._content

	def call(self, urlargs, timeout=100, interval=5, **kwargs):
		args = copy.deepcopy(self._default_param[self.function])
		args.update((k, v) for k, v in kwargs.items() if k in args)
		elapsed, ret_error = 0, True
		while ret_error and elapsed < timeout:
			try:
				args['ticket'] = self.authenticate()
				self.request_tgt()
				ret_error = False
			except ConnectionError as e:
				print(e)
				time.sleep(interval)
				elapsed += interval
		urlparams = copy.deepcopy(self._url_param[self.function])
		urlparams.update((k, v) for k, v in urlargs.items() if k in urlparams)
		url = self.func_url + '/'.join(filter(None, urlparams.values()))
		elapsed, ret_error = 0, True
		while ret_error and elapsed < timeout:
			try:
				res = APIClient.call(self, url, **args)
				ret_error = False
			except ConnectionError as e:
				print(e)
				time.sleep(interval)
				elapsed += interval
		return {} if ret_error else res


if __name__ == '__main__':
	cuis = ['C1561643', 'C0007222']
	print([c['name'] for c in fetch_umls(ids=cuis)])
	print([c for c in fetch_umls(ids=cuis, info='definitions')])
	snomedct_us_ids = ['9468002', '204958008', '236403004']
	print([c['name'] for c in fetch_snomedct_us(ids=snomedct_us_ids)])
	msh_ids = ['D065710', 'C562884', 'D005155']
	print([c['name'] for c in fetch_msh(ids=msh_ids)])
