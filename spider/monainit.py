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

import os, sys, json, copy, time, urllib3, requests
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
ANT_PATH = os.path.join(DATA_PATH, 'monainit')
SC=';;'


def annotext(text, ontos=[], mode='get', interval=10, **kwargs):
	client = MonaInitSciGraphAPI(function='annotate', mode=mode)
	res = client.call(content=text, **kwargs)
	res = [dict(id=r['token']['id'], text=' '.join(r['token']['terms']), loc=(r['start'], r['end'])) for r in res if any([r['token']['id'].startswith(onto.upper()) for onto in ontos])] if len(ontos) > 0 else res
	# res = [r for r in res if any([r['id'].startswith(onto.upper()) for onto in ontos])] if len(ontos) > 0 else res
	for r in res: r['id'] = r['id'].replace(':', '_')
	if interval>0: time.sleep(interval)
	return res


def phenopubs(pheno_ids, ontos=[]):
	client = MonaInitBioLinkAPI(function='phenopub')
	print('Querying publications for phenotypes: %s' % pheno_ids)
	res = [client.call(args=[phnid.replace('_', ':')]) for phnid in pheno_ids]
	# pheno_pubs = [[(r['subject']['id'].replace(':', '_'), pub['id']) for r in rs['associations'] for pub in r['publications']] for rs in res]
	pheno_pubs = [[(r['subject']['id'].replace(':', '_'), r['object']['id'], len(set([e['sub'] for e in r['evidence_graph']['edges'] if e['sub'].startswith('MONARCH')]))) for r in rs['associations']] for rs in res]
	pheno_pubs = [[pairs for pairs in r if any([pairs[0].startswith(onto.upper()) for onto in ontos])] for r in pheno_pubs] if len(ontos) > 0 else pheno_pubs
	return pheno_pubs

def phenodzs(pheno_ids, ontos=[]):
	client = MonaInitBioLinkAPI(function='phenodz')
	print('Querying diseases for phenotypes: %s' % pheno_ids)
	res = [client.call(args=[phnid.replace('_', ':')]) for phnid in pheno_ids]
	pheno_dzs = [[(r['subject']['id'].replace(':', '_'), r['object']['id'], len(set([e['sub'] for e in r['evidence_graph']['edges'] if e['sub'].startswith('MONARCH')]))) for r in rs['associations']] for rs in res]
	pheno_dzs = [[pairs for pairs in r if any([pairs[0].startswith(onto.upper()) for onto in ontos])] for r in pheno_dzs] if len(ontos) > 0 else pheno_dzs
	return pheno_dzs


def phenocases(pheno_ids, ontos=[]):
	client = MonaInitBioLinkAPI(function='phenocase')
	print('Querying cases for phenotypes: %s' % pheno_ids)
	res = [client.call(args=[phnid.replace('_', ':')]) for phnid in pheno_ids]
	pheno_cases = [[(r['subject']['id'].replace(':', '_'), r['object']['id'], len(set([e['sub'] for e in r['evidence_graph']['edges'] if e['sub'].startswith(':MONARCH')])), r['publications']) for r in rs['associations']] for rs in res]
	pheno_cases = [[pairs for pairs in r if any([pairs[0].startswith(onto.upper()) for onto in ontos])] for r in pheno_cases] if len(ontos) > 0 else pheno_cases
	return pheno_cases


def pubphenos(pmids, ontos=[]):
	client = MonaInitBioLinkAPI(function='pubpheno')
	print('Querying phenotypes for PMID: %s' % pmids)
	res = [client.call(args=['PMID:%s' % pmid]) for pmid in pmids]
	pub_phenos = [[(r['subject']['id'], r['object']['id'], len(set([n['id'] for n in r['evidence_graph']['nodes'] if n['id'].startswith('MONARCH')]))) for r in rs['associations']] for rs in res]
	pub_phenos = [[pairs for pairs in r if any([pairs[1].startswith(onto.upper()) for onto in ontos])] for r in pub_phenos] if len(ontos) > 0 else pub_phenos
	return pub_phenos


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


class MonaInitSciGraphAPI(APIClient, object):

	BASE_URL = 'https://scigraph-ontology.monarchinitiative.org/scigraph'
	_function_url = {'annotate-raw':'annotations', 'annotate':'annotations/complete'}
	_default_param = {'annotate-raw':dict(content='', minLength=4, longestOnly='false', includeAbbrev='false', includeAcronym='false', includeNumbers='false'), 'annotate':dict(content='', minLength=4, longestOnly='false', includeAbbrev='false', includeAcronym='false', includeNumbers='false')}
	_func_restype = {'annotate-raw':'html', 'annotate':'json'}

	def __init__(self, function='annotate', mode='get'):
		if (function not in self._default_param):
			raise ValueError('The function %s is not supported!' % function)
		APIClient.__init__(self)
		self.function = function
		self.func_url = self._function_url[function]
		self.restype = self._func_restype.setdefault(function, 'json')
		self.mode = mode

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
			try:
				res_str = ftfy.fix_text(response.data.decode('utf-8', errors='replace')).replace('\\', '')
				res = json.loads(res_str)
			except Exception as err:
				print(res_str)
				print(err)
				raise err
			return res

	def call(self, max_trail=-1, interval=3, **kwargs):
		args = copy.deepcopy(self._default_param[self.function])
		args.update((k, v) for k, v in kwargs.items() if k in args)
		for k in args.keys():
			if type(args[k]) is str and len(args[k]) > 20: args[k] = args[k].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
		trail = 0
		while max_trail <= 0 or trail < max_trail:
			try:
				if self.mode == 'post':
					headers={'content-type':'application/x-www-form-urlencoded', 'accept':'application/json'}
					res = requests.post('%s/%s' % (MonaInitSciGraphAPI.BASE_URL, self.func_url), data=args, headers=headers).json()
				else:
					res = APIClient.call(self, '/%s' % self.func_url, **args)
				break
			except json.JSONDecodeError as err:
				print(u'Encountered json decoding error when calling %s with parameters %s' % (u'%s/%s' % (MonaInitSciGraphAPI.BASE_URL, self.func_url), args))
				res = {}
				if trail > 2: break
				time.sleep(interval)
				trail += 1
			except Exception as e:
				print(e)
				time.sleep(interval)
				trail += 1
		return res


class MonaInitBioLinkAPI(APIClient, object):

	BASE_URL = 'https://api.monarchinitiative.org/api'
	_function_url = {'phenocase':'/bioentity/phenotype/{}/cases', 'phenodz':'/bioentity/phenotype/{}/diseases', 'phenopub':'/bioentity/phenotype/{}/publications', 'pubpheno':'/bioentity/publication/{}/phenotypes'}
	_default_param = {'phenocase':(1, dict()), 'phenodz':(1, dict()), 'phenopub':(1, dict()), 'pubpheno':(1, dict())} # (number of positional parameters to be filled in the url, keyword parameters to be added to the request)
	_func_restype = {'phenocase':'json', 'phenodz':'json', 'phenopub':'json', 'pubpheno':'json'}

	def __init__(self, function='phenopub'):
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

	def call(self, args, max_trail=-1, interval=3, **kwargs):
		num_args, kw_args = copy.deepcopy(self._default_param[self.function])
		if num_args > len(args): raise ValueError('Insufficient number of positional parameters for API %s' % MonaInitBioLinkAPI.BASE_URL + self.func_url)
		kw_args.update((k, v) for k, v in kwargs.items() if k in kw_args)
		trail = 0
		while max_trail <= 0 or trail < max_trail:
			try:
				res = APIClient.call(self, self.func_url.format(*args[:num_args]), **kw_args)
				break
			except Exception as e:
				print(e)
				time.sleep(interval)
				trail += 1
		return res


if __name__ == '__main__':
	text = 'Melanoma is a malignant tumor of melanocytes which are found predominantly in skin but also in the bowel and the eye.'
	print([a['id'] for a in annotext(text, ontos=['HP'], includeAcronym='true', includeCat='phenotype')])
	print(phenocases(['HP:0011842'], ontos=['HP']))
	print(phenodzs(['HP:0011842'], ontos=['HP']))
	print(phenopubs(['HP:0011842'], ontos=['HP']))
	print(pubphenos(['11818962'], ontos=['HP']))
