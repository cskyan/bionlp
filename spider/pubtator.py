#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: pubtator.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2019-01-18 19:34:28
###########################################################################
#

import os, sys, copy, json, time, requests, hashlib, pickle, urllib

import numpy as np

from .. import nlp
from ..util import fs, io, func


if sys.platform.startswith('win32'):
	DATA_PATH = 'C:\\data\\bionlp'
elif sys.platform.startswith('linux2'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
PUBTATOR_PATH = os.path.join(DATA_PATH, 'pubtator')

SC='\n'


class PubTatorAPI():
	BASE_URL = 'https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/'
	_type_url = {'chemical':'Chemical', 'disease':'Disease', 'gene':'Gene', 'mutation':'Mutation', 'species':'Species', 'all':'BioConcept'}
	_type_trg = {'chemical':'tmChem', 'disease':'DNorm', 'gene':'GNormPlus', 'mutation':'tmVar', 'species':'GNormPlus'}
	_format_ext = {'bioc':'xml', 'pubtator':'txt', 'json':'json'}
	def __init__(self, cache_dir=PUBTATOR_PATH):
		self.format = None
		self.cache_dir = cache_dir

	def _check_type(self, type):
		if not (type in PubTatorAPI._type_url or type in PubTatorAPI._type_trg):
			raise ValueError('The type %s is not supported!' % type)

	def _handle_response(self, response):
		if (self.format == 'JSON'):
			res = json.loads(nlp.clean_text(response.data, encoding='utf-8', replacement=None).replace('\\', ''))
		else:
			res = response.data
		self.format = None
		return res

	def get_concepts_pmid(self, type, pmid, fmt='JSON'):
		type = type.lower()
		# if (type == 'all'):
			# res_list = [self.get_concepts_pmid(type=tp, pmid=pmid, fmt='JSON') for tp in PubTatorAPI._type_url.keys()]
			# return [func.update_dict(copy.deepcopy(prgrph[0]), {u'denotations':func.flatten_list([res['denotations'] for res in prgrph])}) for prgrph in zip(*res_list)]
		self.format = fmt if fmt in ('PubTator', 'BioC', 'JSON') else None
		self._check_type(type)
		fs.mkdir(os.path.join(self.cache_dir, type))
		cache_path = os.path.join(self.cache_dir, type, '%s.%s' % (pmid, PubTatorAPI._format_ext[fmt.lower()]))
		if (os.path.exists(cache_path)):
			if fmt == 'JSON':
				res = io.read_json(cache_path)['data']
			else:
				res = '\n'.join(fs.read_file(cache_path, code='utf-8'))
			return res
		else:
			res = requests.get(url=urllib.parse.urljoin(PubTatorAPI.BASE_URL, '%s/%s/%s' % (PubTatorAPI._type_url[type], pmid, fmt)))
			if fmt == 'JSON':
				io.write_json(res.json() if res.text else [], cache_path, code='utf-8')
			else:
				fs.write_file(res.text, cache_path, code='utf-8')
			return res.json() if fmt=='JSON' else res.text

	def get_concepts_rawtxt(self, ctype, text, sleep_time=10, timeout=600):
		ctype = ctype.lower()
		if (ctype == 'all'):
			res_list = [self.get_concepts_rawtxt(ctype=tp, text=text, sleep_time=sleep_time, timeout=timeout) for tp in PubTatorAPI._type_trg.keys()]
			denotations = func.flatten_list([res['denotations'] for res in res_list])
			return func.update_dict(copy.deepcopy(res_list[0]), {u'denotations':denotations})
		self._check_type(ctype)
		fs.mkdir(os.path.join(self.cache_dir, ctype))
		cache_path = os.path.join(self.cache_dir, ctype, 'rawtxt.pkl')
		txt_md5, cache = hashlib.md5(text.encode('utf-8')).hexdigest(), {}
		if os.path.isfile(cache_path):
			try:
				with open(cache_path, 'rb') as fd:
					cache.update(pickle.load(fd))
					if txt_md5 in cache and cache[txt_md5][0] == text: return cache[txt_md5][1]
			except Exception as e:
				print(e)
		data = '{"sourcedb":"PubMed","sourceid":"1000001","text":"%s"}' % text
		sess_id = requests.post(url=urllib.parse.urljoin(PubTatorAPI.BASE_URL, '%s/Submit' % PubTatorAPI._type_trg[ctype]), data=data).text
		wait_time, res = 0, {}
		while (not res and wait_time < timeout):
			res = requests.get(url=urllib.parse.urljoin(PubTatorAPI.BASE_URL, '%s/Receive' % sess_id))
			res = {} if not res.ok or 'The Result is not ready' in res.text else res
			time.sleep(sleep_time)
			wait_time += sleep_time
		try:
			out = {} if not res.text or res.text.isspace() else res.json()
		except Exception as e:
			print(e)
			out = res
		with open(cache_path, 'wb') as fd:
			if type(out)==dict: cache[txt_md5] = (text, out)
			pickle.dump(cache, fd)
		return out
		# return {} if not res.text or res.text.isspace() else res.json()

	def get_concepts_rawtxtlist(self, ctype, text_list, sleep_time=10, timeout=600):
		text = SC.join(text_list)
		all_res = self.get_concepts_rawtxt(ctype=ctype, text=text, sleep_time=sleep_time, timeout=timeout)
		if (not all_res): return all_res
		sc_chnum = len(SC)
		txt_len = [0] + [len(txt) + sc_chnum for txt in text_list]
		txt_bndr = np.cumsum(txt_len)
		dntn_list = [[] for x in range(len(text_list))]
		for concept in all_res['denotations']:
			idx = np.searchsorted(txt_bndr, concept['span']['begin'], side='left')
			concept.update({u'span':{u'begin':concept['span']['begin'] - txt_bndr[idx], u'end':concept['span']['end'] - txt_bndr[idx]}})
			dntn_list[idx].append(concept)
		return [func.update_dict(copy.deepcopy(all_res[0]), {u'denotations':dntn, u'text':txt}) for dntn, txt in zip(dntn_list, text_list)]
