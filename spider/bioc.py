#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2019 by Caspar. All rights reserved.
# File Name: bioc.py
# Author: Shankai Yan
# E-mail: shankai.yan@nih.gov
# Created Time: 2020-02-25 17:17:35
###########################################################################
#

import os, re, sys, copy, time, json, string
from contextlib import ExitStack

from nltk.tokenize import sent_tokenize, TreebankWordTokenizer
from apiclient import APIClient
import pysolr
import spacy
try:
    nlp = spacy.load('en_core_sci_md')
except Exception as e:
    print(e)
    try:
        nlp = spacy.load('en_core_sci_sm')
    except Exception as e:
        print(e)
        nlp = spacy.load('en_core_web_sm')


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
SC=';;'


def preprocess(text, tokenize=False):
    text = text.lstrip('nullnullnull').strip()
    if len(text) > 2:
        text = re.sub(r'[\t\r\n]+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        if tokenize:
            tokenized = TreebankWordTokenizer().tokenize(text)
            text = ' '.join(tokenized)
        # to address single quotation mark issue
        text = re.sub(r"^'(\w)", r"' \1", text)
        text = re.sub(r"\s'(\w)", r" ' \1", text)
        # to address 's issue
        text = re.sub(r"\s's\b", "'s", text)
        return text
    else:
        return None


def fetch_artcls(all_pmids, out_fpaths=None, solr_url=None, pid=0, cache=False, abs_only=False, retdoc=False):
	client = BioCAPI(function='pubmed') if abs_only else BioCAPI(function='pmc')
	# res = [client.call(id=pmid, encoding='ascii') for pmid in all_pmids]
	# res = [dict(id=doc['documents'][0]['id'], text=' \n '.join([span['text'] for span in doc['documents'][0]['passages']])) for doc in res]
	solr = pysolr.Solr(solr_url, always_commit=True, results_cls=dict) if solr_url else None
	res = []
	with ExitStack() as stack:
		if out_fpaths is not None:
			try:
				outfile_pmid, outfile0, outfile1, outfile2, outfile3 = out_fpaths
				fout = stack.enter_context(open(outfile_pmid, 'a+', encoding='utf-8'))
				# fout_list = [stack.enter_context(open(f, 'a', encoding='utf-8')) if f else None for f in [outfile0, outfile1, outfile2, outfile3]]
				fout_list = [stack.enter_context(open(f, 'a', encoding='utf-8')) if f else None for f in [None, None, outfile2, outfile3]]
				pmids = set([int(pmid.rstrip('\n')) for pmid in fout.readlines()]) if cache else set([])
				if solr:
					pmid_counts = solr.search('', **{'facet':'on', 'facet.field':'id', 'facet.limit':-1})['facet_counts']['facet_fields']['id']
					pmids.update([pmid_counts[i] for i in range(0, len(pmid_counts), 2)])
			except Exception as e:
				print(e)
				out_fpaths = None
		for i, pmid in enumerate(all_pmids):
			if not retdoc and (pmid in pmids or (out_fpaths is None and solr is None)): continue
			doc = client.call(id=pmid, encoding='ascii')
			if len(doc) == 0: continue
			# pmc_id, text = doc['documents'][0]['id'], preprocess(' '.join([span['text'] if span['text'] and span['text'][-1] in string.punctuation else '%s.' % span['text'] for span in doc['documents'][0]['passages']]))
			pmc_id, text = doc['documents'][0]['id'], [span['text'] for span in doc['documents'][0]['passages']]
			# if not text or text.isspace(): continue
			saved = False
			if solr:
				try:
					if solr.search('id:%s' % pmid)['response']['numFound'] == 0:
						print('Uploading PMID:%s to Solr...' % pmid)
						solr.add([{'id':pmid, 'pmcid':pmc_id, 'fulltext':text}])
					else:
						print('PMID:%s exists on Solr...' % pmid)
					saved = True
				except Exception as e:
					print(e)
			if not saved and out_fpaths:
				fout_list[:2] = [stack.enter_context(open((lambda x: '%s_%s%s'%(x[0], pmid, x[1]))(os.path.splitext(f)), 'a', encoding='utf-8')) if f else None for f in [outfile0, outfile1]]
				text = '\n'.join(text)
				if outfile0: fout_list[0].write(text + '\n')
				if outfile1: fout_list[1].write(text.lower() + '\n')
				if outfile2 or outfile3:
					for sent in nlp(text).sents:
						if not sent.text or sent.text.isspace(): continue
						if outfile2: fout_list[2].write(sent.text + '\n')
						if outfile3: fout_list[3].write(sent.text.lower() + '\n')
				saved = True
			if saved:
				fout.write(str(pmid) + '\n')
			if i and i % 10000 == 0:
			   print('Retrieved %i documents ' % i)
			   sys.stdout.flush()
			if retdoc: res.append(dict(id=pmid, pmcid=pmc_id, text=text))
	return res if retdoc else None


class BioCAPI(APIClient, object):
	BASE_URL = 'https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful'
	_function_url = {'pubmed':'pubmed.cgi', 'pmc':'pmcoa.cgi'}
	_default_param = {'pubmed':dict(id='', encoding='ascii'), 'pmc':dict(id='', encoding='ascii')}
	_func_restype = {'pubmed':'json', 'pmc':'json'}

	def __init__(self, function='annotate'):
		if (function not in self._default_param):
			raise ValueError('The function %s is not supported!' % function)
		APIClient.__init__(self)
		self.function = function
		self.func_url = self._function_url[function]
		self.restype = self._func_restype.setdefault(function, 'json')
		self.runtime_args = {}

	def _handle_response(self, response):
		if (self.restype == 'xml'):
			pass
		elif (self.restype == 'json'):
			res = response.data
			try:
				res_json = json.loads(res)
			except Exception as e:
				if type(res) is bytes and 'The PMC article is not available in open access subset' in res.decode('utf-8', errors='replace'):
					print('The PMC article %s is not available in open access subset!' % self.runtime_args['id'])
				else:
					print(e)
					print(res)
				res_json = {}
			return res_json

	def call(self, max_trail=-1, interval=3, **kwargs):
		args = copy.deepcopy(self._default_param[self.function])
		args.update((k, v) for k, v in kwargs.items() if k in args)
		self.runtime_args = args
		trail = 0
		while max_trail <= 0 or trail < max_trail:
			try:
				res = APIClient.call(self, '/%s/BioC_%s/%s/%s' % (self.func_url, self.restype, kwargs['id'], kwargs['encoding']))
				break
			except Exception as e:
				print(e)
				time.sleep(interval)
				trail += 1
		return res

if __name__ == '__main__':
	print([doc for doc in fetch_artcls(['29378527'], out_fpaths=['pmid.txt', 'doc.txt', None, None, None], retdoc=True)])
