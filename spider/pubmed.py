#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: pubmed.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-03-01 21:04:40
###########################################################################
#

import os, sys, time, codecs, urllib, itertools

# from ..util import fs
# from . import xmlextrc
from bionlp.util import fs, njobs
from bionlp.spider import xmlextrc

if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ABS_PATH = os.path.join(DATA_PATH, 'abstracts')
BASE_URL = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
SC=';;'


class PubMedBuilder():
	def __init__(self):
		self._tag = ''
		self._tag_stack = []
		self.pmid = ''
		self.journal = ''
		self.year = ''
		self.title = ''
		self.authors = []
		self.abs_list = []
		self.keywords = []
		self.mesh_headings = []
		self.chemicals = []

	def start(self, tag, attrib):
		self._tag = tag
		self._tag_stack.append(self._tag)
		self._tag_attrib = attrib
		# Process the attributes

	def end(self, tag):
		self._tag_stack.pop()

	def data(self, data):
		if data.isspace():
			return
		data = data.strip()
		# Process the text content
		if (self._tag == 'PMID'):
			self.pmid = data
		if (self._tag == 'Title' and self._tag_stack[-2] == 'Journal'):
			self.journal = data
		if (self._tag == 'Year' and self._tag_stack[-2] == 'PubDate'):
			self.year = data
		if (self._tag == 'ArticleTitle'):
			self.title = data
		if (self._tag == 'AbstractText'):
			self.abs_list.append(data)
			#print codecs.encode(data, 'utf8')
		if (self._tag == 'Keyword'):
			self.keywords.append(data)
		if (self._tag == 'DescriptorName'):
			if (self._tag_stack[-2] == 'MeshHeading'):
				self.mesh_headings.append((self._tag_attrib['UI'], data))
		if (self._tag == 'NameOfSubstance'):
			if (self._tag_stack[-2] == 'Chemical'):
				self.chemicals.append((self._tag_attrib['UI'], data))

	def close(self):
		pass

	def build(self):
		return {'pmid':self.pmid, 'title':self.title, 'abs':'\n'.join(self.abs_list).strip(), 'kws':SC.join(self.keywords), 'mesh':self.mesh_headings, 'chem':self.chemicals}


class PubMedInfoBuilder():
	def __init__(self):
		self._tag = ''
		self._tag_stack = []
		self.name = None
		self.number = None
		self.last_update = None
		self.properties = []
		self.links = []

	def start(self, tag, attrib):
		self._tag = tag
		self._tag_stack.append(self._tag)
		self._tag_attrib = attrib
		# Process the attributes

	def end(self, tag):
		self._tag_stack.pop()

	def data(self, data):
		if data.isspace():
			return
		data = data.strip()
		# Process the text content
		if (self._tag == 'MenuName' and self._tag_stack[-2] == 'DbInfo'):
			self.name = data
		if (self._tag == 'Count' and self._tag_stack[-2] == 'DbInfo'):
			self.number = int(data)
		if (self._tag == 'LastUpdate' and self._tag_stack[-2] == 'DbInfo'):
			self.last_update = data

	def close(self):
		pass

	def build(self):
		return {'name':self.name, 'number':self.number, 'last_update':self.last_update}


def get_pubmed_info():
	import datetime, json
	now = datetime.datetime.now()
	url = BASE_URL + "einfo.fcgi?db=pubmed"
	res = urllib.request.urlopen(url).read()
	builder = PubMedInfoBuilder()
	parser = xmlextrc.get_parser(builder)
	parser.feed(res)
	res = builder.build()
	with open('pubmed_info_%d%d%d.json' % (now.year, now.month, now.day), 'w') as fd:
		json.dump(res, fd)
	return res


def get_pmids(ss, max_num=1000):
	url = BASE_URL + "esearch.fcgi?db=pubmed&term=" + urllib.parse.quote(ss) + "&rettype=abstract&retmode=text&usehistory=y&retmax=%i" % (max_num)
	print("Search String:\n" + url + '\n')
	trail = 0
	while max_trail <= 0 or trail < max_trail:
		try:
			res = urllib.request.urlopen(url).read()
			break
		except Exception as e:
			print(e)
			time.sleep(1)
			trail += 1

	pmid_list = xmlextrc.extrc_list('IdList', 'Id', '.', None, res, 'text')
	print("Number of obtained pmids: %i\n" % len(pmid_list))
	return pmid_list


def fetch_abs(pmid_list, saved_path=ABS_PATH):
	fs.mkdir(saved_path)
	url = BASE_URL + "efetch.fcgi?db=pubmed&retmode=xml&id=" + ','.join(pmid_list)
	print("Fetch String:\n" + url + '\n')
	trail = 0
	while max_trail <= 0 or trail < max_trail:
		try:
			res = urllib.request.urlopen(url).read()
			break
		except Exception as e:
			print(e)
			time.sleep(1)
			trail += 1

	abs_tree = xmlextrc.extrc_list('PubmedArticleSet', 'PubmedArticle', './MedlineCitation/Article/Abstract', None, res, 'elem')
	abs_text = []
	for abs in abs_tree:
		contents = xmlextrc.extrc_list('Abstract', 'AbstractText', '.', abs, '', 'text')
		abs_text.append(' \n'.join(contents))
	print("Number of obtained abstracts: %i\n" % len(abs_text))
	if saved_path is not None:
		for id, abs in zip(pmid_list, abs_text):
			fs.write_file(abs, os.path.join(ABS_PATH, str(id)+'.txt'), code='utf8')
	return abs_text


def _save_feat(articles, feat_attr, saved_path=DATA_PATH):
	for artcl in articles:
		for f, a in feat_attr.items():
			if (hasattr(artcl[f], '__iter__')):
				attr_text = '\n'.join([':'.join(x) for x in artcl[f]] if artcl[f] and type(artcl[f][0]) is tuple else artcl[f])
			else:
				attr_text = artcl[f]
			fs.write_file(attr_text, os.path.join(saved_path, a[0], artcl['pmid']+'.'+a[1]), code='utf8')
		print('Processed PMID:%s' % artcl['pmid'])
		sys.stdout.flush()


def process_artcls(xml_str, saved_path=DATA_PATH, n_jobs=1):
	artcl_xml = xmlextrc.extrc_list('PubmedArticleSet', 'PubmedArticle', '.', None, xml_str, 'xml')
	articles = []
	for artcl in artcl_xml:
		builder = PubMedBuilder()
		parser = xmlextrc.get_parser(builder)
		parser.feed(artcl)
		articles.append(builder.build())
	print("Number of obtained articles: %i\n" % len(artcl_xml))
	if saved_path is not None:
		feat_attr = {'title':['title', 'txt'], 'abs':['abs', 'txt'], 'kws':['kws', 'txt'], 'mesh':['mesh', 'mesh'], 'chem':['chem', 'chem']}
		for f, a in feat_attr.items():
			fs.mkdir(os.path.join(saved_path, a[0]))
		if n_jobs == 1:
			_save_feat(articles, feat_attr, saved_path=saved_path)
		else:
			task_bnd = njobs.split_1d(len(articles), split_num=n_jobs, ret_idx=True)
			_ = njobs.run_pool(_save_feat, n_jobs=n_jobs, dist_param=['articles'], articles=[articles[task_bnd[i]:task_bnd[i+1]] for i in range(n_jobs)], feat_attr=feat_attr, saved_path=saved_path)
	return articles


def fetch_artcls(pmid_list, saved_path=DATA_PATH, max_trail=-1):
	url = BASE_URL + "efetch.fcgi?db=pubmed&retmode=xml&id=" + ','.join(pmid_list)
	print("Fetch String:\n" + url + '\n')
	trail = 0
	while max_trail <= 0 or trail < max_trail:
		try:
			res = urllib.request.urlopen(url).read()
			break
		except Exception as e:
			print(e)
			time.sleep(1)
			trail += 1
	return process_artcls(res, saved_path=saved_path)


def _fetch_archive(archive_files, saved_path=DATA_PATH, write_n_jobs=1):
	import builtins, gzip
	articles, open, fmode, open_kwargs = [], getattr(builtins, 'open'), 'r', {'encoding':'utf-8'}
	for fpath in archive_files:
		sys.stdout.flush()
		if fpath.endswith('.gz'): open, fmode, open_kwargs = gzip.open, 'rb', {}
		with open(fpath, fmode, **open_kwargs) as fd:
			res = fd.read()
		articles.extend(process_artcls(res, saved_path=saved_path, n_jobs=write_n_jobs))
		print("Processed archive file %s" % os.path.basename(fpath))
		sys.stdout.flush()
	return articles


def fetch_archive(archive_paths, saved_path=DATA_PATH, n_jobs=(1,1)):
	archive_paths, articles = archive_paths if type(archive_paths) is list else archive_paths, []
	all_archive_files = list(itertools.chain.from_iterable([fs.listf(archive_path, full_path=True) for archive_path in archive_paths]))
	n_jobs, write_n_jobs = n_jobs if hasattr(n_jobs, '__iter__') else (n_jobs, 1)
	if n_jobs == 1:
		return _fetch_archive(all_archive_files, saved_path=saved_path)
	else:
		task_bnd = njobs.split_1d(len(all_archive_files), split_num=n_jobs, ret_idx=True)
		articles_grps = njobs.run_pool(_fetch_archive, n_jobs=n_jobs, dist_param=['archive_files'], archive_files=[all_archive_files[task_bnd[i]:task_bnd[i+1]] for i in range(n_jobs)])
		return list(itertools.chain.from_iterable(articles_grps))


if __name__ == '__main__':
	print(get_pubmed_info())
	print(fetch_artcls(['205546'], saved_path=None))
