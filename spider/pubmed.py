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

import os
import sys
import urllib
import urllib2
import codecs
import xmlextrc

from ..util import fs

if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux2'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
ABS_PATH = os.path.join(DATA_PATH, 'abstracts')
BASE_URL = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
SC=';;'


class PubMedBuilder():
	def __init__(self):
		self._tag = ''
		self._tag_stack = []
		self.id = ''
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
		# Process the attributes
	
	def end(self, tag):
		self._tag_stack.pop()
		
	def data(self, data):
		if data.isspace():
			return
		data = data.strip()
		# Process the text content
		if (self._tag == 'PMID'):
			self.id = data
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
				self.mesh_headings.append(data)
		if (self._tag == 'NameOfSubstance'):
			if (self._tag_stack[-2] == 'Chemical'):
				self.chemicals.append(data)
		
	def close(self):
		pass

	def build(self):
		return {'title':self.title, 'abs':'\n'.join(self.abs_list).strip(), 'kws':SC.join(self.keywords), 'mesh':self.mesh_headings, 'chem':self.chemicals}


def get_pmids(ss, max_num=1000):
	url = BASE_URL + "esearch.fcgi?db=pubmed&term=" + urllib.quote(ss) + "&rettype=abstract&retmode=text&usehistory=y&retmax=%i" % (max_num)
	print "Search String:\n" + url + '\n'
	res = urllib2.urlopen(url).read()

	pmid_list = xmlextrc.extrc_list('IdList', 'Id', '.', None, res, 'text')
	print "Number of obtained pmids: %i\n" % len(pmid_list)
	return pmid_list

	
def fetch_abs(pmid_list, saved_path=ABS_PATH):
	fs.mkdir(saved_path)
	url = BASE_URL + "efetch.fcgi?db=pubmed&retmode=xml&id=" + ','.join(pmid_list)
	print "Fetch String:\n" + url + '\n'
	res = urllib2.urlopen(url).read()
	
	abs_tree = xmlextrc.extrc_list('PubmedArticleSet', 'PubmedArticle', './MedlineCitation/Article/Abstract', None, res, 'elem')
	abs_text = []
	for abs in abs_tree:
		contents = xmlextrc.extrc_list('Abstract', 'AbstractText', '.', abs, '', 'text')
		abs_text.append(' \n'.join(contents))
	print "Number of obtained abstracts: %i\n" % len(abs_text)
	if saved_path is not None:
		for id, abs in zip(pmid_list, abs_text):
			fs.write_file(abs, os.path.join(ABS_PATH, str(id)+'.txt'), code='utf8')
	return abs_text


def fetch_artcls(pmid_list, saved_path=DATA_PATH):
	url = BASE_URL + "efetch.fcgi?db=pubmed&retmode=xml&id=" + ','.join(pmid_list)
	print "Fetch String:\n" + url + '\n'
	res = urllib2.urlopen(url).read()
	
	artcl_xml = xmlextrc.extrc_list('PubmedArticleSet', 'PubmedArticle', '.', None, res, 'xml')
	articles = []
	for artcl in artcl_xml:
		builder = PubMedBuilder()
		parser = xmlextrc.get_parser(builder)
		parser.feed(artcl)
		articles.append(builder.build())
	print "Number of obtained articles: %i\n" % len(artcl_xml)
	if saved_path is not None:
		feat_attr = {'abs':['abs', 'txt'], 'kws':['kws', 'txt'], 'mesh':['mesh', 'mesh'], 'chem':['chem', 'chem']}
		for f, a in feat_attr.iteritems():
			fs.mkdir(os.path.join(saved_path, a[0]))
		for id, artcls in zip(pmid_list, articles):
			for f, a in feat_attr.iteritems():
				if (hasattr(artcls[f], '__iter__')):
					attr_text = '\n'.join(artcls[f])
				else:
					attr_text = artcls[f]
				fs.write_file(attr_text, os.path.join(saved_path, a[0], str(id)+'.'+a[1]), code='utf8')
	return articles