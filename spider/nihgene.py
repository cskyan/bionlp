#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: nihgene.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2018-01-31 09:02:48
###########################################################################
#

import os, re, sys, types, string, codecs, urllib
from io import StringIO

from .. import nlp
from ..util import fs, func, njobs
from . import xmlextrc

if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
GENE_PATH = os.path.join(DATA_PATH, 'nihgene')
BASE_URL = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
SC=';;'


class NIHGene():
#	__metaclass__ = ABCMeta
	def __init__(self):
		self._tag = ''
		self._tag_stack = []
		self._obj = {}
		self.id = ''
		self.symbol = ''
		self.fullname = ''
		self.type = ''
		self.status = ''
		self.organism = ''
		self.locs = []
		self.sources = []
		self.lineage = []
		self.synonyms = []

		self.enter_commprop = False
		self.enter_comm = False

		self.enter_prop = False
		self.enter_src = False


		self.enter_gene = False
		self.enter_generefdb = False
		self.enter_db = False

		self.enter_comment = False
		self.enter_org = False

		self.enter_loc = False
		self.enter_map = False

		self.enter_syn = False

	def start(self, tag, attrib):
		self._tag = tag.split('}')[1] if '}' in tag else tag # Remove the namespace
		self._tag_stack.append(self._tag)
		# Process the attributes
		if (self._tag == 'Entrezgene_type'):
			self.type = attrib['value']


		if (self._tag == 'Gene-commentary_properties'):
			self.enter_commprop = True
		if (self._tag == 'Gene-commentary'):
			self.enter_comm = True

		if (self._tag == 'Entrezgene_properties'):
			self.enter_prop = True
		if (self._tag == 'Entrezgene_source'):
			self.enter_src = True

		if (self._tag == 'Entrezgene_gene'):
			self.enter_gene = True
		if (self._tag == 'Gene-ref_db'):
			self.enter_generefdb = True
		if (self._tag == 'Dbtag'):
			self.enter_db = True

		if (self._tag == 'Entrezgene_comments'):
			self.enter_comment = True
		if (self._tag == 'BioSource_org'):
			self.enter_org = True

		if (self._tag == 'Entrezgene_location'):
			self.enter_loc = True
		if (self._tag == 'Maps'):
			self.enter_map = True
		if (self._tag == 'Maps_method_map-type'):
			self._obj['mtype'] = attrib['value']

		if (self._tag == 'Gene-ref_syn'):
			self.enter_syn = True

	def end(self, tag):
		self._tag = tag.split('}')[1] if '}' in tag else tag # Remove the namespace
		self._tag_stack.pop()

		if (self._tag == 'Gene-commentary_properties'):
			self.enter_commprop = False
		if (self._tag == 'Gene-commentary'):
			self.enter_comm = False

		if (self._tag == 'Entrezgene_properties'):
			self.enter_prop = False
		if (self._tag == 'Entrezgene_source'):
			self.enter_src = False

		if (self._tag == 'Entrezgene_gene'):
			self.enter_gene = False
		if (self._tag == 'Gene-ref_db'):
			self.enter_generefdb = False
		if (self._tag == 'Dbtag'):
			self.enter_db = False

		if (self._tag == 'Entrezgene_comments'):
			self.enter_comment = False
		if (self._tag == 'BioSource_org'):
			self.enter_org = False

		if (self._tag == 'Entrezgene_location'):
			self.enter_loc = False
		if (self._tag == 'Maps'):
			self.enter_map = False
			self.locs.append(':'.join([self._obj['mtype'], self._obj['mstr']]))

		if (self._tag == 'Gene-ref_syn'):
			self.enter_syn = False

	def data(self, data):
		if data.isspace(): return
		data = data.strip()
		# Process the text content
		if (self._tag == 'Gene-track_geneid'):
			self.id = data

		if (self._tag == 'Gene-commentary_type' and self.enter_comm):
			self._obj['type'] = data
		if (self._tag == 'Gene-commentary_heading' and self.enter_comm):
			self._obj['heading'] = data
		if (self._tag == 'Gene-commentary_label' and self.enter_comm):
			self._obj['label'] = data
			if (self._obj.setdefault('heading', '') == 'RefSeq Status'):
				self.status = data
		if (self._tag == 'Gene-commentary_text' and self.enter_comm):
			if (self._obj.setdefault('label', '') == 'Official Symbol' and self.enter_commprop and self.enter_prop):
				self.symbol = data
			elif (self._obj.setdefault('label', '') == 'Official Full Name' and self.enter_commprop and self.enter_prop):
				self.fullname = data

		if (self._tag == 'Dbtag_db' and self.enter_db and self.enter_generefdb and self.enter_gene):
			self._obj['db'] = data
		if (self._tag == 'Object-id_str' and self.enter_db and self.enter_generefdb and self.enter_gene):
			self.sources.append(':'.join([self._obj['db'], data]))

		if (self._tag == 'Org-ref_taxname' and self.enter_org and self.enter_src):
			self.organism = data
		if (self._tag == 'OrgName_lineage' and self.enter_org and self.enter_src):
			self.lineage = list(map(string.strip, data.strip(' ;').split(';')))

		if (self._tag == 'Maps_display-str' and self.enter_map and self.enter_loc):
			self._obj['mstr'] = data

		if (self._tag == 'Gene-ref_syn_E' and self.enter_syn and self.enter_gene):
			self.synonyms.append(data)

	def close(self):
		pass

	def build(self):
		return {'id':self.id, 'symbol':self.symbol, 'fullname':self.fullname, 'type':self.type, 'status':self.status, 'organism':self.organism, 'locs':';'.join(self.locs), 'sources':';'.join(self.sources), 'lineage':';'.join(self.lineage), 'synonyms':';'.join(self.synonyms)}


def fetch_gene(ids, fmt='xml', buff_size=1, saved_path=GENE_PATH, ret_strio=False):
	fmt2ext = {'text':'txt', 'xml':'xml', 'html':'html'}
	fs.mkdir(saved_path)
	count, results = 0, {}
	page_idx = njobs.split_1d(len(ids), task_size=1, split_size=buff_size, ret_idx=True)
	for i in range(len(page_idx) - 1):
		sub_ids, existed_ids = ids[page_idx[i]:page_idx[i+1]], []
		for j, gene_id in enumerate(sub_ids):
			# Nested gene ids
			if (type(gene_id) is list):
				nested_genes = list(fetch_gene(gene_id, fmt=fmt, buff_size=buff_size, saved_path=saved_path, ret_strio=False))
				gene_id = sub_ids[j] = SC.join(func.flatten_list(gene_id))
				results[gene_id] = nested_genes
			# Determine whether the gene id duplicate or invalid
			if (gene_id in results or not gene_id or gene_id.isspace() or gene_id=='0' or gene_id=='0.0'):
				existed_ids.append(gene_id)
				continue
			# Read cache
			tgt_file = os.path.join(saved_path, '%s.%s' % (gene_id, fmt2ext[fmt]))
			if os.path.exists(tgt_file):
				results[gene_id] = '\n'.join(fs.read_file(tgt_file))
				existed_ids.append(gene_id)
		# Query the remaining gene
		query_ids = [x for x in sub_ids if x not in existed_ids]
		if (len(query_ids) > 0):
			id_str = ','.join(query_ids)
			url = BASE_URL + "efetch.fcgi?db=gene&id=" + urllib.parse.quote(id_str) + "&retmode=xml"
			try:
				res = urllib.request.urlopen(url).read()
			except:
				print('Failed to fetch the genes: %s in NIH GENE database!' % id_str)
			else:
				count += len(query_ids)
				for gene_id, gene_res in zip(query_ids, xmlextrc.extrc_list('Entrezgene-Set', 'Entrezgene', '.', None, res, 'xml')):
					results[gene_id] = gene_res
					tgt_file = os.path.join(saved_path, gene_id + '.%s' % fmt2ext[fmt])
					if (saved_path): fs.write_file(gene_res, tgt_file)
		for gene_id in sub_ids:
			if (type(results.setdefault(gene_id, '')) is list):
				yield [StringIO(x) for x in results[gene_id]] if ret_strio else results[gene_id]
			else:
				yield StringIO(results[gene_id]) if ret_strio else results[gene_id]
	# if (count > 0):
		# print "Number of newly downloaded NIHGENE documents: %i\n" % count


def parse_gene(gene_fpath, fmt='xml'):
	gene_str = '\n'.join(fs.read_file(gene_fpath))
	if (not gene_str or gene_str.isspace()): return {}
	builder = NIHGene()
	parser = xmlextrc.get_parser(builder)
	try:
		parser.feed(nlp.clean_text(re.sub(u'[\x00-\x08\x0b-\x0c\x0e-\x1f]+', u'', gene_str), encoding='utf-8', replacement=None))
	except Exception as err:
		print('Can not parse the file: %s' % gene_fpath)
		raise err
	parser.close()
	return builder.build()


def parse_genes(gene_fpaths, fmt='xml'):
	for i, gene_fpath in enumerate(gene_fpaths):
		# Nested gene file paths
		if (type(gene_fpath) is list):
			yield list(parse_genes(gene_fpath, fmt=fmt))
			continue
		gene_str = '\n'.join(fs.read_file(gene_fpath))
		if (not gene_str or gene_str.isspace()):
			yield {}
			continue
		builder = NIHGene()
		parser = xmlextrc.get_parser(builder)
		try:
			parser.feed(nlp.clean_text(re.sub(u'[\x00-\x08\x0b-\x0c\x0e-\x1f]+', u'', gene_str), encoding='utf-8', replacement=None))
		except Exception as err:
			print('Can not parse the file: %s' % ('No.%i'%i if isinstance(gene_fpaths, types.GeneratorType) else gene_fpaths[i]))
			raise err
		parser.close()
		yield builder.build()
