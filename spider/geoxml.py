#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: geoxml.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-10-14 21:04:40
###########################################################################
#

import os, re, sys, codecs, urllib
from . import xmlextrc

from io import StringIO
from abc import ABCMeta

import pandas as pd

from .. import nlp
from ..util import fs


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\bionlp'
elif sys.platform.startswith('linux2'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'bionlp')
GEO_PATH = os.path.join(DATA_PATH, 'geo')


class GSEBuilder():
#	__metaclass__ = ABCMeta

	def __init__(self, view='brief'):
		self.view = view
		self._tag = ''
		self._tag_stack = []
		self.db = ''
		self.submit_date = ''
		self.release_date = ''
		self.last_update = ''
		self.samples = []

		self.ftype = 'SERIES'	# GEO type [SAMPLE | SERIES | PLATFORM | DATASET]
		self.id = ''	# GEO Accession
		self.pmid = ''
		self.title = ''
		self.summary = ''

	def start(self, tag, attrib):
		self._tag = tag.split('}')[1] if '}' in tag else tag
		self._tag_stack.append(self._tag)
		# Process the attributes
		if (self._tag == 'Database'):
			self.db = attrib['iid']
		if (self._tag == 'Series'):
			self.id = attrib['iid']
		if (self._tag == 'Sample'):
			self.samples.append(attrib['iid'])

	def end(self, tag):
		self._tag_stack.pop()

	def data(self, data):
		if data.isspace():
			return
		data = data.strip()
		# Process the text content
		if (self._tag == 'Submission-Date'):
			self.submit_date = data
		if (self._tag == 'Release-Date'):
			self.release_date = data
		if (self._tag == 'Last-Update-Date'):
			self.last_update = data

		if (self._tag == 'Pubmed-ID'):
			self.pmid = data
		if (self._tag == 'Title'):
			self.title = data
		if (self._tag == 'Summary'):
			self.keywords = []
			paragraphs = data.split('\n')
			i = len(paragraphs) - 1
			for i in range(len(paragraphs) - 1, 0, -1):
				if (paragraphs[i].startswith('Keywords')):
					phrases = paragraphs[i].split(':')
					if (len(phrases) == 1):
						phrases = paragraphs[i].split('=')
					self.keywords.append(phrases[1].strip())
				else:
					break
			self.keywords.reverse()
			self.summary = '\n'.join(paragraphs[:i+1])

	def close(self):
		pass

	def build(self):
		return {'db':self.db, 'submit_date':self.submit_date, 'release_date':self.release_date, 'last_update':self.last_update, 'samples':self.samples, 'id':self.id, 'ftype':self.ftype, 'pmid':self.pmid, 'title':self.title, 'summary':self.summary, 'keywords':';'.join(self.keywords)}


class GSMBuilder():
#	__metaclass__ = ABCMeta

	def __init__(self, view='brief'):
		self.view = view
		self._tag = ''
		self._tag_stack = []
		self.platform = ''
		self.db = ''
		self.submit_date = ''
		self.release_date = ''
		self.last_update = ''

		self.ftype = 'SAMPLE'	# GEO type [SAMPLE | SERIES | PLATFORM | DATASET]
		self.id = ''	# GEO Accession
		self.title = ''
		self.description = ''
		self.data_processing = ''
		self.type = ''
		self.source = ''
		self.organism = ''
		self.molecule = ''
		self.tissue = ''
		self.tissue_type = ''
		self.treat_protocol = ''
		self.growth_protocol = ''
		self.extract_protocol = ''
		self.label_protocol = ''
		self.label = ''
		self.hybrid_protocol = ''
		self.scan_protocol = ''
		self.trait = []

		# Internal data
		self.enter_dt = False
		self.enter_col = False
		self.col_pos = []
		self.col_name = []
		self.row_num = 0
		self.df = None

	def start(self, tag, attrib):
		self._tag = tag.split('}')[1] if '}' in tag else tag # Remove the namespace
		self._tag_stack.append(self._tag)
		# Process the attributes
		if (self._tag == 'Platform'):
			self.platform = attrib['iid']
		if (self._tag == 'Database'):
			self.db = attrib['iid']
		if (self._tag == 'Sample'):
			self.id = attrib['iid']
		if (self._tag == 'Characteristics'):
			self._subtag = attrib['tag'] if 'tag' in attrib else ''

		if (self.view != 'full'): return
		if (self._tag == 'Data-Table'):
			self.enter_dt = True
		if (self._tag == 'Column'):
			self.enter_col = True
			if (self.enter_dt):
				self.col_pos.append(int(attrib['position']))

		if (self._tag == 'Internal-Data' and self.enter_dt):
			self.row_num = int(attrib['rows'])

	def end(self, tag):
		self._tag = tag.split('}')[1] if '}' in tag else tag # Remove the namespace
		self._tag_stack.pop()
		if (self.view != 'full'): return
		if (self._tag == 'Data-Table'):
			self.enter_dt = False
		if (self._tag == 'Column'):
			self.enter_col = False

	def data(self, data):
		if data.isspace(): return
		data = data.strip()
		# Process the text content
		if (self._tag == 'Submission-Date'):
			self.submit_date = data
		if (self._tag == 'Release-Date'):
			self.release_date = data
		if (self._tag == 'Last-Update-Date'):
			self.last_update = data

		if (self._tag == 'Title'):
			self.title = data
		if (self._tag == 'Description'):
			if (self._tag_stack[-2] == 'Sample'):
				self.description = data
		if (self._tag == 'Data-Processing'):
			self.data_processing = data
		if (self._tag == 'Type'):
			self.type = data
		if (self._tag == 'Source'):
			self.source = data
		if (self._tag == 'Organism'):
			self.organism = data
		if (self._tag == 'Molecule'):
			self.molecule = data
		if (self._tag == 'Characteristics'):
			if (self._subtag == 'tissue'):
				self.tissue = data
			elif (self._subtag == 'tissue type'):
				self.tissue_type = data
			self.trait.append(data)
		if (self._tag == 'Treatment-Protocol'):
			self.treat_protocol = data
		if (self._tag == 'Growth-Protocol'):
			self.growth_protocol = data
		if (self._tag == 'Extract-Protocol'):
			self.extract_protocol = data
		if (self._tag == 'Label-Protocol'):
			self.label_protocol = data
		if (self._tag == 'Label'):
			self.label = data
		if (self._tag == 'Hybridization-Protocol'):
			self.hybrid_protocol = data
		if (self._tag == 'Scan-Protocol'):
			self.scan_protocol = data
		if (self.view != 'full'): return
		if (self._tag == 'Name' and self.enter_dt and self.enter_col):
			self.col_name.append(data.upper())
		if (self._tag == 'Internal-Data'):
			data_sio = StringIO('\n'.join(['\t'.join(self.col_name), data.replace('\"', '')]))
			if (self.df is None):
				self.df = pd.read_csv(data_sio, sep='\t', header=0, index_col=0, na_values=['null']).fillna(value=0)
			else:
				df = pd.read_csv(data_sio, sep='\t', header=0, index_col=0, na_values=['null']).fillna(value=0)
				self.df = self.df.append(df)


	def close(self):
		pass

	def build(self):
		return {'platform':self.platform,'db':self.db, 'submit_date':self.submit_date, 'release_date':self.release_date, 'last_update':self.last_update, 'id':self.id, 'ftype':self.ftype, 'title':self.title, 'description':self.description, 'data_processing':self.data_processing, 'type':self.type, 'source':self.source, 'organism':self.organism, 'tissue':self.tissue, 'tissue_type':self.tissue_type, 'treat_protocol':self.treat_protocol, 'growth_protocol':self.growth_protocol, 'extract_protocol':self.extract_protocol, 'label_protocol':self.label_protocol, 'label':self.label, 'hybrid_protocol':self.hybrid_protocol, 'scan_protocol':self.scan_protocol, 'trait':';'.join(self.trait), 'data':self.df}


class GPLBuilder():
#	__metaclass__ = ABCMeta

	def __init__(self, view='brief'):
		self.view = view
		self._tag = ''
		self._tag_stack = []
		self.submit_date = ''
		self.release_date = ''
		self.last_update = ''

		self.id = ''	# GEO Accession
		self.title = ''
		self.description = ''
		self.tech = ''
		self.organism = ''
		self.manufacturer = ''

		# Internal data
		self.enter_plfm = False
		self.enter_dt = False
		self.enter_col = False
		self.col_pos = []
		self.col_name = []
		self.col_desc = {}
		self.row_num = 0
		self.df = None

	def start(self, tag, attrib):
		self._tag = tag.split('}')[1] if '}' in tag else tag # Remove the namespace
		self._tag_stack.append(self._tag)
		# Process the attributes
		if (self._tag == 'Platform'):
			self.id = attrib['iid']
			self.enter_plfm = True

		if (self.view != 'full'): return
		if (self._tag == 'Data-Table'):
			self.enter_dt = True
		if (self._tag == 'Column'):
			self.enter_col = True
			if (self.enter_dt):
				self.col_pos.append(int(attrib['position']))

		if (self._tag == 'Internal-Data' and self.enter_dt):
			self.row_num = int(attrib['rows'])

	def end(self, tag):
		self._tag = tag.split('}')[1] if '}' in tag else tag # Remove the namespace
		self._tag_stack.pop()
		if (self._tag == 'Platform'):
			self.enter_plfm = False
		if (self.view != 'full'): return
		if (self._tag == 'Data-Table'):
			self.enter_dt = False
		if (self._tag == 'Column'):
			self.enter_col = False

	def data(self, data):
		if data.isspace(): return
		data = data.strip()
		# Process the text content
		if (self._tag == 'Submission-Date'):
			self.submit_date = data
		if (self._tag == 'Release-Date'):
			self.release_date = data
		if (self._tag == 'Last-Update-Date'):
			self.last_update = data

		if (self._tag == 'Title' and self.enter_plfm):
			self.title = data
		if (self._tag == 'Description' and self.enter_plfm):
			self.description = data
		if (self._tag == 'Technology' and self.enter_plfm):
			self.tech = data
		if (self._tag == 'Organism' and self.enter_plfm):
			self.organism = data
		if (self._tag == 'Manufacturer' and self.enter_plfm):
			self.manufacturer = data

		if (self.view != 'full'): return
		if (self._tag == 'Name' and self.enter_dt and self.enter_col):
			self.col_name.append(data.upper())
		if (self._tag == 'Description' and self.enter_dt and self.enter_col):
			self.col_desc[self.col_pos[-1]] = data
		if (self._tag == 'Internal-Data'):
			data_sio = StringIO('\n'.join(['\t'.join(self.col_name), data.replace('\"', '')]))
			if (self.df is None):
				self.df = pd.read_csv(data_sio, sep='\t', header=0, index_col=0, na_values=['null']).fillna(value=0)
			else:
				df = pd.read_csv(data_sio, sep='\t', header=0, index_col=0, na_values=['null']).fillna(value=0)
				self.df = self.df.append(df)


	def close(self):
		pass

	def build(self):
		return {'submit_date':self.submit_date, 'release_date':self.release_date, 'last_update':self.last_update, 'id':self.id, 'title':self.title, 'description':self.description, 'tech':self.tech, 'organism':self.organism, 'manufacturer':self.manufacturer, 'col_desc':[self.col_desc[i] if i in self.col_desc else '' for i in self.col_pos], 'data':self.df}


BUILDER_MAP = {'gse':GSEBuilder, 'gsm':GSMBuilder, 'gpl':GPLBuilder}


def fetch_geo(accessions, type='self', view='full', fmt='xml', saved_path=GEO_PATH):
	fmt2ext = {'text':'txt', 'xml':'xml', 'html':'html'}
	fs.mkdir(saved_path)
	geo_set = set(accessions)
	count = 0
	for acc in geo_set:
		tgt_file = os.path.join(saved_path, acc + '.%s' % fmt2ext[fmt])
		if os.path.exists(tgt_file):
			res = '\n'.join(fs.read_file(tgt_file))
		else:
			url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=%s&targ=%s&view=%s&form=%s" % (acc, type, view, fmt)
			try:
				res = urllib.request.urlopen(url).read()
			except:
				print('Failed to fetch the GEO file: %s' % acc)
				continue
			count += 1
			fs.write_file(res, tgt_file)
		yield res
	print("Number of obtained GEO documents: %i\n" % count)


def parse_geo(geo_fpath, view='brief', type='gse', fmt='xml'):
	Builder = BUILDER_MAP[type]
	builder = Builder(view=view)
	geo_str = fs.read_file(geo_fpath)
	parser = xmlextrc.get_parser(builder)
	try:
		parser.feed(nlp.clean_text('\n'.join(geo_str), encoding='utf-8', replacement=None))
	except Exception as err:
		print('Can not parse the file: %s' % geo_fpath)
		raise err
	parser.close()
	return builder.build()


def parse_geos(geo_fpaths, view='brief', type='gse', fmt='xml'):
	Builder = BUILDER_MAP[type]
	for i, geo_str in enumerate(fs.read_files(geo_fpaths)):
		builder = Builder(view=view)
		parser = xmlextrc.get_parser(builder)
		try:
			parser.feed(nlp.clean_text(re.sub(u'[\x00-\x08\x0b-\x0c\x0e-\x1f]+', u'', '\n'.join(geo_str)), encoding='utf-8', replacement=None))
		except Exception as err:
			print('Can not parse the file: %s' % geo_fpaths[i])
			raise err
		parser.close()
		yield builder.build()
