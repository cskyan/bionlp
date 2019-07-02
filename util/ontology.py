#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: ontology.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-10-16 21:21:37
###########################################################################
#

import os, re, sys, logging, itertools
from operator import itemgetter
from optparse import OptionParser

import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix

from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery as rprepareq

from ..spider.sparql import SPARQL
from .. import nlp
from . import fs, func


if sys.platform.startswith('win32'):
	DATA_PATH = 'D:\\data\\ontolib'
elif sys.platform.startswith('linux'):
	DATA_PATH = os.path.join(os.path.expanduser('~'), 'data', 'ontolib')

RDFS_LABEL_MAP = {('rdfs','label'):'lb'}
MESH_EQPRDC_MAP = {('meshv','concept'):'co', ('meshv','preferredConcept'):'pc', ('meshv','relatedConcept'):'rc', ('meshv','term'):'tm', ('meshv','preferredTerm'):'pt'}
MESH_BTPRDC_MAP = {('meshv','broader'):'br', ('meshv','broaderConcept'):'bc', ('meshv','broaderDescriptor'):'bd'}
MESH_NTPRDC_MAP = {('meshv','narrowerConcept'):'nc'}

DCTERMS = Namespace('http://purl.org/dc/terms/')
RDFS = Namespace('http://www.w3.org/2000/01/rdf-schema#')
XSD = Namespace('http://www.w3.org/2001/XMLSchema#')
OWL = Namespace('http://www.w3.org/2002/07/owl#')
MESHV = Namespace('http://id.nlm.nih.gov/mesh/vocab#')
OBO = Namespace('http://purl.obolibrary.org/obo/')
OBOWL = Namespace('http://www.geneontology.org/formats/oboInOwl#')
OMIM = Namespace('http://identifiers.org/omim/')
DBID = Namespace('http://www.drugbank.ca/drugbank-id/')
DBV = Namespace('http://www.drugbank.ca/vocab#')
DGIDBV = Namespace('http://dgidb.genome.wustl.edu/vocab#')
DGIDB_GENE = Namespace('http://dgidb.genome.wustl.edu/gene/')
DGIDB_DRUG = Namespace('http://dgidb.genome.wustl.edu/drug/')

opts, args = {}, []


def replace_invalid_sparql_str(text, replacement=''):
	return re.sub(r'^\+|[?|$|!|#]', replacement, nlp.clean_txt(str(text))).strip()


def replace_invalid_str(text, replacement=''):
	return re.sub(r'^\+|[\(|\)|\[|\]|?|$|!|#]', replacement, nlp.clean_txt(str(text))).strip()


def clean_result(db):
	if (db == 'dgnet'):
		def clean_dgnet(text, replacement=''):
			return re.sub(r'\[.*\]', replacement, nlp.clean_txt(str(text))).strip()
		return clean_dgnet
	else:
		return None


def filter_result(db):
	dbres_maxlen = {'dgidb':255}
	max_len = dbres_maxlen.setdefault('db', 50)
	def filter_dgidb(txt_list):
		new_list = []
		for txt in txt_list:
			if (len(txt) <= max_len):
				new_list.append(txt)
		return new_list
	return filter_dgidb


def get_dburi(db_path, type='', **kwargs):
	if (type == 'sqlite'):
		fs.mkdir(fs.pardir(db_path))
		return 'sqlite:///' + os.path.splitext(db_path)[0] + '.db'
	elif (type == 'bsddb'):
		db_path = os.path.join(db_path, 'store')
		fs.mkdir(db_path)
		return db_path
	return db_path


def get_prepareq(g):
	if (isinstance(g, Graph)):
		return rprepareq
	elif (isinstance(g, SPARQL)):
		return SPARQL.prepareQuery


def files2db(fpaths, fmt='xml', db_name='', db_type='Sleepycat', saved_path='.', cache=False):
	if (type(fpaths) is not list):
		if (os.path.isfile(fpaths)):
			fpaths = [fpaths]
		elif (os.path.isdir(fpaths)):
			fpaths = sorted(fs.listf(fpaths, full_path=True))
	db_name = os.path.splitext(os.path.basename(fpaths[0]))[0] if db_name == '' else db_name
	db_path = os.path.join(saved_path, db_name)
	is_existed = False
	if (db_type == 'SQLAlchemy'):
		from rdflib_sqlalchemy import registerplugins
		registerplugins()
		dburi = get_dburi(db_path, type='sqlite')
		if (os.path.exists(db_path+'.db')):
			is_existed = True
	else:
		dburi = get_dburi(db_path, type='bsddb')
		if (len(fs.listf(dburi)) > 0):
			is_existed = True
	if (is_existed):
		print('%s Database %s exists!' % (db_type, dburi))
		return db_path

	print('Creating RDF database %s in %s...' % (db_name, dburi))
	graph = Graph(store=db_type, identifier=db_name)
	graph.open(dburi, create=True)
	for fpath in fpaths:
		graph.parse(fpath, format=fmt)
	graph.close()
	return db_path


def files2dbs(fpaths, fmt='xml', db_names=[], db_type='Sleepycat', saved_path='.', merge=False, merged_dbname='', cache=False):
	if (type(fpaths) is not list):
		if (os.path.isfile(fpaths)):
			fpaths = [fpaths]
		elif (os.path.isdir(fpaths)):
			fpaths = sorted(fs.listf(fpaths, full_path=True))
	db_names = [os.path.splitext(os.path.basename(fpath))[0] for fpath in fpaths] if (len(db_names) == 0 or len(db_names) != len(fpaths)) else db_names
	db_paths = []
	for fpath, db_name in zip(fpaths, db_names):
		try:
			db_paths.append(files2db([fpath], fmt=fmt, db_name=db_name, db_type=db_type, saved_path=saved_path, cache=cache))
		except Exception as e:
			print('Cannot create RDF database %s from file %s!' % (db_name, fpath))
			print(e)
	if (not merge):
		return db_paths, None
	merged_dbname = db_names[0]+'_all' if (merged_dbname == '') else merged_dbname
	merged_dbpath = os.path.join(saved_path, merged_dbname)
	if (db_type == 'SQLAlchemy'):
		merged_dburi = get_dburi(merged_dbpath, type='sqlite')
	else:
		merged_dburi = get_dburi(merged_dbpath, type='bsddb')
	print('Creating merged RDF database %s in %s...' % (merged_dbname, merged_dburi))
	graph = Graph(store=db_type, identifier=merged_dbname)
	graph.open(merged_dburi, create=True)
	for db_path, db_name in zip(db_paths, db_names):
		try:
			g = get_db_graph(db_path, db_name=db_name, db_type=db_type)
			graph += g
		except Exception as e:
			print('Cannot merge RDF database %s in path %s !' % (db_name, db_path))
			print(e)
			g.close()
			continue
	graph.close()
	return db_paths, merged_dbpath


def get_db_graph(db_path, db_name='', db_type='Sleepycat'):
	db_name = os.path.splitext(os.path.basename(db_path))[0] if db_name == '' else db_name
	if (db_type == 'SQLAlchemy'):
		from rdflib_sqlalchemy import registerplugins
		registerplugins()
		dburi = get_dburi(db_path, type='sqlite')
	else:
		dburi = get_dburi(db_path, type='bsddb')
	print('Reading RDF database %s in %s' % (db_name, dburi))
	graph = Graph(store=db_type, identifier=db_name)
	graph.open(dburi)
	return graph


def get_id(g, label, lang='en', idns='', prdns=[], idprds={}):
	prepareQuery = get_prepareq(g)
	where_clause = ' UNION '.join(['''{?x %s ?c}''' % ':'.join(p) for p in idprds.keys()])
	# The 'i' parameter in regex function means case insensitive
	q_str = '''
		SELECT DISTINCT ?x ?c WHERE {
			%s .
			FILTER regex(str(?c), "%s", "i")
			FILTER langMatches(lang(?c), "%s")}
		''' % (where_clause, replace_invalid_str(label, ' '), lang)
	q = prepareQuery(q_str, initNs=dict([('rdfs', RDFS)]+prdns))
	result = g.query(q)
	return [(str(row[0]), row[1].toPython()) for row in result] if idns.isspace() else [(str(row[0]).strip(idns), row[1].toPython()) for row in result if row[0].startswith(idns)]


def get_label(g, id, lang='en', idns='', prdns=[], lbprds={}):
	id, idns_str = replace_invalid_sparql_str(id, '_'), str(idns)
	idns = Namespace(idns_str)
	prepareQuery = get_prepareq(g)
	where_clause = ' UNION '.join(['''{%s %s ?o}''' % (id if idns_str.isspace() else 'idns:'+id, ':'.join(p)) for p in lbprds.keys()])
	# The 'i' parameter in regex function means case insensitive
	q_str = '''
		SELECT DISTINCT ?o WHERE {
			%s .
			FILTER langMatches(lang(?o), "%s")}
		''' % (where_clause, lang)
	q = prepareQuery(q_str, initNs=dict([('rdfs', RDFS)]+prdns+([] if idns_str.isspace() else [('idns', idns)])))
	result = g.query(q)
	return [row[0].toPython() for row in result]


def slct_sim_terms(g, label, lang='en', exhausted=False, prdns=[], eqprds={}):
	prepareQuery = get_prepareq(g)
	sim_terms = []
	if (exhausted):
		prdc_str = ' | '.join(['%s | ^%s' % (':'.join(p), ':'.join(p)) for p in eqprds.keys()])
		q_str = '''
			SELECT DISTINCT ?x WHERE {
				"%s"%s ^rdfs:label / (%s)+ / rdfs:label ?x . }
			''' % (label, '@%s'%lang if lang else '', prdc_str)
		q = prepareQuery(q_str, initNs=dict([('rdfs', RDFS)]+prdns))
		result = g.query(q)
		for row in result:
			sim_terms.append(row[0].toPython())
	else:
		for p in eqprds.keys():
			q_str = '''
				SELECT DISTINCT ?x WHERE {
					"%s"%s ^rdfs:label / (%s | ^%s) / rdfs:label ?x . }
				''' % (label, '@%s'%lang if lang else '', ':'.join(p), ':'.join(p))
			q = prepareQuery(q_str, initNs=dict([('rdfs', RDFS)]+prdns))
			result = g.query(q)
			for row in result:
				sim_terms.append('%s_%s' % (eqprds[p], row[0].toPython()))
	return sim_terms


def define_mesh_fn(g, lang='en', prdns=[], eqprds={}):
	prepareQuery = get_prepareq(g)
	def find_neighbors(g, vertices):
		neighbors = []
		for vertex in vertices:
			if not vertex: continue
			if (len(eqprds) == 0):
				q_str = '''
						SELECT DISTINCT ?x WHERE {
							"%s"%s ^rdfs:label ?c1 .
							{?c2 ?p ?c1} UNION {?c1 ?p ?c2} .
							?c2 rdfs:label ?x . }
					''' % (vertex, '@%s'%lang if lang else '')
			else:
				prdc_str = ' | '.join(['%s | ^%s' % (':'.join(p), ':'.join(p)) for p in eqprds.keys()])
				q_str = '''
						SELECT DISTINCT ?x WHERE {
							"%s"%s ^rdfs:label / (%s) / rdfs:label ?x . }
					''' % (vertex, '@%s'%lang if lang else '', prdc_str)
			q = prepareQuery(q_str, initNs=dict([('rdfs', RDFS)]+prdns))
			result = g.query(q)
			for row in result:
				neighbors.append(row[0].toPython())
		return neighbors
	return find_neighbors


def define_meshtree_fn(g, lang='en', prdns=[], eqprds={}):
	prepareQuery = get_prepareq(g)
	def find_neighbors(g, vertices):
		neighbors = []
		for vertex in vertices:
			if not vertex: continue
			q_str = '''
					SELECT DISTINCT ?x WHERE {
						"%s"%s ^meshv:treeNumber ?c1 .
						{?c2 ?p ?c1} UNION {?c1 ?p ?c2} .
						?c2 meshv:treeNumber ?x . }
				''' % (vertex, '@%s'%lang if lang else ''),
			q = prepareQuery(q_str, initNs=dict([('rdfs', RDFS), ('meshv', MESHV)]+prdns))
			result = g.query(q)
			for row in result:
				neighbors.append(row[0].toPython())
		return neighbors
	return find_neighbors


def define_fn(g, type='exact', has_id=True, **kwargs):
	if (type == 'exact'):
		return define_fn_exact(g, **kwargs) if has_id else define_fn_exact_noid(g, **kwargs)
	elif (type == 'fuzzy'):
		return define_fn_fuzzy(g, **kwargs) if has_id else define_fn_fuzzy_noid(g, **kwargs)


def define_fn_fuzzy(g, lang='', idns=[], prdns=[], eqprds={}):
	prepareQuery = get_prepareq(g)
	def find_neighbors(g, vertices):
		neighbors = []
		for vertex in vertices:
			if not vertex: continue
			if (len(eqprds) == 0):
				q_str = '''
						SELECT DISTINCT ?x WHERE {
							?s ^rdfs:label ?c1 .
							{?c2 ?p ?c1} UNION {?c1 ?p ?c2} .
							?c2 rdfs:label ?x .
							FILTER(!isBlank(?c1)) .
							FILTER(!isBlank(?c2)) .
							FILTER(regex(?s, "%s", "i")) .
							FILTER(lang(?s)="%s") .}
					''' % (replace_invalid_str(vertex), '%s'%lang if lang else '')
			else:
				prdc_str = ' | '.join(['%s | ^%s' % (':'.join(p), ':'.join(p)) for p in eqprds.keys()])
				q_str = '''
						SELECT DISTINCT ?x WHERE {
							?s ^rdfs:label / (%s) ?o .
							?o rdfs:label ?x .
							FILTER(!isBlank(?o)) .
							FILTER(regex(?s, "%s", "i")) .
							FILTER(lang(?s)="%s") .}
					''' % (prdc_str, replace_invalid_str(vertex), '%s'%lang if lang else '')
			q = prepareQuery(q_str, initNs=dict([('rdfs', RDFS)]+idns+prdns))
			result = g.query(q)
			if (not result): continue
			for row in result:
				neighbors.append(row[0].toPython())
		return neighbors
	return find_neighbors


def define_fn_exact(g, lang='', idns=[], prdns=[], eqprds={}):
	prepareQuery = get_prepareq(g)
	def find_neighbors(g, vertices):
		neighbors = []
		for vertex in vertices:
			if not vertex: continue
			if (len(eqprds) == 0):
				q_str = '''
						SELECT DISTINCT ?x WHERE {
							"%s"%s ^rdfs:label ?c1 .
							{?c2 ?p ?c1} UNION {?c1 ?p ?c2} .
							?c2 rdfs:label ?x .
							FILTER(!isBlank(?c1)) .
							FILTER(!isBlank(?c2)) .}
					''' % (replace_invalid_str(vertex), '@%s'%lang if lang else '')
			else:
				prdc_str = ' | '.join(['%s | ^%s' % (':'.join(p), ':'.join(p)) for p in eqprds.keys()])
				q_str = '''
						SELECT DISTINCT ?x WHERE {
							"%s"%s ^rdfs:label / (%s) ?o .
							?o rdfs:label ?x .
							FILTER(!isBlank(?o)) . }
					''' % (replace_invalid_str(vertex), '@%s'%lang if lang else '', prdc_str)
			q = prepareQuery(q_str, initNs=dict([('rdfs', RDFS)]+idns+prdns))
			result = g.query(q)
			for row in result:
				neighbors.append(row[0].toPython())
		return neighbors
	return find_neighbors


def define_fn_fuzzy_noid(g, lang='', idns=[], prdns=[], eqprds={}):
	prepareQuery = get_prepareq(g)
	def find_neighbors(g, vertices):
		neighbors = []
		for vertex in vertices:
			if not vertex: continue
			if (len(eqprds) == 0):
				q_str = '''
						SELECT DISTINCT ?x WHERE {
							{?c ?p ?x} UNION {?x ?p ?c} .
							FILTER(!isBlank(?x)) .
							FILTER(regex(str(?c), "%s", "i")) .}
					''' % replace_invalid_str(vertex)
			else:
				prdc_str = ' | '.join(['%s | ^%s' % (':'.join(p), ':'.join(p)) for p in eqprds.keys()])
				q_str = '''
						SELECT DISTINCT ?x WHERE {
							?c (%s) ?x .
							FILTER(!isBlank(?x)) .
							FILTER(regex(str(?c), "%s", "i")) .}
					''' % (prdc_str, replace_invalid_str(vertex))
			q = prepareQuery(q_str, initNs=dict([('rdfs', RDFS)]+idns+prdns))
			result = g.query(q)
			for row in result:
				neighbors.append(row[0].toPython())
		return neighbors
	return find_neighbors


def define_fn_exact_noid(g, lang='', idns=[], prdns=[], eqprds={}):
	prepareQuery = get_prepareq(g)
	def find_neighbors(g, vertices):
		neighbors = []
		for vertex in vertices:
			if not vertex: continue
			if (len(eqprds) == 0):
				q_str = '''
						SELECT DISTINCT ?x WHERE {
							{?c ?p ?x} UNION {?x ?p ?c} .
							FILTER(str(?c)="%s") .
							FILTER(!isBlank(?x)) .}
					''' % replace_invalid_str(vertex)
			else:
				prdc_str = ' | '.join(['%s | ^%s' % (':'.join(p), ':'.join(p)) for p in eqprds.keys()])
				q_str = '''
						SELECT DISTINCT ?x WHERE {
							?c (%s) ?x .
							FILTER(str(?c)="%s") .
							FILTER(!isBlank(?x)) .}
					''' % (prdc_str, replace_invalid_str(vertex))
			q = prepareQuery(q_str, initNs=dict([('rdfs', RDFS)]+idns+prdns))
			result = g.query(q)
			for row in result:
				neighbors.append(row[0].toPython())
		return neighbors
	return find_neighbors


def _default_fn(g, vertices):
	return set(func.flatten([np.where(g[i] == 1)[0].astype('int').tolist() for i in vertices])) - set(vertices)


# Transitive Closure in Subgraph
def transitive_closure_sg(g, vertices, find_neighbors=_default_fn, max_length=100):
	if (len(vertices) == 0):
		return coo_matrix([], dtype='int64')
	csgraph_dict = {}
	zero_lbs, clstr_lbs, neighbor_record, neighbor_set = np.zeros(len(vertices), dtype='int64'), np.arange(len(vertices), dtype='int64'), [[set([v])] for v in vertices], [set([v]) for v in vertices]
	for k in range(max_length):
		# All the vertices are clustered together
		if (np.array_equal(zero_lbs, clstr_lbs)):
			break
		# Find neighbors for each cluster with distance of 1
		for i in range(len(vertices)):
			neighbors = set(find_neighbors(g, neighbor_record[i][-1])) - neighbor_set[i]
			neighbor_record[i].append(neighbors)
			neighbor_set[i] |= neighbors
		for vrtx_pair in itertools.combinations(range(len(vertices)), 2):
			if (clstr_lbs[vrtx_pair[0]] == clstr_lbs[vrtx_pair[1]]):
				continue
			overlapped = False
			overlaps = neighbor_record[vrtx_pair[0]][-1] & neighbor_record[vrtx_pair[1]][-2]
			if (len(overlaps) > 0):
				overlapped = True
				# Update the latest neighbor record
				neighbor_record[vrtx_pair[0]][-1] -= overlaps
				# Update the distance between two vertices
				csgraph_dict[vrtx_pair] = (k + 1) * 2 - 1
			overlaps = neighbor_record[vrtx_pair[1]][-1] & neighbor_record[vrtx_pair[0]][-2]
			if (len(overlaps) > 0):
				overlapped = True
				neighbor_record[vrtx_pair[1]][-1] -= overlaps
				csgraph_dict[vrtx_pair[::-1]] = (k + 1) * 2 - 1
			overlaps = neighbor_record[vrtx_pair[0]][-1] & neighbor_record[vrtx_pair[1]][-1]
			if (len(overlaps) > 0):
				overlapped = True
				csgraph_dict[vrtx_pair] = (k + 1) * 2
				csgraph_dict[vrtx_pair[::-1]] = (k + 1) * 2
			if (overlapped):
				# Updtae the cluster labels
				label = min(clstr_lbs[vrtx_pair[0]], clstr_lbs[vrtx_pair[1]])
				clstr_lbs[vrtx_pair[0]], clstr_lbs[vrtx_pair[1]] = label, label
		# print clstr_lbs
		# print neighbor_record
	if (len(csgraph_dict) == 0):
		return coo_matrix([], dtype='int64')
	# print csgraph_dict
	idx_list, v_list = zip(*csgraph_dict.items())
	indices, values = np.array(idx_list, dtype='int64'), np.array(v_list, dtype='int64')
	return coo_matrix((values, (indices[:,0], indices[:,1])), shape=(len(vertices), len(vertices)), dtype='int64')


# Transitive Closure in Dynamic Subgraph
def transitive_closure_dsg(g, vertices, find_neighbors=_default_fn, filter=None, cleaner=None, min_length=1, max_length=5):
	vertices = list(set(vertices))
	if (len(vertices) == 0):
		return coo_matrix([], dtype='int64')
	# Graph with involed vertices, new vertices besides the concerned ones, vertex index map, all vertices set
	csgraph_dict, new_vrtx, vrtx_idx, all_vrtx = {}, [], dict([(x, i) for i, x in enumerate(vertices)]), set(vertices)
	# Zero label, cluster labels for the concerned vertices, two slots queue to keep the last two neighbor records, the distances between the newest neighbor sets to the corresponding concerned vertices, all neighbors for each concerned vertex
	zero_lbs, clstr_lbs, neighbor_record, neighbor_dist, neighbor_set = np.zeros(len(vertices), dtype='int64'), np.arange(len(vertices), dtype='int64'), [[set([]), set([v])] for v in vertices], [0 for v in vertices], [set([v]) for v in vertices]
	for k in range(max_length):
		if (np.array_equal(zero_lbs, clstr_lbs) and k > min_length):
			break
		# Obtain the new neighbors for each vertex
		for i in range(len(vertices)):
			neighbors = set(find_neighbors(g, neighbor_record[i][-1])) - neighbor_set[i]
			# Filter the vertices
			if (filter is not None):
				neighbors = set(filter(neighbors))
			if (cleaner is not None):
				neighbors = set(map(cleaner, neighbors))
			# Update the neighbor record queue, neighbor distance, whole neighbor set (leave overlap for later calculation)
			del neighbor_record[i][0]
			neighbor_record[i].append(neighbors)
			neighbor_dist[i] += 1
			neighbor_set[i] |= neighbors
			# Recognize new vertices and update index
			new_v = neighbors - all_vrtx
			vrtx_idx.update(dict([(x, j) for j, x in zip(range(len(vertices) + len(new_vrtx), len(vertices) + len(new_vrtx) + len(new_v)), new_v)]))
			new_vrtx.extend(list(new_v))
			all_vrtx |= new_v
			# print neighbors
			# Update the distance between new neighbors and the concerned vertex
			for v in neighbors:
				csgraph_dict[i, vrtx_idx[v]] = csgraph_dict[vrtx_idx[v], i] = neighbor_dist[i]
		# vrtx_idx_r = dict(zip(vrtx_idx.values(), vrtx_idx.keys()))
		# print [(vrtx_idx_r[k[0]], vrtx_idx_r[k[1]], v) for k, v in csgraph_dict.iteritems()]
		# print neighbor_record, neighbor_dist
		# Recognize the neighbor overlap of each pair of concerned vertex
		for vrtx_pair in itertools.combinations(range(len(vertices)), 2):
			# These two concerned vertices are connected
			if (clstr_lbs[vrtx_pair[0]] == clstr_lbs[vrtx_pair[1]]):
				continue
			# Consider three types of overlaps, update the neighbor record queue and the graph
			pair_dist = neighbor_dist[vrtx_pair[0]] + neighbor_dist[vrtx_pair[1]]
			overlapped = False
			overlaps = neighbor_record[vrtx_pair[0]][1] & neighbor_record[vrtx_pair[1]][0]
			if (len(overlaps) > 0):
				overlapped = True
				neighbor_record[vrtx_pair[0]][1] -= overlaps
				csgraph_dict[vrtx_pair] = min(csgraph_dict.setdefault(vrtx_pair, pair_dist - 1), pair_dist - 1)
			overlaps = neighbor_record[vrtx_pair[1]][1] & neighbor_record[vrtx_pair[0]][0]
			if (len(overlaps) > 0):
				overlapped = True
				neighbor_record[vrtx_pair[1]][1] -= overlaps
				csgraph_dict[vrtx_pair[::-1]] = min(csgraph_dict.setdefault(vrtx_pair[::-1], pair_dist - 1), pair_dist - 1)
			overlaps = neighbor_record[vrtx_pair[0]][1] & neighbor_record[vrtx_pair[1]][1]
			if (len(overlaps) > 0):
				overlapped = True
				neighbor_record[vrtx_pair[0]][1] -= overlaps
				csgraph_dict[vrtx_pair] = min(csgraph_dict.setdefault(vrtx_pair, pair_dist), pair_dist)
				csgraph_dict[vrtx_pair[::-1]] = min(csgraph_dict.setdefault(vrtx_pair[::-1], pair_dist), pair_dist)
			if (overlapped):
				# Updtae the cluster labels indicating whether these two vertices are connected
				label = min(clstr_lbs[vrtx_pair[0]], clstr_lbs[vrtx_pair[1]])
				clstr_lbs[vrtx_pair[0]], clstr_lbs[vrtx_pair[1]] = label, label
		# print clstr_lbs
		# vrtx_idx_r = dict(zip(vrtx_idx.values(), vrtx_idx.keys()))
		# print [(vrtx_idx_r[k[0]], vrtx_idx_r[k[1]], v) for k, v in csgraph_dict.iteritems()]
		# print neighbor_record, neighbor_dist
	if (len(csgraph_dict) == 0):
		return coo_matrix([], shape=(len(vertices), len(vertices)), dtype='int64'), vertices
	# print csgraph_dict
	# Construct the graph matrix
	idx_list, v_list = zip(*csgraph_dict.items())
	indices, values = np.array(idx_list, dtype='int64'), np.array(v_list, dtype='int64')
	vrtx_num = indices.max() + 1
	return coo_matrix((values, (indices[:,0], indices[:,1])), shape=(vrtx_num, vrtx_num), dtype='int64'), vertices + new_vrtx


def test_mss():
	from scipy.sparse.csgraph import shortest_path
	init_vrtx = [0, 2, 5]
	vrtx_num = 10
	data = np.zeros((vrtx_num, vrtx_num), dtype='int64')
	for x, y in [(0,1), (1,2), (2,3), (3,4), (4,5), (3,8), (5,9), (0,6), (0,7)]:
		data[x,y] = 1
		data[y,x] = 1
	# dist = transitive_closure_sg(data, range(vrtx_num), _default_fn)
	# dist = transitive_closure_sg(data, init_vrtx, _default_fn)
	# dist, vname = transitive_closure_dsg(data, range(vrtx_num), _default_fn)
	dist, vname = transitive_closure_dsg(data, init_vrtx, _default_fn)
	# val = np.array(dist.data)
	# row = np.array(itemgetter(*dist.row)(idx))
	# col = np.array(itemgetter(*dist.col)(idx))
	# print row.shape, col.shape, val.shape
	dist = coo_matrix((dist.data, (itemgetter(*dist.row)(idx), itemgetter(*dist.col)(idx))), shape=(vrtx_num, vrtx_num))
	print(data)
	print(dist.todense())
	print(shortest_path(dist.tocsr()))
