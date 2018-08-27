#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: nlp.py
# Author: Shankai Yan
# E-mail: sk.yan@mY.cityu.edu.hk
# Created Time: 2016-10-19 17:39:18
###########################################################################
#

import os
import string
import itertools

import numpy as np
import scipy as sp
import pandas as pd
import nltk

from util import func
from util import fs
from util import io


def get_nltk_words():
	return nltk.corpus.words.words()
	
	
def clean_txt(text):
	return text.encode('ascii', 'ignore').decode('ascii').replace('\\', '')


def clean_text(text, encoding='ascii', replacement=' '):
	import unicodedata
	if (encoding == ''):
		return unicodedata.normalize('NFC', text.decode('utf-8', errors='replace'))
	elif (replacement is None):
		return unicodedata.normalize('NFC', text.decode('utf-8', errors='replace')).encode(encoding, errors='replace')
	else:
		return unicodedata.normalize('NFC', text.decode('utf-8', errors='replace')).encode(encoding, errors='replace').replace('?', replacement)

	
def find_location(text, tokens):
	offset, location = 0, []
	for t in tokens:
		new_offset = text.find(t, offset)
		if (new_offset == -1):
			location.append((-1, -1))
		else:
			offset = new_offset
			location.append((offset, offset + len(t)))
	return location
	
	
def annot_align(annot_locs, token_locs, error=0):
	token2annot, annot2token = [[] for x in range(len(token_locs))], [[] for x in range(len(annot_locs))]
	for i in xrange(len(annot_locs)):
		token_list = []
		for loc in annot_locs[i]:
			for j in xrange(len(token_locs)):	# Make sure to start from the early position
				# Align the starting position of the annotation. It may be aligned to the middle of a token.
				if (abs(token_locs[j][0] - loc[0]) <= error or (token_locs[j][0] < loc[0] and loc[0] < token_locs[j][1])):
					# Determine whether the following tokens should be aligned to the annotation
					for k in xrange(j, len(token_locs)):
						token2annot[k].append(i)
						token_list.append(k)
						# Align the ending position of the annotation. It may be aligned to the middle of a token.
						if (abs(token_locs[k][1] - loc[1]) <= error or loc[1] < token_locs[k][1]):
							break
					else:
						print 'Failed to find ending position of the \'%s\' annotation from the %i token.' % (i, j)
						# print loc, token_locs
					break
			else:
				print 'Failed to find opening position of the \'%s\' annotation.' % i
				# print loc, token_locs
		if (len(token_list) == 0):
			print 'Failed to find tokens of the \'%s\' annotation.' % i
			# print loc, token_locs
		annot2token[i].extend(token_list)
	# print '\t'.join(['-'.join([str(l), '<%s>'%a]) for l,a in zip(token_locs, token2annot)])
	# print '\t'.join(['-'.join([str(l), '<%s>'%a]) for l,a in zip(annot_locs, annot2token)])
	return annot2token, token2annot
	
	
def annot_list_align(annots_locs, tokens_locs, error=0):
	annots_list_idx, tokens_list_idx = sum([[(i, j) for j in range(len(annots_locs[i]))] for i in range(len(annots_locs))], []), sum([[(i, j) for j in range(len(tokens_locs[i]))] for i in range(len(tokens_locs))], [])
	annot_locs, token_locs = sum(annots_locs, []), sum(tokens_locs, [])
	annot2token, token2annot = annot_align(annot_locs, token_locs, error)
	annots2tokens, tokens2annots = [[] for i in range(len(annots_locs))], [[] for i in range(len(tokens_locs))]
	for i, a2ts in enumerate(annot2token):
		annots2tokens[annots_list_idx[i][0]].append([tokens_list_idx[a2t] for a2t in a2ts])
	for i, t2as in enumerate(token2annot):
		tokens2annots[tokens_list_idx[i][0]].append([annots_list_idx[t2a] for t2a in t2as])
	return annots2tokens, tokens2annots
	

def tokenize(text, model='word', ret_loc=False, **kwargs):
	if (model == 'casual'):
		tokens = nltk.tokenize.casual.casual_tokenize(text, **kwargs)
	elif (model == 'mwe'):
		from nltk.tokenize import MWETokenizer
		tknzr = MWETokenizer(mwes, separator='_')	# mwes should look like "[('a', 'little'), ('a', 'little', 'bit')]"
		tokens = tknzr.tokenize(text.split())
	elif (model == 'stanford'):
		from nltk.tokenize import StanfordTokenizer
		tknzr = StanfordTokenizer(**kwargs)
		tokens = tknzr.tokenize(text)
	elif (model == 'treebank'):
		from nltk.tokenize import TreebankWordTokenizer
		tknzr = TreebankWordTokenizer()
		tokens = tknzr.tokenize(text)
	elif (model == 'sent'):
		tokens = nltk.tokenize.sent_tokenize(text, **kwargs)
	elif (model == 'word'):
		tokens = nltk.tokenize.word_tokenize(text, **kwargs)
	if (ret_loc):
		return tokens, find_location(text, tokens)
	return tokens

	
def span_tokenize(text):
	from nltk.tokenize import WhitespaceTokenizer
	return list(WhitespaceTokenizer().span_tokenize(text))


# def del_punct(tokens, location=None):
	# if (location is not None):
		# tkn_locs = [(t, loc) for t, loc in zip(tokens, location) if t not in string.punctuation]
		# if (len(tkn_locs) == 0): 
			# return [], []
		# else:
			# return zip(*tkn_locs)
	# return [t for t in tokens if t not in string.punctuation]
def del_punct(tokens, ret_idx=False):
	return zip(*[(t, i) for i, t in enumerate(tokens) if t not in string.punctuation]) if ret_idx else [t for t in tokens if t not in string.punctuation]
	
	
def lemmatize(tokens, model='wordnet', **kwargs):
	if (model == 'wordnet'):
		from nltk.stem import WordNetLemmatizer
		wnl = WordNetLemmatizer()
		lemmatized_tokens = [wnl.lemmatize(t, **kwargs) for t in tokens]
		return lemmatized_tokens


def stem(tokens, model='porter', **kwargs):
	if (model == 'porter'):
		from nltk.stem.porter import PorterStemmer
		stemmer = PorterStemmer()
	elif (model == 'isri'):
		from nltk.stem.isri import ISRIStemmer
		stemmer = ISRIStemmer()
	elif (model == 'lancaster'):
		from nltk.stem.lancaster import LancasterStemmer
		stemmer = LancasterStemmer()
	elif (model == 'regexp'):
		from nltk.stem.regexp import RegexpStemmer
		stemmer = RegexpStemmer(**kwargs)
	elif (model == 'rslp'):
		from nltk.stem import RSLPStemmer
		stemmer = RSLPStemmer()
	elif (model == 'snowball'):
		from nltk.stem.snowball import SnowballStemmer
		stemmer = SnowballStemmer(**kwargs)
	stemmed_tokens = [stemmer.stem(t) for t in tokens]
	return stemmed_tokens
	
	
def pos(tokens):
	from nltk import pos_tag
	return pos_tag(tokens)


def corenlp2tree(sentence):
	for rel, head_i, word_i in sentence['indexeddependencies']:
		head_i_list = head_i.split('-')
		word_i_list = word_i.split('-')
		head, head_idx = '-'.join(head_i_list[:-1]), head_i_list[-1]
		word, word_idx = '-'.join(word_i_list[:-1]), word_i_list[-1]
		idx = int(word_idx)
		if rel == 'root':
			rel = 'ROOT' # NLTK expects that the root relation is labelled as ROOT!
		yield int(idx), int(head_idx), rel
	
	
def dpnd_trnsfm(dict_data, shape):
	from scipy.sparse import coo_matrix
	idx_list, v_list = zip(*dict_data.items())
	idx_list = np.array(idx_list)
	# Convert the value vector to a singular
	int_func = np.frompyfunc(int, 1, 1)
	hash_func = np.frompyfunc(lambda x: int(x) if str(x).isdigit() else 1, 1, 1)
	idx_list = int_func(idx_list)
	data = hash_func(v_list)
	return coo_matrix((data, (idx_list[:,0], idx_list[:,1])), shape=shape, dtype='int32')


def set_mt_point(id, pid, relation, mt={}, tree_shape='symm'):
	if (relation.upper() == 'ROOT'):
		mt[(id, id)] = relation
		return
	if (tree_shape == 'symm'):
		mt[(pid, id)] = relation
		mt[(id, pid)] = relation
	elif (tree_shape == 'td'):
		mt[(pid, id)] = relation
	elif (tree_shape == 'bu'):
		mt[(id, pid)] = relation
	return mt


def parse(text, method='spacy', fmt='mt', tree_shape='symm', cached_id=None, cache_path=None, **kwargs):			
	# From cache
	if (cached_id is not None):
		cache_path = os.path.join('.', '.parsed') if cache_path is None else cache_path
		fs.mkdir(cache_path)
		cache_file = os.path.join(cache_path, '_'.join([cached_id, method, fmt, tree_shape]) + '.pkl')
		if (os.path.exists(cache_file)):
			tokens, dpnd_dfs, coref = io.read_obj(cache_file)
			return tokens, dpnd_dfs, coref

	# Split the text into sentences
	if (type(text) is list):
		sentences = text
		text = ' '.join(sentences)
	else:
		sentences = tokenize(text, model='sent')
	# Parse the text or sentences
	tokens, dpnd_dfs, coref = [[] for x in range(3)]
	if (method == 'spacy'):
		import spacy
		spacy_nlp = spacy.load('en')
		doc = spacy_nlp(text, **kwargs)
		offset = 0
		for sent in doc.sents:
			tokens.append([dict(str=w.text.encode('ascii', 'replace'), loc=(w.idx, w.idx + len(w)), stem=w.lemma_, pos=w.pos_, stem_pos=pos(w.lemma_)[0][1], net=w.ent_id) if w.text not in string.punctuation else dict(str=w.text, loc=(w.idx, w.idx + len(w)), stem=w.lemma_, pos=w.pos_, stem_pos=w.pos_, net=w.ent_id) for w in sent])
			sub_dpnd_mt, sddf = {}, None
			for w in sent:
				set_mt_point(w.i - offset, w.head.i - offset, w.dep_, sub_dpnd_mt, tree_shape=tree_shape)
			if (len(sub_dpnd_mt) > 0):
				sdmt = dpnd_trnsfm(sub_dpnd_mt, (len(sent), len(sent)))
				sddf = pd.DataFrame(sdmt.tocsr().todense())
			dpnd_dfs.append(sddf)
			offset += len(sent)
	elif (method == 'bllip'):
		from bllipparser import RerankingParser
		rrp = RerankingParser()
		# Load the parser module
		try:
			rrp.check_models_loaded_or_error('auto')
		except ValueError as e:
			print e
			rrp = RerankingParser.fetch_and_load('GENIA+PubMed', verbose=True)
		# Parse each sentence
		global_tokens = []
		for sent in sentences:
			parse = rrp.parse(sent)[0]
			token_list, sub_dpnd_mt, sddf = [], {}, None
			for token in parse.ptb_parse.sd_tokens():
				token_list.append(dict(str=token.form, pos=token.pos))
				global_tokens.append(token.form)
				id, pid = token.index - 1, token.head - 1
				set_mt_point(id, pid, token.deprel, sub_dpnd_mt, tree_shape=tree_shape)
			if (len(sub_dpnd_mt) > 0):
				sdmt = dpnd_trnsfm(sub_dpnd_mt, (len(token_list), len(token_list)))
				sddf = pd.DataFrame(sdmt.tocsr().todense())
			tokens.append(token_list)
			dpnd_dfs.append(sddf)
		stems = stem(global_tokens)
		stem_pos = pos(stems)
		locations = find_location(text, global_tokens)
		i = 0
		for token_list in tokens:
			for token in token_list:
				token['stem'] = stems[i]
				token['loc'] = locations[i]
				token['stem_pos'] = stem_pos[i]
				i += 1
	elif (method == 'stanford'):
		from corenlp import StanfordCoreNLP
		corenlp_dir = os.environ['CORENLP_DIR']
		corenlp = StanfordCoreNLP(corenlp_dir)
		sents = corenlp.raw_parse(text)
		for sent in sents['sentences']:
			tokens.append([dict(str=word[0], loc=(int(word[1]['CharacterOffsetBegin']), int(word[1]['CharacterOffsetEnd'])), stem=word[1]['Lemma'], pos=word[1]['PartOfSpeech'], stem_pos=pos([word[1]['Lemma']])[0][1], net=word[1]['NamedEntityTag']) if word[0] not in string.punctuation else dict(str=word[0], loc=(int(word[1]['CharacterOffsetBegin']), int(word[1]['CharacterOffsetEnd'])), stem=word[1]['Lemma'], pos=word[1]['PartOfSpeech'], stem_pos=word[1]['PartOfSpeech'], net=word[1]['NamedEntityTag']) for word in sent['words']])
			sub_dpnd_mt, sddf = {}, None
			for id, pid, rel in corenlp2tree(sent):
				id, pid = id - 1, pid - 1
				set_mt_point(id, pid, rel, sub_dpnd_mt, tree_shape=tree_shape)
			if (len(sub_dpnd_mt) > 0):
				sdmt = dpnd_trnsfm(sub_dpnd_mt, (len(sent['words']), len(sent['words'])))
				sddf = pd.DataFrame(sdmt.tocsr().todense())
			dpnd_dfs.append(sddf)
		coref = sents.setdefault('coref', [])
			
	# Write to cache
	if (cached_id is not None):
		io.write_obj((tokens, dpnd_dfs, coref), cache_file)
	return tokens, dpnd_dfs, coref

	
def parse_all(text, method='spacy', fmt='mt', tree_shape='symm', cached_id=None, cache_path=None, **kwargs):	
	sf_tokens, sf_dpnd_dfs, sf_coref = parse(text, method=method, fmt=fmt, tree_shape=tree_shape, cached_id=cached_id, cache_path=cache_path)
	if (method == 'stanford'):
		return sf_tokens, sf_dpnd_dfs, sf_coref
	tokens, dpnd_dfs, coref = parse(text, method=method, fmt=fmt, tree_shape=tree_shape, cached_id=cached_id, cache_path=cache_path, **kwargs)
	# Coreference resolution
	sftkns2tokens, tokens2sftkns = annot_list_align([[[token['loc']] for token in token_list] for token_list in sf_tokens], [[token['loc'] for token in token_list] for token_list in tokens], error=0)
	coref = [[((pairs[0][0], sftkns2tokens[pairs[0][1]][pairs[0][2]][0][0], sftkns2tokens[pairs[0][1]][pairs[0][2]][0][1], sftkns2tokens[pairs[0][1]][pairs[0][3]][0][1], sftkns2tokens[pairs[0][1]][min(pairs[0][4], len(sftkns2tokens[pairs[0][1]]) - 1)][0][1]), (pairs[1][0], sftkns2tokens[pairs[1][1]][pairs[1][2]][0][0], sftkns2tokens[pairs[1][1]][pairs[1][2]][0][1], sftkns2tokens[pairs[1][1]][pairs[1][3]][0][1], sftkns2tokens[pairs[1][1]][min(pairs[1][4], len(sftkns2tokens[pairs[1][1]]) - 1)][0][1])) for pairs in crf] for crf in sf_coref] # The word offset in coreference resolution of Stanford parser may exceed the number of return tokens
	return tokens, dpnd_dfs, coref