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

import os, string, itertools

import numpy as np
import scipy as sp
import pandas as pd
import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .util import fs, io, func


class AdvancedVectorizer(object):
    def __init__(self, vctrzr_cls, lemma=False, stem=False, synonym=False, phraser_fpath=None, keep_orig=False):
        self.vctrzr_cls = vctrzr_cls
        self.lemmatizer = nltk.stem.WordNetLemmatizer() if lemma else None
        self.stemmer = nltk.stem.SnowballStemmer('english') if stem else None
        self.synonym = synonym
        if phraser_fpath and os.path.exists(phraser_fpath):
            from gensim.models.phrases import Phraser
            self.phraser = Phraser.load(phraser_fpath)
        else:
            self.phraser = None
        self.keep_orig = keep_orig
    def build_analyzer(self):
        analyzer = self.vctrzr_cls.build_analyzer(self)
        def func(doc):
            res = (w for w in analyzer(doc))
            orig_words, res = itertools.tee(res)
            orig_words = set(orig_words)
            if self.lemmatizer:
                if self.keep_orig:
                    orig_res, res, cur_res = itertools.tee(res, 3)
                    orig_words |= set(cur_res)
                    res = itertools.chain(orig_res, (self.lemmatizer.lemmatize(w) for w in res if self.lemmatizer.lemmatize(w) not in orig_words))
                else:
                    res = (self.lemmatizer.lemmatize(w) for w in res)
            if self.stemmer:
                if self.keep_orig:
                    orig_res, res, cur_res = itertools.tee(res, 3)
                    orig_words |= set(cur_res)
                    res = itertools.chain(orig_res, (self.stemmer.stem(w) for w in res if self.stemmer.stem(w) not in orig_words))
                else:
                    res = (self.stemmer.stem(w) for w in res)
            if self.synonym:
                if self.keep_orig:
                    orig_res, res, cur_res = itertools.tee(res, 3)
                    orig_words |= set(cur_res)
                    res = itertools.chain(orig_res, (l.name() for w in res for s in nltk.corpus.wordnet.synsets(w) for l in s.lemmas() if l.name() not in orig_words))
                else:
                    res = (l.name() for w in res for s in nltk.corpus.wordnet.synsets(w) for l in s.lemmas())
            if self.phraser:
                if self.keep_orig:
                    orig_res, res, cur_res = itertools.tee(res, 3)
                    orig_words |= set(cur_res)
                    res = itertools.chain(orig_res, (phrz for phrz in self.phraser[(w for w in res)] if phrz not in orig_words))
                else:
                    res = (phrz for phrz in self.phraser[(w for w in res)])
            return res
        return func


class AdvancedCountVectorizer(AdvancedVectorizer, CountVectorizer):
    def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.int64, lemma=False, stem=False, synonym=False, phraser_fpath=None, keep_orig=False):
        CountVectorizer.__init__(self, input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range, analyzer=analyzer, max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype)
        AdvancedVectorizer.__init__(self, CountVectorizer, lemma=lemma, stem=stem, synonym=synonym, phraser_fpath=phraser_fpath, keep_orig=keep_orig)

    def build_analyzer(self):
        return AdvancedVectorizer.build_analyzer(self)


class AdvancedTfidfVectorizer(AdvancedVectorizer, TfidfVectorizer):
    def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer='word', stop_words=None, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False, lemma=False, stem=False, synonym=False, phraser_fpath=None, keep_orig=False):
        TfidfVectorizer.__init__(self, input=input, encoding=encoding, decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase, preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer, stop_words=stop_words, token_pattern=token_pattern, ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=vocabulary, binary=binary, dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
        AdvancedVectorizer.__init__(self, CountVectorizer, lemma=lemma, stem=stem, synonym=synonym, phraser_fpath=phraser_fpath, keep_orig=keep_orig)

    def build_analyzer(self):
        return AdvancedVectorizer.build_analyzer(self)


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
	for i in range(len(annot_locs)):
		token_list = []
		for loc in annot_locs[i]:
			for j in range(len(token_locs)):	# Make sure to start from the early position
				# Align the starting position of the annotation. It may be aligned to the middle of a token.
				if (abs(token_locs[j][0] - loc[0]) <= error or (token_locs[j][0] < loc[0] and loc[0] < token_locs[j][1])):
					# Determine whether the following tokens should be aligned to the annotation
					for k in range(j, len(token_locs)):
						token2annot[k].append(i)
						token_list.append(k)
						# Align the ending position of the annotation. It may be aligned to the middle of a token.
						if (abs(token_locs[k][1] - loc[1]) <= error or loc[1] < token_locs[k][1]):
							break
					else:
						print('Failed to find ending position of the \'%s\' annotation from the %i token.' % (i, j))
						# print loc, token_locs
					break
			else:
				print('Failed to find opening position of the \'%s\' annotation.' % i)
				# print loc, token_locs
		if (len(token_list) == 0):
			print('Failed to find tokens of the \'%s\' annotation.' % i)
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
		tknzr = nltk.tokenize.MWETokenizer(mwes, separator='_')	# mwes should look like "[('a', 'little'), ('a', 'little', 'bit')]"
		tokens = tknzr.tokenize(text.split())
	elif (model == 'stanford'):
		tknzr = nltk.tokenize.StanfordTokenizer(**kwargs)
		tokens = tknzr.tokenize(text)
	elif (model == 'treebank'):
		tknzr = nltk.tokenize.TreebankWordTokenizer()
		tokens = tknzr.tokenize(text)
	elif (model == 'sent'):
		tokens = nltk.tokenize.sent_tokenize(text, **kwargs)
	elif (model == 'word'):
		tokens = nltk.tokenize.word_tokenize(text, **kwargs)
	if (ret_loc):
		return tokens, find_location(text, tokens)
	return tokens


def span_tokenize(text):
	return list(nltk.tokenize.WhitespaceTokenizer().span_tokenize(text))


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
		wnl = nltk.stem.WordNetLemmatizer()
		lemmatized_tokens = [wnl.lemmatize(t, **kwargs) for t in tokens]
		return lemmatized_tokens


def stem(tokens, model='porter', **kwargs):
	if (model == 'porter'):
		stemmer = nltk.stem.PorterStemmer()
	elif (model == 'isri'):
		stemmer = nltk.stem.ISRIStemmer()
	elif (model == 'lancaster'):
		stemmer = nltk.stem.LancasterStemmer()
	elif (model == 'regexp'):
		stemmer = nltk.stem.RegexpStemmer(**kwargs)
	elif (model == 'rslp'):
		stemmer = nltk.stem.RSLPStemmer()
	elif (model == 'snowball'):
		stemmer = nltk.stem.SnowballStemmer(**kwargs)
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


def dpnd_trnsfm(dict_data, shape, encoding='categorical', **kwargs):
	from scipy.sparse import coo_matrix
	idx_list, v_list = zip(*dict_data.items())
	idx_list = np.array(idx_list)
	# Convert the value vector to a singular
	int_func = np.frompyfunc(int, 1, 1)
	if (encoding == 'binary'):
		hash_func = np.frompyfunc(lambda x: 1 if x and not str(x).isspace() else 0, 1, 1)
	elif (encoding == 'categorical'): # 0: No relation; others: relation types
		class_lbs = set(map(str.strip, map(str, v_list)))
		try:
			class_lbs.remove('')
		except:
			pass
		if 'class_lbmap' in kwargs:
			orig_class_num = len(kwargs['class_lbmap'])
			class_lbs -= set(kwargs['class_lbmap'].keys())
			class_lbmap = func.update_dict(kwargs['class_lbmap'], dict(zip(class_lbs, range(orig_class_num, orig_class_num + len(class_lbs)))))
		else:
			class_lbmap = dict(zip(class_lbs, np.arange(len(class_lbs))+1))
		hash_func = np.frompyfunc(lambda x: class_lbmap.setdefault(str(x), 0), 1, 1)
	else:
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


def parse(text, method='spacy', fmt='mt', tree_shape='symm', dpnd_encoding='categorical', dpnd_classmap={}, cached_id=None, cache_path=None, **kwargs):
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
				sdmt = dpnd_trnsfm(sub_dpnd_mt, (len(sent), len(sent)), encoding=dpnd_encoding, class_lbmap=dpnd_classmap)
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
			print(e)
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
				sdmt = dpnd_trnsfm(sub_dpnd_mt, (len(token_list), len(token_list)), class_lbmap=dpnd_classmap)
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
				sdmt = dpnd_trnsfm(sub_dpnd_mt, (len(sent['words']), len(sent['words'])), class_lbmap=dpnd_classmap)
				sddf = pd.DataFrame(sdmt.tocsr().todense())
			dpnd_dfs.append(sddf)
		coref = sents.setdefault('coref', [])

	# Write to cache
	if (cached_id is not None):
		io.write_obj((tokens, dpnd_dfs, coref), cache_file)
	return tokens, dpnd_dfs, coref


def parse_all(text, method='spacy', fmt='mt', tree_shape='symm', dpnd_encoding='categorical', dpnd_classmap={}, cached_id=None, cache_path=None, **kwargs):
	sf_tokens, sf_dpnd_dfs, sf_coref = parse(text, method=method, fmt=fmt, tree_shape=tree_shape, cached_id=cached_id, cache_path=cache_path)
	if (method == 'stanford'):
		return sf_tokens, sf_dpnd_dfs, sf_coref
	tokens, dpnd_dfs, coref = parse(text, method=method, fmt=fmt, tree_shape=tree_shape, dpnd_encoding=dpnd_encoding, dpnd_classmap=dpnd_classmap, cached_id=cached_id, cache_path=cache_path, **kwargs)
	# Coreference resolution
	sftkns2tokens, tokens2sftkns = annot_list_align([[[token['loc']] for token in token_list] for token_list in sf_tokens], [[token['loc'] for token in token_list] for token_list in tokens], error=0)
	coref = [[((pairs[0][0], sftkns2tokens[pairs[0][1]][pairs[0][2]][0][0], sftkns2tokens[pairs[0][1]][pairs[0][2]][0][1], sftkns2tokens[pairs[0][1]][pairs[0][3]][0][1], sftkns2tokens[pairs[0][1]][min(pairs[0][4], len(sftkns2tokens[pairs[0][1]]) - 1)][0][1]), (pairs[1][0], sftkns2tokens[pairs[1][1]][pairs[1][2]][0][0], sftkns2tokens[pairs[1][1]][pairs[1][2]][0][1], sftkns2tokens[pairs[1][1]][pairs[1][3]][0][1], sftkns2tokens[pairs[1][1]][min(pairs[1][4], len(sftkns2tokens[pairs[1][1]]) - 1)][0][1])) for pairs in crf] for crf in sf_coref] # The word offset in coreference resolution of Stanford parser may exceed the number of return tokens
	return tokens, dpnd_dfs, coref
