#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: plot.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-03-29 22:31:14
###########################################################################
#

from __future__ import division
import os
import sys
import platform

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
if (platform.system() == 'Linux'):
	mpl.use('Agg')
	# If you are using GTK windows manager, please un-comment the following line
	# mpl.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.path as mpath

import io

MON=False

# Font config
plt.rc('savefig', dpi=350)
#plt.rc('text', usetex=True)


def gen_colors(num):
	base_color = ['b', 'g', 'r', 'm', 'y', 'c']
	base_alphas = np.linspace(0.8, 0.5, max(1, np.ceil(num / len(base_color))))
	colors = [base_color[i % len(base_color)] for i in range(num)]
	alphas = [base_alphas[int(i / len(base_color))] for i in range(num)]
	return colors, alphas
	
	
def gen_color_groups(num):
	base_color = ['b', 'g', 'r', 'm', 'y', 'c']
	group_color = list(base_color)
	group_color *= int(num / len(base_color))
	group_color += base_color[:num % len(base_color)]
	def cg(m, n):
		colors = [group_color[m]] * n
		alphas = np.linspace(0.8, 0.5, n)
		return colors, alphas
	return cg
	
	
def gen_colorls_groups(num):
	base_color = ['b', 'g', 'r', 'm', 'y', 'c']
	group_color = list(base_color)
	group_color *= int(num / len(base_color))
	group_color += base_color[:num % len(base_color)]
	def clsg(m, n):
		colors = [group_color[m]] * n
		line_styles = ['-', '--', '-.', ':', 'steps']
		lss = list(line_styles)
		lss *= int(num / len(line_styles))
		lss += line_styles[:num % len(line_styles)]
		return colors, lss
	return clsg
	
	
def gen_markers(num):
	base_markers = ['o','^','s','*','+','x','D','d','p','H','h','v','<','>','1','2','3','4','.','|','_',r'$\clubsuit$']
	markers = []
	if (num > 0):
		if (num <= len(base_markers)):
			markers.extend(base_markers[:num])
		else:
			markers.extend(base_markers[:])
			for i in range(num - len(base_markers)):
				markers.append(mpath.Path.unit_regular_polygon(i+7))
	return markers
	
	
def handle_annot(fig, annotator, annotation):
	global MON
	if (annotator is not None):
		annotator(axes=fig.get_axes(), draggable=True, display='one-per-axes', formatter='({x:.0f},{y:.0f})'.format, bbox=None, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
		MON=True
	if (len(annotation) > 0):
		for point, text in annotation.items():
			fig.get_axes()[0].annotate(text, xy=point, xycoords='data', xytext=(-15, +30), textcoords='offset points', fontsize=20, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
			
			
def smooth_data(x, y, pnum=300):
	from scipy.interpolate import spline
	new_x = np.linspace(x.min(), x.max(), pnum)
	new_y = spline(x, y, new_x)
	return new_x, new_y


def plot_roc(data, labels, groups=None, mltl_ls=False, title='Receiver operating characteristic', fname='roc', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	global MON

	fig = plt.figure()
	ax = plt.axes()

	if (groups is None):
		for i in xrange(len(data)):
			plt.plot(data[i][0], data[i][1], lw=1, label=labels[i])
	else:
		glbl_id = 0
		if (mltl_ls):
			colorls_groups = gen_colorls_groups(len(groups))
			for i, grp in enumerate(groups):
				colors, lss = colorls_groups(i, len(grp))
				for j, idx in enumerate(grp):
					plt.plot(data[idx][0], data[idx][1], lw=1, label=labels[glbl_id], color=colors[j], ls=lss[j])
					glbl_id += 1
		else:
			color_groups = gen_color_groups(len(groups))
			for i, grp in enumerate(groups):
				colors, alphas = color_groups(i, len(grp))
				for j, idx in enumerate(grp):
					plt.plot(data[idx][0], data[idx][1], lw=1, label=labels[glbl_id], color=colors[j], alpha=alphas[j])
					glbl_id += 1
	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
	
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	
	plt.title(title.title(), fontsize=20)

	plt.xlabel('False positive rate'.title(), fontsize=15)
	plt.ylabel('True positive rate'.title(), fontsize=15)
	
	plt.legend(loc="lower right", prop={'size':8})

	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_roc', data=data, labels=labels, groups=groups, title=title), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname)
	plt.close()


def plot_prc(data, labels, groups=None, mltl_ls=False, title='Precision recall characteristic', fname='prc', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	global MON
	
	fig = plt.figure()
	ax = plt.axes()
	
	if (groups is None):
		for i in xrange(len(data)):
			plt.plot(data[i][0], data[i][1], lw=1, label=labels[i])
	else:
		glbl_id = 0
		if (mltl_ls):
			colorls_groups = gen_colorls_groups(len(groups))
			for i, grp in enumerate(groups):
				colors, lss = colorls_groups(i, len(grp))
				for j, idx in enumerate(grp):
					plt.plot(data[idx][0], data[idx][1], lw=1, label=labels[glbl_id], color=colors[j], ls=lss[j])
					glbl_id += 1
		else:
			color_groups = gen_color_groups(len(groups))
			for i, grp in enumerate(groups):
				colors, alphas = color_groups(i, len(grp))
				for j, idx in enumerate(grp):
					plt.plot(data[idx][0], data[idx][1], lw=1, label=labels[glbl_id], color=colors[j], alpha=alphas[j])
					glbl_id += 1
	
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	
	plt.title(title.title(), fontsize=20)

	plt.xlabel('Recall'.title(), fontsize=15)
	plt.ylabel('Precision'.title(), fontsize=15)
	
	plt.legend(loc="lower left", prop={'size':8})

	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_prc', data=data, labels=labels, groups=groups, title=title), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname)
	plt.close()


def plot_bar(avg, std, xlabels, labels=None, title='Scores of mean and standard deviation', fname='score_mean_std', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	global MON
	
	fig = plt.figure()
	ax = plt.axes()
	
	colors, alphas = gen_colors(len(labels) if labels != None else 1)
	ind = np.arange(avg.shape[1])
	width = 3.5 * ax.get_xlim()[1] / len(xlabels) / (len(labels) if labels != None else 1)
	bars_list = []
	for i in xrange(avg.shape[0]):
		bar_list = plt.bar(ind+width*i, avg[i], width, yerr=std[i], color=colors[i], alpha=alphas[i], error_kw={'ecolor':'0.3','elinewidth':1.5,'capsize':4})
		bars_list.append(bar_list)
	plt.ylim([0, (avg+std).max()+(avg.max()-ax.get_ylim()[0])/12.])

	plt.title(title.title(), fontsize=20)
	plt.xticks(ind+width/2, xlabels, rotation=15)
	plt.ylabel('Scores'.title(), fontsize=15)
	
	if (labels != None):
		plt.legend([bar_list[0] for bar_list in bars_list], labels, loc="upper right", prop={'size':8})

	def autolabel(bars):
		for bar in bars:
			height = bar.get_height()
			ax.text(bar.get_x()+bar.get_width()/2.+0.15, height+0.005, '%.2f'%height, ha='center', va='bottom', fontsize=6)

	for i in xrange(len(bars_list)):
		autolabel(bars_list[i])

	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_bar', avg=avg, std=std, xlabels=xlabels, labels=labels, title=title), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname)
	plt.close()
	
	
def plot_hist(data, xlabel, ylabel, normed=False, cumulative=False, log=False, fit_line=False, title='Histogram', fname='hist', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	global MON
	
	fig = plt.figure()
	ax = plt.axes()
	n, bins, patch = plt.hist(data, max(10, min(20, int(data.max()))), normed=normed, cumulative=cumulative, log=log, facecolor='green', alpha=0.75)
	if (fit_line):
		y = mlab.normpdf(bins, data.mean(), data.std())
		l = plt.plot(bins, y, 'r--', linewidth=1)
	if (len(ref_lines) > 0):
		x_lines = ref_lines.setdefault('x', [])
		if (len(x_lines) > 0):
			for x in x_lines:
				ax.axvline(x=x, c='c', ls='--')
		y_lines = ref_lines.setdefault('y', [])
		if (len(y_lines) > 0):
			for y in y_lines:
				ax.axhline(y=y, c='r', ls='--')
	plt.xlabel(xlabel, fontsize=20)
	plt.ylabel(ylabel, fontsize=20)
	plt.title(title)
	plt.grid(True)
	
	new_annot = {}
	for point, text in annotation.items():
		point = bins[point[0]+1], n[point[0]]
		text = '(%i,%i)' % (point[0], point[1]) if text == '' else text
		new_annot[point] = text
	annotation = new_annot
	
	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_hist', data=data, xlabel=xlabel, ylabel=ylabel, normed=normed, cumulative=cumulative, log=log, fit_line=fit_line, title=title), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname)
	plt.close()
	
	
def plot_2hist(data1, data2, xlabel, ylabel, normed=False, cumulative=False, log=False, title='2Histogram', fname='2hist', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	global MON
	
	fig = plt.figure()
	ax = plt.axes()
	
	bin_num = max(10, min(min(20, int(data1.max())), int(data2.max())))
	n, bins, patch = plt.hist(data1, bin_num, normed=normed, cumulative=cumulative, log=log, facecolor='c', alpha=0.75)
	n, bins, patch = plt.hist(data2, bin_num, normed=normed, cumulative=cumulative, log=log, facecolor='g', alpha=0.85)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	ax.set_yticks([0.5, 1])
	ax.set_yticklabels(['0.5', '1'])
	plt.title(title)
	plt.grid(True)

	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_2hist', data1=data1, data2=data2, xlabel=xlabel, ylabel=ylabel, normed=normed, cumulative=cumulative, log=log, title=title), fname)
	
	fig_size = fig.get_size_inches()
	fig.set_size_inches(fig_size[0], 0.55 * fig_size[1])
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname)
	plt.close()
	
	
def plot_scat(data, xlabel, ylabel, scale=(None, None), title='Scatter', fname='scat', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	global MON
	
	fig = plt.figure()
	ax = plt.axes()
	if (scale[0]):
		ax.set_xscale(scale[0], basex=10)
	if (scale[1]):
		ax.set_yscale(scale[1], basex=10)
	
	# plt.scatter(data[:,0], data[:,1], c='black', alpha=0.75)
	plt.scatter(data[:,0], data[:,1], facecolors='none', edgecolors='k')
	plt.xlim([data[:,0].min() - 0.05, data[:,0].max() + 0.05])
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True)
	
	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_scat', data=data, xlabel=xlabel, ylabel=ylabel, title=title), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname)
	plt.close()
	
	
def plot_param(values, score_avg, score_std, xlabel='Parameter Value', ylabel='Metric Score', title='Micro F1 Score of default RF with different feature numbers', fname='params', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	global MON
	
	fig = plt.figure()
	ax = plt.axes()
	
	lower_val, higher_val = score_avg - score_std, score_avg + score_std
	
	plt.scatter(values, score_avg)
	plt.plot(values, score_avg, linewidth=2, color='r')
	plt.fill_between(values, lower_val, higher_val, color='w', facecolor='r', alpha=0.3, interpolate=True)

	plt.ylim(lower_val.min() * 0.8, higher_val.max() * 1.2)
	plt.xlabel(xlabel, fontsize=15)
	plt.ylabel(ylabel, fontsize=15)
	plt.title(title)
	plt.grid(True)
	
	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_param', values=values, score_avg=score_avg, score_std=score_std, xlabel=xlabel, ylabel=ylabel, title=title), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname)
	plt.close()
	
	
def plot_ftnum(data, labels, marker=False, title='Micro F1 Score of default RF with different feature numbers', fname='ftnum', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	global MON
	
	fig = plt.figure()
	ax = plt.axes()
	ax.set_xscale('log', basex=10)
	
	colors, alphas = gen_colors(len(data))
	markers = gen_markers(len(data))
	for i in xrange(len(data)):
		if (marker):
			plt.plot(data[i][0], data[i][1], lw=1, label=labels[i], color=colors[i], alpha=alphas[i], marker=markers[i])
		else:
			plt.plot(data[i][0], data[i][1], lw=1, label=labels[i], color=colors[i], alpha=alphas[i])
		
	plt.title(title.title(), fontsize=12)
	ax.yaxis.grid()

	plt.xlabel('Number of features'.title(), fontsize=15)
	plt.ylabel('Micro f1 score'.title(), fontsize=15)
	
	plt.legend(loc="upper right", prop={'size':8}, numpoints=1)

	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_ftnum', data=data, labels=labels, title=title), fname)

	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname)
	plt.close()

	
def plot_clt(data, labels, decomp=False, title='Clustering', fname='clustering', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	global MON
	
	fig = plt.figure()
	ax = plt.axes()
	
	if (data.shape[1] > 2 and decomp):
		from sklearn.pipeline import make_pipeline
		from .. import ftdecomp
		pipeline = make_pipeline(ftdecomp.DecompTransformer(n_components, ftdecomp.t_sne, initial_dims=min(15*n_components, X.shape[1]), perplexity=30.0), Normalizer(copy=False), MinMaxScaler(copy=False))
		new_data = pipeline.fit_transform(data.as_matrix())
		if (isinstance(data, pd.DataFrame)):
			data = pd.DataFrame(new_data, index=data.index, dtype=data.dtypes[0])
		else:
			data = new_data
	
	unq_labels = np.unique(labels)
	label_map = dict(zip(unq_labels, range(len(unq_labels))))
	markers = gen_markers(len(unq_labels))
	if (unq_labels[0] == -1):
		colors, alphas = gen_colors(len(unq_labels) - 1)
		colors, alphas = ['grey'] + colors, [1] + alphas
	else:
		colors, alphas = gen_colors(len(unq_labels))
	plot_colors = [colors[label_map[x]] for x in labels]
	plt.scatter(data[:,0], data[:,1], c=plot_colors, edgecolors=plot_colors)

	plt.xlabel('$x_{1}$', fontsize=15)
	plt.ylabel('$x_{2}$', fontsize=15)
	plt.title(title)
	plt.grid(True)
	
	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_clt', data=data, labels=labels, decomp=decomp, title=title), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname)
	plt.close()
	
	
def plot_fzyclt(data, labels, decomp=False, title='Clustering', fname='clustering', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	if (data.shape[0] != labels.shape[0]):
		print 'Input data shape %s is not consistent with the label shape %s !' % (data.shape, labels.shape)
		return
	global MON
	
	fig = plt.figure()
	ax = plt.axes()
	
	if (data.shape[1] > 2 and decomp):
		from sklearn.pipeline import make_pipeline
		from .. import ftdecomp
		pipeline = make_pipeline(ftdecomp.DecompTransformer(n_components, ftdecomp.t_sne, initial_dims=min(15*n_components, X.shape[1]), perplexity=30.0), Normalizer(copy=False), MinMaxScaler(copy=False))
		new_data = pipeline.fit_transform(data.as_matrix())
		if (isinstance(data, pd.DataFrame)):
			data = pd.DataFrame(new_data, index=data.index, dtype=data.dtypes[0])
		else:
			data = new_data
	
	markers = gen_markers(labels.shape[1])
	colors, alphas = gen_colors(labels.shape[1])
	label_sum = labels.sum(axis=1)
	full_data_list, full_colors, edge_colors = [[] for x in range(3)]
	for i, ls in enumerate(label_sum):
		if (ls == 0):
			full_data_list.append(data[i,:])
			full_colors.append('grey')
			edge_colors.append('grey')
		else:
			label = labels[i]
			full_data_list.append(data[i,:].reshape((1, data.shape[1])).repeat(ls, axis=0))
			full_colors.extend([colors[x] for x in np.where(label > 0)[0]])
			label_idx = np.where(label > 0)[0]
			edge_colors.extend([colors[label_idx[0]]] + [colors[-1-x] for x in label_idx[1:]])
	full_data = np.vstack(full_data_list)
	plt.scatter(full_data[:,0], full_data[:,1], c=full_colors, edgecolors=edge_colors)

	plt.xlabel('$x_{1}$', fontsize=15)
	plt.ylabel('$x_{2}$', fontsize=15)
	plt.title(title)
	plt.grid(True)
	
	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_clt', data=data, labels=labels, decomp=decomp, title=title), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname)
	plt.close()
	
	
def plot_clt_hrc(data, title='', dist_metric='euclidean', fname='hrc_clustering', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	global MON
	
	if (not style is None):
		plt.style.use(style)

	fig = plt.figure()
	
	if ((dist_metric == 'precomputed' and data.shape[0] != data.shape[1]) or len(data.shape) != 2):
		dist_metric='euclidean'
	from sklearn.metrics.pairwise import pairwise_distances
	D = pairwise_distances(data, metric=dist_metric)
	
	# dendrogram
	import scipy.cluster.hierarchy as sch
	ax_dendro_l, ax_dendro_t = fig.add_axes([0.01,0.01,0.2,0.8]), fig.add_axes([0.22,0.82,0.7,0.17])
	Y_l, Y_t = sch.linkage(D, method='centroid'), sch.linkage(D, method='centroid')
	Z_l, Z_t = sch.dendrogram(Y_l, orientation='left', ax=ax_dendro_l), sch.dendrogram(Y_t, orientation='top', ax=ax_dendro_t)
	axdlx, axdly, axdrx, axdry = ax_dendro_l.set_xticks([]), ax_dendro_l.set_yticks([]), ax_dendro_t.set_xticks([]), ax_dendro_t.set_yticks([])

	# distance matrix
	ax_matrix = fig.add_axes([0.22,0.01,0.7,0.8])
	index = Z_l['leaves']
	D = D[index,:][:,index]
	im = ax_matrix.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.coolwarm)
	axmx, axmy = ax_matrix.set_xticks([]), ax_matrix.set_yticks([])
	
	# colorbar
	ax_color = fig.add_axes([0.93,0.01,0.02,0.8])
	fig.colorbar(im, cax=ax_color)
	
	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_clt', data=data, labels=labels, decomp=decomp, title=title), fname)
	
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname)
	plt.close()


def plot_fig(fig, fname='fig'):
	global MON
	if (MON):
		fig.show()
	else:
		fig.savefig(fname)
		
		
def plot_data(data, fname='fig', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	global MON
	func = globals()[data['func'].item()]
	params = dict(data)
	del params['func']
	for k, v in params.iteritems():
		if (len(v.shape) == 0):
			params[k] = v.item()
	func(style=style, ref_lines=ref_lines, plot_cfg=plot_cfg, annotator=annotator, annotation=annotation, **params)
	
	
def plot_files(fpaths, style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	if (type(fpaths) is not list):
		fpaths = [fpaths]
	for fpath in fpaths:
		fname, fext = os.path.splitext(fpath)
		if (fext == '.pkl'):
			fig = io.read_obj(fpath)
			plot_fig(fig, fname)
		elif (fext == '.npz'):
			data = io.read_npz(fpath)
			plot_data(data, fname=fname, style=style, ref_lines=ref_lines, plot_cfg=plot_cfg, annotator=annotator, annotation=annotation)
	

def main():
	pass


if __name__ == '__main__':
	main()