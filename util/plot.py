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
import os, sys, platform, pkgutil, itertools

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
if (platform.system() == 'Linux'):
	if (mpl.get_backend() != 'module://ipykernel.pylab.backend_inline' and not globals().has_key('plt')): mpl.use('Agg')
	# If you are using GTK windows manager, please un-comment the following line
	# mpl.use('GTKAgg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.path as mpath

import io, func

MON = False
BASE_COLOR = ['b', 'g', 'r', 'y', 'c', 'm']

# Font config
plt.rc('savefig', dpi=600)
#plt.rc('text', usetex=True)
SC = ';;'


def mix_white(colors, ratio=0.1):
	if (type(colors) != np.ndarray):
		colors = np.array(colors)
	return (1 - ratio) * colors + ratio


def gen_colors(num, cm=None):
	if (cm is None):
		base_color = BASE_COLOR
	else:
		cmap = plt.cm.get_cmap(cm, num)
		base_color = map(cmap, range(num))
	base_alphas = np.linspace(0.8, 0.5, max(1, np.ceil(num / len(base_color))))
	colors = [base_color[i % len(base_color)] for i in range(num)]
	alphas = [base_alphas[int(i / len(base_color))] for i in range(num)]
	return colors, alphas
	
	
def gen_color_groups(num, cm=None):
	if (cm is None):
		base_color = BASE_COLOR
	else:
		cmap = plt.cm.get_cmap(cm, num)
		base_color = map(cmap, range(num))
	group_color = list(base_color)
	group_color *= int(num / len(base_color))
	group_color += base_color[:num % len(base_color)]
	def cg(m, n):
		colors = [group_color[m]] * n
		alphas = np.linspace(0.8, 0.5, n)
		return colors, alphas
	return cg
	
	
def gen_colorls_groups(num, cm=None):
	if (cm is None):
		base_color = BASE_COLOR
	else:
		cmap = plt.cm.get_cmap(cm, num)
		base_color = map(cmap, range(num))
	group_color = list(base_color)
	group_color *= int(num / len(base_color))
	group_color += base_color[:num % len(base_color)]
	def clsg(m, n):
		colors = [group_color[m]] * n
		line_styles = ['-', '--', '-.', ':', 'steps']
		lss = list(line_styles)
		lss *= int(n / len(line_styles))
		lss += line_styles[:n % len(line_styles)]
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
	
	
def handle_annot(fig, annotator, annotation, annr_props={}, annn_props={}):
	global MON
	if (annotator is not None):
		annotator(axes=fig.get_axes(), draggable=True, display='one-per-axes', formatter='({x:.0f},{y:.0f})'.format, bbox=None, arrowprops=annr_props.setdefault('arrow', dict(arrowstyle='->', connectionstyle='arc3,rad=.2')))
		MON=True
	if (len(annotation) > 0):
		for point, text in annotation.items():
			fig.get_axes()[0].annotate(text, xy=point, xycoords='data', xytext=annn_props.setdefault('xytext', (-15, +30)), textcoords='offset points', fontsize=annn_props.setdefault('fontsize', 20), arrowprops=annn_props.setdefault('arrow', dict(arrowstyle='->', connectionstyle='arc3,rad=.2')))


def handle_refline(ax, ref_lines, **kwargs):
	def _setdefault_param(params, param_name, default_value, length):
		param_values = kwargs.setdefault(param_name, [default_value] * length)
		param_values = param_values if type(param_values) is list else [param_values]
		param_values = func.padding_list(param_values, length, param_values[-1])
		return param_values
	if (len(ref_lines) > 0):
		# Vertical lines
		x_lines = ref_lines.setdefault('x', [])
		x_lines = x_lines if type(x_lines) is list else [x_lines]
		if (len(x_lines) > 0):
			# x_colors = kwargs.setdefault('xcolors', ['c'] * len(x_lines))
			# x_colors = x_colors if type(x_colors) is list else [x_colors]
			# x_colors = func.padding_list(x_colors, len(x_lines), x_colors[-1])
			x_colors = _setdefault_param(kwargs, 'xcolors', 'c', len(x_lines))
			x_widths = _setdefault_param(kwargs, 'xwidths', '2', len(x_lines))
			for x, color, width in zip(x_lines, x_colors, x_widths):
				ax.axvline(x=x, c=color, ls='--', linewidth=width)
		# Horizontal lines
		y_lines = ref_lines.setdefault('y', [])
		y_lines = y_lines if type(y_lines) is list else [y_lines]
		if (len(y_lines) > 0):
			# y_colors = kwargs.setdefault('ycolors', ['r'] * len(y_lines))
			# y_colors = y_colors if type(y_colors) is list else [y_colors]
			# y_colors = func.padding_list(y_colors, len(y_lines), y_colors[-1])
			y_colors = _setdefault_param(kwargs, 'ycolors', 'r', len(y_lines))
			y_widths = _setdefault_param(kwargs, 'ywidths', '2', len(y_lines))
			for y, color, width in zip(y_lines, y_colors, y_widths):
				ax.axhline(y=y, c=color, ls='--', linewidth=width)
		# Curves
		curves = ref_lines.setdefault('c', [])
		curves = curves if type(curves) is list else [curves]
		if (len(curves) > 0):
			# c_colors = kwargs.setdefault('ccolors', ['y'] * len(curves))
			# c_colors = c_colors if type(c_colors) is list else [c_colors]
			# c_colors = func.padding_list(c_colors, len(curves), c_colors[-1])
			c_colors = _setdefault_param(kwargs, 'ccolors', 'y', len(curves))
			c_widths = _setdefault_param(kwargs, 'cwidths', '2', len(curves))
			for curve, color, width in zip(curves, c_colors, c_widths):
				try:
					ax.plot(curve['x'], curve['y'], c=color, linewidth=width)
				except:
					pass
			
			
def smooth_data(x, y, pnum=300):
	from scipy.interpolate import spline
	new_x = np.linspace(x.min(), x.max(), pnum)
	new_y = spline(x, y, new_x)
	return new_x, new_y


def plot_roc(data, labels, groups=None, mltl_ls=False, title='Receiver operating characteristic', fname='roc', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, **kwargs):
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)
	cmap = plot_cfg.setdefault('cmap', 'jet')
	
	if (style is not None):
		plt.style.use(style)

	fig = plt.figure()
	ax = plt.axes()
	
	params = dict(lw=1)
	params.update(kwargs)
	if (groups is None):
		for i in xrange(len(data)):
			plt.plot(data[i][0], data[i][1], label=labels[i], **params)
	else:
		glbl_id = 0
		col_kwargs = dict(cm=cmap if len(groups) > len(BASE_COLOR) else None)
		if (mltl_ls):
			colorls_groups = gen_colorls_groups(len(groups), **col_kwargs)
			for i, grp in enumerate(groups):
				colors, lss = colorls_groups(i, len(grp))
				for j, idx in enumerate(grp):
					plt.plot(data[idx][0], data[idx][1], label=labels[glbl_id], color=colors[j], ls=lss[j], **params)
					glbl_id += 1
		else:
			color_groups = gen_color_groups(len(groups), **col_kwargs)
			for i, grp in enumerate(groups):
				colors, alphas = color_groups(i, len(grp))
				for j, idx in enumerate(grp):
					plt.plot(data[idx][0], data[idx][1], label=labels[glbl_id], color=colors[j], alpha=alphas[j], **params)
					glbl_id += 1
	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
	
	plt.xlim(plot_cfg.setdefault('xlim', [-0.05, 1.05]))
	plt.ylim(plot_cfg.setdefault('ylim', [-0.05, 1.05]))
	
	if (not plot_cfg.setdefault('notitle', False)):
		plt.title(title.title(), fontsize=20)

	plt.xlabel('False positive rate'.title(), fontsize=15)
	plt.ylabel('True positive rate'.title(), fontsize=15)
	
	plt.legend(loc="lower right", ncol=plot_cfg.setdefault('lgnd_ncol', 1), prop={'size':plot_cfg.setdefault('lgnd_fontsize', 8)})

	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_roc', data=data, labels=labels, groups=groups, mltl_ls=mltl_ls, title=title, **params), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname+'.%s'%fmt, format=fmt)
	plt.close()


def plot_prc(data, labels, groups=None, mltl_ls=False, title='Precision recall characteristic', fname='prc', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, **kwargs):
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)
	cmap = plot_cfg.setdefault('cmap', 'jet')
	
	if (style is not None):
		plt.style.use(style)
	
	fig = plt.figure()
	ax = plt.axes()

	params = dict(lw=1)
	params.update(kwargs)
	if (groups is None):
		for i in xrange(len(data)):
			plt.plot(data[i][0], data[i][1], label=labels[i], **params)
	else:
		glbl_id = 0
		col_kwargs = dict(cm=cmap if len(groups) > len(BASE_COLOR) else None)
		if (mltl_ls):
			colorls_groups = gen_colorls_groups(len(groups), **col_kwargs)
			for i, grp in enumerate(groups):
				colors, lss = colorls_groups(i, len(grp))
				for j, idx in enumerate(grp):
					plt.plot(data[idx][0], data[idx][1], label=labels[glbl_id], color=colors[j], ls=lss[j], **params)
					glbl_id += 1
		else:
			color_groups = gen_color_groups(len(groups), **col_kwargs)
			for i, grp in enumerate(groups):
				colors, alphas = color_groups(i, len(grp))
				for j, idx in enumerate(grp):
					plt.plot(data[idx][0], data[idx][1], label=labels[glbl_id], color=colors[j], alpha=alphas[j], **params)
					glbl_id += 1
	
	plt.xlim(plot_cfg.setdefault('xlim', [-0.05, 1.05]))
	plt.ylim(plot_cfg.setdefault('ylim', [-0.05, 1.05]))
	
	if (not plot_cfg.setdefault('notitle', False)):
		plt.title(title.title(), fontsize=20)

	plt.xlabel('Recall'.title(), fontsize=15)
	plt.ylabel('Precision'.title(), fontsize=15)
	
	plt.legend(loc="lower left", prop={'size':plot_cfg.setdefault('lgnd_fontsize', 8)})

	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_prc', data=data, labels=labels, groups=groups, mltl_ls=mltl_ls, title=title, **params), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname+'.%s'%fmt, format=fmt)
	plt.close()


def plot_bar(avg, std, xlabels, labels=None, title='Scores of mean and standard deviation', fname='score_mean_std', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, **kwargs):
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)

	if (style is not None):
		plt.style.use(style)
	
	fig = plt.figure()
	ax = plt.axes()
	
	params = dict(error_kw={'ecolor':'0.3','elinewidth':1.5,'capsize':4})
	params.update(kwargs)
	colors, alphas = gen_colors(len(labels) if labels != None else 1)
	ind = np.arange(avg.shape[1])
	width = 3.5 * ax.get_xlim()[1] / len(xlabels) / (len(labels) if labels != None else 1)
	bars_list = []
	for i in xrange(avg.shape[0]):
		bar_list = plt.bar(ind+width*i, avg[i], width, yerr=std[i], color=colors[i], alpha=alphas[i], **params)
		bars_list.append(bar_list)
	plt.ylim([0, (avg+std).max()+(avg.max()-ax.get_ylim()[0])/12.])

	if (not plot_cfg.setdefault('notitle', False)):
		plt.title(title.title(), fontsize=20)
	plt.xticks(ind+width/2, xlabels, rotation=15)
	plt.ylabel('Scores'.title(), fontsize=15)
	
	if (labels != None):
		plt.legend([bar_list[0] for bar_list in bars_list], labels, loc="upper right", prop={'size':plot_cfg.setdefault('lgnd_fontsize', 8)})

	def autolabel(bars):
		for bar in bars:
			height = bar.get_height()
			ax.text(bar.get_x()+bar.get_width()/2.+0.15, height+0.005, '%.2f'%height, ha='center', va='bottom', fontsize=6)

	for i in xrange(len(bars_list)):
		autolabel(bars_list[i])

	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_bar', avg=avg, std=std, xlabels=xlabels, labels=labels, title=title, **params), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname+'.%s'%fmt, format=fmt)
	plt.close()
	
	
def plot_hist(data, xlabel, ylabel, normed=False, cumulative=False, scale=(None, None), fit_line=False, title='Histogram', fname='hist', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, **kwargs):
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)
	xlabel_fontsize = plot_cfg.setdefault('xlabel_fontsize', 20)
	ylabel_fontsize = plot_cfg.setdefault('ylabel_fontsize', 20)
	title_fontsize = plot_cfg.setdefault('title_fontsize', 20)
	
	if (style is not None):
		plt.style.use(style)
	
	fig = plt.figure()
	ax = plt.axes()

	params = dict(bins=max(10, min(20, int(data.max()))), facecolor='green', alpha=0.75)
	params.update(kwargs)
	if (scale[0]):
		ax.set_xscale(scale[0], basex=10)
	if (scale[1]):
		ax.set_yscale(scale[1], basey=10)
	n, bins, patch = plt.hist(data, normed=normed, cumulative=cumulative, log=scale[1]=='log', **params)
	if (fit_line):
		y = mlab.normpdf(bins, data.mean(), data.std())
		l = plt.plot(bins, y, 'r--', linewidth=1)
	handle_refline(ax, ref_lines)
	plt.xlabel(xlabel, fontsize=xlabel_fontsize)
	plt.ylabel(ylabel, fontsize=ylabel_fontsize)
	if (not plot_cfg.setdefault('notitle', False)):
		plt.title(title.title(), fontsize=title_fontsize)
	plt.grid(plot_cfg.setdefault('grid', True))
	
	new_annot = {}
	for point, text in annotation.items():
		point = bins[point[0]+1], n[point[0]]
		text = '(%i,%i)' % (point[0], point[1]) if text == '' else text
		new_annot[point] = text
	annotation = new_annot
	
	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_hist', data=data, xlabel=xlabel, ylabel=ylabel, normed=normed, cumulative=cumulative, scale=scale, fit_line=fit_line, title=title, **params), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname+'.%s'%fmt, format=fmt)
	plt.close()
	
	
def plot_2hist(data1, data2, xlabel, ylabel, normed=False, cumulative=False, log=False, title='2Histogram', fname='2hist', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, **kwargs):
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)
	
	if (style is not None):
		plt.style.use(style)
	
	fig = plt.figure()
	ax = plt.axes()

	params = dict()
	params.update(kwargs)
	bin_num = max(10, min(min(20, int(data1.max())), int(data2.max())))
	n, bins, patch = plt.hist(data1, bin_num, normed=normed, cumulative=cumulative, log=log, facecolor='c', alpha=0.75, **params)
	n, bins, patch = plt.hist(data2, bin_num, normed=normed, cumulative=cumulative, log=log, facecolor='g', alpha=0.85, **params)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	ax.set_yticks([0.5, 1])
	ax.set_yticklabels(['0.5', '1'])
	if (not plot_cfg.setdefault('notitle', False)):
		plt.title(title.title(), fontsize=20)
	plt.grid(plot_cfg.setdefault('grid', True))

	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_2hist', data1=data1, data2=data2, xlabel=xlabel, ylabel=ylabel, normed=normed, cumulative=cumulative, log=log, title=title, **params), fname)
	
	fig_size = fig.get_size_inches()
	fig.set_size_inches(fig_size[0], 0.55 * fig_size[1])
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname+'.%s'%fmt, format=fmt)
	plt.close()
	
	
def plot_scat(data, xlabel, ylabel, groups=None, scale=(None, None), title='Scatter', fname='scat', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, annr_props={}, annn_props={}, **kwargs):
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)

	if (style is not None):
		plt.style.use(style)
	
	fig = plt.figure()
	ax = plt.axes()
	
	params = dict(facecolors='none', edgecolors='k')
	params.update(kwargs)
	if (scale[0]):
		ax.set_xscale(scale[0], basex=10)
	if (scale[1]):
		ax.set_yscale(scale[1], basey=10)
	if (groups is None):
		plt.scatter(data[:,0], data[:,1], **params)
	else:
		groups = pd.DataFrame(groups) if (type(groups) != pd.DataFrame) else groups
		legends = []
		if (len(groups.shape) > 1 and groups.shape[1] > 1):
			major_labels, minor_labels = groups.iloc[:,0], groups.iloc[:,1]
			uniq_major_lbs, uniq_minor_lbs = func.alignstrs(np.unique(major_labels).tolist(), plot_cfg.setdefault('lgnd_refmjrlbs', []), ret_all=True), func.alignstrs(np.unique(minor_labels).tolist(), plot_cfg.setdefault('lgnd_refmnrlbs', []), ret_all=True)
			mjrlb_map, mnrlb_map = dict(zip(uniq_major_lbs, range(len(uniq_major_lbs)))), dict(zip(uniq_minor_lbs, range(len(uniq_minor_lbs))))
			labels = pd.Series([SC.join(x) for x in zip(major_labels, minor_labels)], index=groups.index)
			unq_lbs = list(np.unique(labels))
			unq_lbs = func.alignstrs(unq_lbs, plot_cfg.setdefault('lgnd_reflbs', []), ret_all=True)
			markers = gen_markers(len(uniq_major_lbs))
			colors, alphas = gen_colors(len(uniq_minor_lbs))
			for i, lb in enumerate(unq_lbs):
				mjr, mnr = lb.split(SC)
				sub_data = data[(major_labels == mjr) & (minor_labels == mnr)]
				scatplt = plt.scatter(sub_data[:,0], sub_data[:,1], marker=markers[mjrlb_map[mjr]], c=colors[mnrlb_map[mnr]], alpha=alphas[mnrlb_map[mnr]], **params)
				legends.append((scatplt, ' '.join([mnr, mjr] if plot_cfg.setdefault('lgnd_xchg', False) else [mjr, mnr]).strip()))
		else:
			labels = groups.iloc[:,0]
			unq_lbs = np.unique(labels)
			unq_lbs = func.alignstrs(unq_lbs, plot_cfg.setdefault('lgnd_reflbs', []), ret_all=True)
			markers = gen_markers(labels.shape[0])
			for i, lb in enumerate(unq_lbs):
				sub_data = data[labels == lb]
				scatplt = plt.scatter(sub_data[:,0], sub_data[:,1], marker=markers[i], **params)
				legends.append((scatplt, lb))
		scatplts, legendlbs = zip(*legends)
		plt.legend(scatplts, legendlbs, scatterpoints=plot_cfg.setdefault('lgnd_size', 1), loc=plot_cfg.setdefault('lgnd_loc', 'upper right'), ncol=plot_cfg.setdefault('lgnd_ncol', 2), fontsize=plot_cfg.setdefault('lgnd_fontsize', 8), bbox_to_anchor=plot_cfg.setdefault('lgnd_bbox', None))
	plt.xlim([data[:,0].min() - 0.05, data[:,0].max() + 0.05])
	handle_refline(ax, ref_lines, **plot_cfg.setdefault('refl_kwargs', {}))
	plt.xlabel(xlabel, fontsize=plot_cfg.setdefault('xlabel_fontsize', 1))
	plt.ylabel(ylabel, fontsize=plot_cfg.setdefault('ylabel_fontsize', 1))
	if (not plot_cfg.setdefault('notitle', False)):
		plt.title(title.title(), fontsize=20)
	ax.set_axisbelow(True)
	plt.grid(plot_cfg.setdefault('grid', True), c='gray', alpha=0.5, linewidth=0.8, ls='--')
	
	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_scat', data=data, xlabel=xlabel, ylabel=ylabel, scale=scale, title=title, **params), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation, annr_props=annr_props, annn_props=annn_props)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname+'.%s'%fmt, format=fmt)
	plt.close()
	
	
def plot_violin(data, xlabel, ylabel, labels, groups=None, title='Violin Plot', fname='violin', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, **kwargs):
	import seaborn as sns
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)
	
	if (style is not None):
		plt.style.use(style)
	
	fig = plt.figure()
	
	params = dict(palette='muted', scale='count', inner='box', scale_hue=False)
	params.update(dict([(k.lstrip('sns_'), v) for k, v in kwargs.iteritems() if k.startswith('sns_')]))
	
	if (groups is None):
		all_data = np.concatenate(data, axis=0)
		df = pd.DataFrame(all_data, columns=[xlabel, ylabel])
		df[xlabel], df[ylabel] = df[xlabel].astype('category'), kwargs['log'] * np.log10(df[ylabel].astype('float')) if (kwargs.setdefault('log', 0) != 0) else df[ylabel].astype('float')
		ax = sns.violinplot(x=xlabel, y=ylabel, data=df, **params)
	else:
		glbl_id, max_grpmem, grp_labels = 0, 0, []
		for i, grp in enumerate(groups):
			max_grpmem = max(max_grpmem, len(grp))
			for j, idx in enumerate(grp):
				grp_labels.append([labels[glbl_id]] * data[idx].shape[0])
				glbl_id += 1
		all_data = np.concatenate([np.concatenate([data[i] for i in func.flatten_list(groups)], axis=0), np.array(func.flatten_list(grp_labels)).reshape((-1,1))], axis=1)
		df = pd.DataFrame(all_data, columns=[xlabel, ylabel, 'groups'])
		df[xlabel], df[ylabel], df['groups'] = df[xlabel].astype('category'), kwargs['log'] * np.log10(df[ylabel].astype('float')) if (kwargs.setdefault('log', 0) != 0) else df[ylabel].astype('float'), df['groups'].astype('category')
		ax = sns.violinplot(x=xlabel, y=ylabel, hue='groups', data=df, split=max_grpmem==2, **params)
		
	handle_refline(ax, ref_lines)
	if (kwargs.has_key('ax_ylim')):
		ax.set(ylim=kwargs['ax_ylim'])
	if (not plot_cfg.setdefault('notitle', False)):
		plt.title(title.title(), fontsize=20)
	legend_handles, legend_labels = ax.get_legend_handles_labels()
	ax.legend(legend_handles, legend_labels, loc=plot_cfg.setdefault('lgnd_loc', 'upper center'), prop={'size':plot_cfg.setdefault('lgnd_fontsize', 8)})

	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_violin', data=data, xlabel=xlabel, ylabel=ylabel, labels=labels, groups=groups, title=title, **kwargs), fname)

	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname+'.%s'%fmt, format=fmt)
	plt.close()
	
	
def plot_param(values, score_avg, score_std, xlabel='Parameter Value', ylabel='Metric Score', title='Micro F1 Score of default RF with different feature numbers', fname='params', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, **kwargs):
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)
	
	if (style is not None):
		plt.style.use(style)
	
	fig = plt.figure()
	ax = plt.axes()
	
	params = dict(color='w', facecolor='r', alpha=0.3, interpolate=True)
	params.update(kwargs)
	lower_val, higher_val = score_avg - score_std, score_avg + score_std
	
	plt.scatter(values, score_avg)
	plt.plot(values, score_avg, linewidth=2, color='r')
	plt.fill_between(values, lower_val, higher_val, **params)

	plt.ylim(lower_val.min() * 0.8, higher_val.max() * 1.2)
	plt.xlabel(xlabel, fontsize=15)
	plt.ylabel(ylabel, fontsize=15)
	if (not plot_cfg.setdefault('notitle', False)):
		plt.title(title.title(), fontsize=20)
	plt.grid(plot_cfg.setdefault('grid', True))
	
	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_param', values=values, score_avg=score_avg, score_std=score_std, xlabel=xlabel, ylabel=ylabel, title=title, **params), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname+'.%s'%fmt, format=fmt)
	plt.close()
	
	
def plot_ftnum(data, labels, marker=False, title='Micro F1 Score of default RF with different feature numbers', fname='ftnum', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}):
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)
	
	if (style is not None):
		plt.style.use(style)
	
	fig = plt.figure()
	ax = plt.axes()
	ax.set_xscale('log', basex=10)

	params = dict(lw=1)
	params.update(kwargs)
	colors, alphas = gen_colors(len(data))
	markers = gen_markers(len(data))
	for i in xrange(len(data)):
		if (marker):
			plt.plot(data[i][0], data[i][1], label=labels[i], color=colors[i], alpha=alphas[i], marker=markers[i], **params)
		else:
			plt.plot(data[i][0], data[i][1], label=labels[i], color=colors[i], alpha=alphas[i], **params)
		
	if (not plot_cfg.setdefault('notitle', False)):
		plt.title(title.title(), fontsize=20)
	ax.yaxis.grid()

	plt.xlabel('Number of features'.title(), fontsize=15)
	plt.ylabel('Micro f1 score'.title(), fontsize=15)
	
	plt.legend(loc="upper right", prop={'size':plot_cfg.setdefault('lgnd_fontsize', 8)}, numpoints={'size':plot_cfg.setdefault('legend_numpoints', 1)})

	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_ftnum', data=data, labels=labels, marker=marker, title=title, **params), fname)

	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname+'.%s'%fmt, format=fmt)
	plt.close()

	
def plot_clt(data, labels, decomp=False, title='Clustering', fname='clustering', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, **kwargs):
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)
	
	if (style is not None):
		plt.style.use(style)
	
	fig = plt.figure()
	ax = plt.axes()
	
	params = dict()
	params.update(kwargs)
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
	plt.scatter(data[:,0], data[:,1], c=plot_colors, edgecolors=plot_colors, **params)

	plt.xlabel('$x_{1}$', fontsize=15)
	plt.ylabel('$x_{2}$', fontsize=15)
	if (not plot_cfg.setdefault('notitle', False)):
		plt.title(title.title(), fontsize=20)
	plt.grid(plot_cfg.setdefault('grid', True))
	
	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_clt', data=data, labels=labels, decomp=decomp, title=title, **params), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname+'.%s'%fmt, format=fmt)
	plt.close()
	
	
def plot_fzyclt(data, labels, decomp=False, title='Clustering', fname='clustering', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, **kwargs):
	if (data.shape[0] != labels.shape[0]):
		print 'Input data shape %s is not consistent with the label shape %s !' % (data.shape, labels.shape)
		return
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)
	
	if (style is not None):
		plt.style.use(style)
	
	fig = plt.figure()
	ax = plt.axes()
	
	params = dict()
	params.update(kwargs)
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
	plt.scatter(full_data[:,0], full_data[:,1], c=full_colors, edgecolors=edge_colors, **params)

	plt.xlabel('$x_{1}$', fontsize=15)
	plt.ylabel('$x_{2}$', fontsize=15)
	if (not plot_cfg.setdefault('notitle', False)):
		plt.title(title.title(), fontsize=20)
	plt.grid(plot_cfg.setdefault('grid', True))
	
	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_clt', data=data, labels=labels, decomp=decomp, title=title, **params), fname)
	
	plt.tight_layout()
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname+'.%s'%fmt, format=fmt)
	plt.close()
	
	
def plot_clt_hrc(data, xlabel='', ylabel='', dist_metric='euclidean', dist_func=None, title='', fname='hrc_clustering', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, **kwargs):
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)
	lcolthrshd = plot_cfg.setdefault('lcolthrshd', style)
	
	if (style is not None):
		plt.style.use(style)
	plt.rc('axes', linewidth=kwargs.setdefault('rcntx_borderwidth', 0))

	fig = plt.figure()

	pdist_params = dict([(k.lstrip('pdist_'), v) for k, v in kwargs.iteritems() if k.startswith('pdist_')])
	if ((dist_metric == 'precomputed' and data.shape[0] != data.shape[1]) or len(data.shape) != 2):
		dist_metric='euclidean'
	from sklearn.metrics.pairwise import pairwise_distances
	D = pairwise_distances(data, metric=dist_metric, **pdist_params)
	if (dist_func is not None):
		orig_D = D
		new_D = dist_func(D)
		if (new_D.shape == D.shape):
			D = new_D
	condensed_D = D[np.triu_indices(D.shape[0], k=1)]

	# dendrogram
	import scipy.cluster.hierarchy as sch
	margin_l, margin_t = 0.04 if ylabel else 0, 0.04 if xlabel else 0
	mat_size = kwargs.setdefault('mat_size', (0.8, 0.9)) # image size (width,height)
	mat_size = [mat_size[0]-margin_l, mat_size[1]-margin_t]
	ax_dendro_l, ax_dendro_t = fig.add_axes([0.01+margin_l,0.01,0.9-margin_l-mat_size[0],mat_size[1]]), fig.add_axes([0.92-mat_size[0],mat_size[1]+0.02,mat_size[0],0.97-margin_t-mat_size[1]])
	linkage_params = func.update_dict(dict(method='ward'), dict([(k.lstrip('linkage_'), v) for k, v in kwargs.iteritems() if k.startswith('linkage_')]))
	Z_l = Z_t = sch.linkage(condensed_D, **linkage_params)
	io.write_npz(Z_t, 'hrc_z')
	dndrgram_params = dict([(k.lstrip('dndrgram_'), v) for k, v in kwargs.iteritems() if k.startswith('dndrgram_')])
	with plt.rc_context({'lines.linewidth':kwargs.setdefault('rcntx_linewidth', 0.1)}):
		R_l, R_t = sch.dendrogram(Z_l, orientation='left', ax=ax_dendro_l, **dndrgram_params), sch.dendrogram(Z_t, orientation='top', ax=ax_dendro_t, **dndrgram_params)
	axdlx, axdly, axdrx, axdry = ax_dendro_l.set_xticks([]), ax_dendro_l.set_yticks([]), ax_dendro_t.set_xticks([]), ax_dendro_t.set_yticks([])

	# distance matrix
	matshow_params = dict([(k.lstrip('matshow_'), v) for k, v in kwargs.iteritems() if k.startswith('matshow_')])
	ax_matrix = fig.add_axes([0.92-mat_size[0],0.01,mat_size[0],mat_size[1]])
	if (kwargs.has_key('dndrgram_truncate_mode') and kwargs['dndrgram_truncate_mode'] is not None and kwargs['dndrgram_truncate_mode'] != 'none'):
		dummy_R_l = dummy_R_t = sch.dendrogram(Z_t, no_plot=True)
	else:
		dummy_R_l, dummy_R_t = R_l, R_t
	io.write_npz(dummy_R_t, 'hrc_r')
	ordered_D = D[dummy_R_l['leaves'],:][:,dummy_R_t['leaves']] if dist_func is None else orig_D[dummy_R_l['leaves'],:][:,dummy_R_t['leaves']]
	im = ax_matrix.matshow(ordered_D, aspect='auto', origin='lower', cmap=plt.cm.coolwarm, **matshow_params)
	axmx, axmy = ax_matrix.set_xticks([]), ax_matrix.set_yticks([])
	
	# colorbar
	cbar_params = dict([(k.lstrip('cbar_'), v) for k, v in kwargs.iteritems() if k.startswith('cbar_')])
	ax_color = fig.add_axes([0.93,0.01,0.02,mat_size[1]])
	cbar = fig.colorbar(im, cax=ax_color, **cbar_params)
	cbarclim_params = dict([(k.lstrip('cbarclim_'), v) for k, v in kwargs.iteritems() if k.startswith('cbarclim_')])
	cbar.set_clim(cbarclim_params.setdefault('vmin', condensed_D.min()), cbarclim_params.setdefault('vmax', condensed_D.max()))
	cbartick_params = dict([(k.lstrip('cbartick_'), v) for k, v in kwargs.iteritems() if k.startswith('cbartick_')])
	cbar.ax.tick_params(labelsize=cbartick_params.setdefault('fontsize', 8))
	
	if (xlabel):
		fig.text(x=0.01, y=(0.01+mat_size[1])/2, s=xlabel, verticalalignment='center', horizontalalignment='left', rotation=90, fontsize=10)
	if (ylabel):
		fig.text(x=0.92-mat_size[0]/2, y=0.99, s=ylabel, verticalalignment='top', horizontalalignment='center', fontsize=10)
	
	if (plot_cfg.setdefault('save_obj', False)):
		io.write_obj(fig, fname)
	if (plot_cfg.setdefault('save_npz', False)):
		io.write_npz(dict(func='plot_clt', data=data, xlabel=xlabel, ylabel=ylabel, dist_metric=dist_metric, title=title, **kwargs), fname)
	
	handle_annot(fig, annotator, annotation)
	if (MON):
		plt.show()
	else:
		plt.savefig(fname+'.%s'%fmt, format=fmt)
	plt.close()


def plot_fig(fig, fname='fig', fmt='png', style=None):
	global MON
	if (style is not None):
		plt.style.use(style)
	if (MON):
		fig.show()
	else:
		fig.savefig(fname+'.%s'%fmt, format=fmt)
		
		
def plot_data(data, fname='fig', fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, **kwargs):
	global MON
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)
	func = globals()[data['func'].item()]
	params = dict(data)
	del params['func']
	params.update(kwargs)
	for k, v in params.iteritems():
		if (type(v) is np.ndarray and len(v.shape) == 0):
			params[k] = v.item()
	func(fname=fname, fmt=fmt, style=style, ref_lines=ref_lines, plot_cfg=plot_cfg, annotator=annotator, annotation=annotation, **params)
	
	
def plot_files(fpaths, fmt='png', style=None, ref_lines={}, plot_cfg={}, annotator=None, annotation={}, **kwargs):
	fmt = plot_cfg.setdefault('fmt', fmt)
	style = plot_cfg.setdefault('style', style)
	if (type(fpaths) is not list):
		fpaths = [fpaths]
	for fpath in fpaths:
		fname, fext = os.path.splitext(fpath)
		if (fext == '.pkl'):
			fig = io.read_obj(fpath)
			plot_fig(fig, fname, fmt=fmt, style=style)
		elif (fext == '.npz'):
			data = io.read_npz(fpath)
			plot_data(data, fname=fname, fmt=fmt, style=style, ref_lines=ref_lines, plot_cfg=plot_cfg, annotator=annotator, annotation=annotation, **kwargs)
	

def main():
	pass


if __name__ == '__main__':
	main()