#!/usr/bin/env python
# -*- coding=utf-8 -*-
###########################################################################
# Copyright (C) 2013-2016 by Caspar. All rights reserved.
# File Name: xmlextrc.py
# Author: Shankai Yan
# E-mail: sk.yan@my.cityu.edu.hk
# Created Time: 2016-03-01 23:01:04
###########################################################################
#

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XMLParser, tostring, ParseError


def get_parser(builder, encoding='utf-8', **kwargs):
	return XMLParser(target=builder, encoding=encoding, **kwargs)


def extrc_list(listhead_path, elem_tag, content_path, xml_tree=None, xml_str='', ret_type='text'):
	if (xml_tree is None):
		try:
			xml_tree = ET.fromstring(xml_str.encode('utf-8'))
		except ParseError as e:
			print(e)
			print(xml_str.encode('utf-8').strip())
	if (xml_tree.tag == listhead_path):
		list_root = xml_tree
	else:
		list_root = xml_tree.find(listhead_path)
	elem_list = []
	for elem in list_root.iterfind(elem_tag):
		if (ret_type == 'elem'):
			elem_list.append(elem.find(content_path))
		elif (ret_type == 'xml'):
			elem_list.append(tostring(elem.find(content_path)))
		elif (ret_type == 'text'):
			elem_list.append(elem.find(content_path).text)
	return elem_list
