from setuptools import setup, find_packages

with open('README.md', 'r') as fd:
	long_description = fd.read()

setup(
	name='bionlp',
	version='0.4.0',
	description='Useful modules for biomedical text mining (BioNLP)',
	long_description=long_description,
	long_description_content_type='text/markdown',
	license='Apache-2.0',
	author='Shankai Yan',
	author_email='dr.skyan@gmail.com',
	url='https://github.com/cskyan/bionlp',
	packages=['bionlp'] + ['bionlp.%s' % subp for subp in find_packages('.')],
	package_dir = {'': '..'},
	python_requires=">=3.6",
	install_requires=[
		'scikit-learn>=0.2',
		'nltk>=3.0.0'
	],
	classifiers=[
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: Apache-2.0 License',
		'Operating System :: OS Independent',
	],
)