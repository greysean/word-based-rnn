#!/usr/bin/env python
# coding: utf-8

import os
import re
from nltk import word_tokenize

re_meta = re.compile(r'^#.*\n', re.MULTILINE)
re_tags = re.compile(r'(?i)<(?!quote|\/quote|verse|\/verse).*?>', re.MULTILINE)
re_dquotes = re.compile(r'[”“]', re.MULTILINE)
re_squotes = re.compile(r'[‘’]', re.MULTILINE)
re_links = re.compile(r'^\[\[.*\]\]\n', re.MULTILINE)
re_fnotes = re.compile(r'^\[\d*\].*\n', re.MULTILINE)


def read_files(path):
    '''
    Arguments: filepath (string)
    Returns: array of strings
    '''
    docs = []
    filelist = os.listdir(path)
    for f in filelist:
        file = open(os.path.join(path + f), 'r')
        docs.append(file.read())
    return docs


def remove_metadata(doc):
    # remove lines that begin with markup (e.g., metadata)
    doc = re.sub(re_meta, '', doc)
    return doc

def remove_tags(doc):
    # remove formatting tags
    # grep -rohi "<[^>]*>" texts/ | sort --unique
    doc = re.sub(re_tags, '', doc)
    return doc

def normalize_quotes(doc):
    # normalize quotes (make same ascii character)
    doc = re.sub(re_dquotes, '"', doc)
    doc = re.sub(re_squotes, "'", doc)
    return doc

def misc_clean(doc):
    # remove muse links
    doc = re.sub(re_links, '', doc)

    #remove footnotes
    doc = re.sub(re_fnotes, '', doc)

    return doc

def clean_doc(doc):
    '''
    arguments: doc (string)
    returns: array of strings
    '''

    doc = remove_metadata(doc)
    doc = remove_tags(doc)
    doc = normalize_quotes(doc)
    doc = misc_clean(doc)

    return doc

def clean_and_tokenize_docs(path):
    quotes = 0
    tags = 0
    fnotes = 0
    links = 0

    documents = read_files(path)
    tokenized_docs = []

    for doc in documents:
        quotes += len(re.findall(r'[”“‘’]', doc))
        tags += len(re.findall(re_tags, doc))
        fnotes += len(re.findall(re_fnotes, doc))
        links += len(re.findall(re_links, doc))
        doc = word_tokenize(clean_doc(doc))
        tokenized_docs.extend(doc)

    print("removed %s footnotes." % fnotes)
    print("removed %s tags." % tags)
    print("removed %s image links." % links)
    print("replaced %s irregular quotes." % quotes)

    return tokenized_docs

