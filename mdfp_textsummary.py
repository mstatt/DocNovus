#!/usr/bin/env python
# coding: utf-8
def getsummary(inputfilename):
	
	import re
	import os
	import nltk
	import heapq
	import spacy

	article_text = ""
	summary = ' '

	file = open(inputfilename, "r")
	filedata = str(file.readlines())
	filedata2 = re.sub(r'[^\x00-\x7f]',r'', filedata)

	index = str(inputfilename).find("csv")
	if (index > -1):
		paragraphs = filedata2.split(". ")
		paragraphs2 = [x for x in paragraphs if x]
	else:
		paragraphs = filedata2.split("\n")
		paragraphs2 = [x for x in paragraphs if x]
	


	for p in paragraphs2:
	    article_text += str(p)


	# Removing Square Brackets and Extra Spaces
	article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
	article_text = re.sub(r'\s+', ' ', article_text)

	# Removing special characters and digits
	formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
	formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

	sentence_list = nltk.sent_tokenize(article_text)

	stopwords = nltk.corpus.stopwords.words('english')

	word_frequencies = {}
	for word in nltk.word_tokenize(formatted_article_text):
	    if word not in stopwords:
	        if word not in word_frequencies.keys():
	            word_frequencies[word] = 1
	        else:
	            word_frequencies[word] += 1

	if (len(word_frequencies.values()) > 2):
		maximum_frequncy = max(word_frequencies.values())

		for word in word_frequencies.keys():
		    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

		sentence_scores = {}
		for sent in sentence_list:
		    for word in nltk.word_tokenize(sent.lower()):
		        if word in word_frequencies.keys():
		            if len(sent.split(' ')) < 30:
		                if sent not in sentence_scores.keys():
		                    sentence_scores[sent] = word_frequencies[word]
		                else:
		                    sentence_scores[sent] += word_frequencies[word]


		summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)

		#summary = ' '.join(summary_sentences)
		summary = ' '.join(summary_sentences)
		#summary = ' '.join(filter(lambda x: x in printable, summary_sentences))

	if (summary.strip() == ''):
		#summary = getsummary2(inputfilename)
		summary = '. '.join(paragraphs[:4])
		summary = summary.replace("[","").replace("]","")
		#summary = 'Zero: No summary available for this file.'
		

	return summary