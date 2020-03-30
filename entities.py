#!/usr/bin/env python
# coding: utf-8
import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')

string = ""

def extract_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]

def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)

def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def extract_names(document):
    names = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))

    names = list(dict.fromkeys(names))
    final_names = ', '.join(map(str, names))

    return final_names

# if __name__ == '__main__':
#     numbers = extract_phone_numbers(string)
#     emails = extract_email_addresses(string)
#     names = extract_names(string)


def getnames(inputfilename):
    f = open(inputfilename)
    txtstream = f.read()
    strNames = re.sub(r'[^\x00-\x7f]',r'', extract_names(txtstream))

    return strNames

