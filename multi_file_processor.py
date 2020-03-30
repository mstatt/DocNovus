def process_files(x):
    # -*- coding: utf-8 -*-
    """
    Created on Tue Dec 13 01:25:12 2017
    Last Update on March 30 11:29:03 2020
    Complete document analysis:
    1) Fuzzy String compare for file similarity
    2) File Summarization
    3) Word frequency counter
    4) Phrase frequency counter
    5) File Sentiment Analyzer
    6) Name Entity Extraction
    7) Sentence Frequency counter
    ************************************************************************
    ## Instructions
    ##---------------------------------------------------
    Prior to running ensure all .txt files are UTF-8 encoded

    ## Run these prior to running DocNovus:
    ##-------------------------------------------------------------------------
    curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
    sha256sum Anaconda3-2019.03-Linux-x86_64.sh
    bash Anaconda3-2019.03-Linux-x86_64.sh
    sudo apt install -y python3-pip
    conda update --all
    source ~/.bashrc
    conda update --all
    conda install -c anaconda ipython
    conda install -c anaconda jupyter
    conda install -c anaconda pandas
    conda install -c anaconda seaborn
    conda install -c anaconda spyder
    conda install -c anaconda tensorflow
    conda install -c conda-forge spacy
    conda install -c plotly plotly-orca psutil requests
    conda install -c conda-forge gensim
    conda install -c conda-forge pyteaser
    conda install -c conda-forge beautifulsoup4
    conda install -c conda-forge poppler
    conda install -c conda-forge textblob
    conda install -c conda-forge pytesseract
    conda install -c bioconda wkhtmltopdf
    conda install -c libgcc
    conda install -c conda-forge python-docx
    conda install -c conda-forge pdftotext
    conda update --all
    python -m nltk.downloader stopwords
    sudo apt-get install -y libpoppler-cpp-dev
    sudo pip install --upgrade pip
    sudo pip install wheel
    sudo pip install pandas
    sudo pip install nltk
    sudo pip install textract
    sudo pip install pdfkit
    sudo pip install pdftotext
    sudo pip install --upgrade tfBinaryURL
    sudo pip install pyteaser 
    sudo pip install fuzzy
    sudo pip install python-docx
    sudo pip install pdftotext
    sudo pip install --pre python-docx
    sudo pip install pypdf2
    sudo apt-get update
    sudo apt-get install build-essential libpoppler-cpp-dev pkg-config python-dev
    sudo apt-get install -y python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev
    pip install pdfkit
    pip install pdftotext
    sudo pip install nltk
    sudo pip install vaderSentiment
    sudo pip install --upgrade nltk
    sudo pip install --upgrade keras
    
    For Linux:
    >> wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.3/wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
    >> tar vxf wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
    >> cp wkhtmltox/bin/wk* /usr/local/bin/
    ************************************************************************
    Inital Run Uncomment the following lines (128-131) to download the NLTK resources, 
    After the 1st run you can remove or commnt them out.
    ************************************************************************
    @author: MStattelman
    """
    print("Loading libraries......")

    #Imports

    import re
    import os
    import sys
    import time
    import uuid
    import glob
    import nltk
    import heapq
    import shutil
    import spacy
    import string
    import codecs
    import pdfkit
    import nltk.data
    import difflib
    import itertools
    import subprocess
    import collections
    import numpy as np
    import pandas as pd
    from math import log
    from nltk import ngrams
    from functools import reduce
    from nltk import sent_tokenize
    from collections import Counter
    from statistics import mean, stdev
    from tfidf_processor import tfidf_calc
    from mdfp_textsummary import getsummary
    from entities import getnames
    from word_counter import word_count
    from nltk.tokenize import sent_tokenize
    from datetime import date, datetime
    # nltk.download('words')
    # nltk.download('punkt')
    # nltk.download('maxent_ne_chunker')
    # nltk.download('averaged_perceptron_tagger')

    #Initialize spacy
    nlp = spacy.load('en_core_web_sm')
    stopwords = nltk.corpus.stopwords.words('english')




    from nltk.tokenize import word_tokenize
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    #Use tweet tokenizer to prevent contracted words from spliting
    from nltk.tokenize import TweetTokenizer

    print("Starting File Process  **********************************")

    #--------------Set up directories and Variables
    #initialize tokenizer
    #Run tokenizer
    tknzr = TweetTokenizer()
    #Initialize Sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    #Set start time to calculate time of processing
    start = time.time()
    #Set file extension for specific filetypes
    fileext = '.txt'
    #Set directory of files for processing
    compdir  = x + '/' +'forprocessing/'

    #Create a output directory based on a UID
    gui = x
    outdir = gui +'/'+ 'Analysis-Complete/'
    errdir = gui + '/Analysis-Complete/'


    print("Created job " + gui)
    filedump = outdir+"inputfiles/"
    if not os.path.exists(filedump):
        os.makedirs(filedump)
    #--------------Set up files and filenames
    tfidfhtml = "tfidf.html"
    entityhtml = "entities.html"
    sentimenthtml = "sentiment.html"
    wordfrequencyhtml = "wordfreq.html"
    filesimilarityhtml = "similarity.html"
    phrasefrequencyhtml = "phrasefreq.html"
    phrasefrequencyhtml3 = "phrasefreq3.html"
    phrasefrequencyhtml4 = "phrasefreq4.html"
    sentencefrequencyhtml = "sentences.html"
    #-------------------Other Variables
    ##Variable Declaration
    phrase_len = 2
    phrase_len3 = 3
    phrase_len4 = 4
    term_len = 1
    comparecount = 0

    corp_url = 'https://highorderanalytics.com'


    #----------------Start Function Definitions------------------------------------------------
    def geo_mean_calc(n):
        """
        Calculate the Geomean
        """
        error_msg(errdir,"Calculating the Geometric Mean  **********************************")
        print("Calculating the Geometric Mean  **********************************")
        geomean = lambda n: reduce(lambda x,y: x*y, n) ** (1.0 / len(n))
        return geomean(n)

    #------------------------------------------------------------------------------------------
    def compareEach(x,y):
        """
        Compare the 2 files passed in using fuzzy string compare
        """
        with open(compdir + x, 'r') as myfile:
            data=myfile.read().replace('\n', '').lower()
            myfile.close()
        with open(compdir + y, 'r') as myfile2:
            data2=myfile2.read().replace('\n', '').lower()
            myfile2.close()

        return difflib.SequenceMatcher(None, data, data2).ratio()

    #------------------------------------------------------------------------------------------
    def remove_punctuation(text):
        # Removes all punctuation and conotation from the string and returns a 'plain' string
        punctuation2 = '-&'+'^®©™€â´‚³©¥ã¼•ž®è±äüöž!@#Â“§$%^*()î_+€$=¿{”}[]:«;"»\â¢â€|<>šË,.?/~`0123456789\n'
        for sign in punctuation2:
            text = text.replace(sign, " ").lower()
        return text

    #------------------------------------------------------------------------------------------
    def error_msg(x1,msg):
        start = time.time()
        start_time = datetime.now()
        now = time.ctime(int(start))
        tmstp = str(datetime.today().strftime("%m-%d-%Y"))
        #----------Write to existing log file ---------------------------------------------
        with open (x1 + "error_process_log.txt","a")as err1:
            err1.write(now + " | " + msg + "\n")
        err1.close()

    def getpercent(num,num2):
        return (num/num2)*100
    #----------------END Function Definitions------------------------------------------------
    #----------------Get List of uploaded files------------------------------------------------
    os.chdir(gui)
    uploaded_files  = list(filter(os.path.isfile, os.listdir(os.curdir)))
    os.chdir('../..')
    #------------------------------------------------------------------------------------------


    #Get all of the files in the directory into a list
    txt_files = list(filter(lambda x: x.endswith(fileext), os.listdir(compdir)))
    txt_files.sort()
    total_txt_cnt = len(txt_files)
    #------------------------------------------------------------------------------------------


    dict_summ = {}
    file_summ_cnt = len(txt_files)
    z = 1
    print("Starting File Summary  ****************************************")
    error_msg(errdir,"Starting File Summary  ****************************************")
    for file in txt_files:
        dict_summ[str(file)] = getsummary(compdir + file)
        print('File Summary process status:  [%d%%]\r'%getpercent(z,file_summ_cnt), end="")
        z = z +1

    #------------------------------------------------------------------------------------------

    file_word_count = {}
    print("Starting File word count  ****************************************")
    error_msg(errdir,"Starting File word count  ****************************************")
    for file in txt_files:
        file_word_count[str(file)] = word_count(compdir + file)

    #------------------------------------------------------------------------------------------
    names_entities = {}
    ne_cnt =1
    print("Starting Name extraction  ****************************************")
    error_msg(errdir,"Starting Name extraction  ****************************************")
    for file in txt_files:
        names_entities[str(file)] = getnames(compdir + file)
        print('Name Entity extraction process status:  [%d%%]\r'%getpercent(ne_cnt,file_summ_cnt), end="")
        ne_cnt = ne_cnt +1

    dfnames_entities = pd.DataFrame(names_entities, index=[0]) 


    #------------------------------------------------------------------------------------------
    print("Starting File comparison  ****************************************")
    error_msg(errdir,"Starting File comparison  ****************************************")
    if (len(txt_files)> 1):
        #Set up lists for file names and Fuzzy logic calculations
        aList = []
        filesim1 = []
        filesim2 = []
        total_operations = len(txt_files) * len(txt_files)
        bList = []
        qr = 1
        #Loop through each list item and compare it against the other items
        for a, b in itertools.combinations(txt_files, 2):
            comparecount = comparecount + 1
            aList.append("File ["+a+"] <> ["+b+"] has a similarity of ");
            filesim1.append(a)
            filesim2.append(b)
            bList.append(compareEach(a,b));
            print('Compare files process status:  [%d%%]\r'%getpercent(qr,total_operations), end="")
            qr = qr + 1


        #Combine both lists into a corolary dictionary
        d = dict(zip(aList, bList))

        #Save sorted dict as new dictionary from most similar to least
        d1 = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))

        error_msg(errdir,"Writing datafile-comparison.txt file *****************************")

        #Save results to file:
        fo = open(outdir+'datafile-comparison.txt', "w")
        #Print Headers to file
        fo.write('File similarity ranked from most to least similar:\n\n')
        fo.write('----------------Average File Similarity Score----------------'+'\n\n')
        fo.write('Geometric Mean:'+str(geo_mean_calc(bList))+'\n\n')
        fo.write('Arithmatic Mean:'+str(mean(bList))+'\n\n')
        fo.write('-------------------------------------------------------------'+'\n\n')
        #Print Output to file
        for k, v in d1.items():
            fo.write(str(k) + ' >>> '+ str(v) + '\n\n')
        fo.write('-------------------------------------------------------------'+'\n\n')
        fo.close()


    #------------------------------------------------------------------------------------------
    #------------------------------------------------------------------------------------------
    #Set length of word combinations for use in counters.
    d1 = {}
    filesentiment = {}
    sentencesA = []
    corpus = []
    file_list = []
    lc = 1
    error_msg(errdir,"Loading the corpus  **********************************************")
    os.chdir(compdir)

    #Get all files in the directory loaded into the corpus
    for file in glob.glob("*.txt"):
        print("Loading "+str(file)+" into corpus ****************************************")
        file_list.append(file)
        f = open(file)
        txtstream = f.read()
        txtstream2 = re.sub(r'[^\x00-\x7f]',r'', remove_punctuation(txtstream))
        corpus.append(txtstream2)
        filesentiment.update({file : sid.polarity_scores(remove_punctuation(txtstream))})
        f.close()
        print('Loading Corpus process status:  [%d%%]\r'%getpercent(lc,total_txt_cnt), end="")
        lc = lc +1


    #Move files for post processing
    file_list.sort()

    os.chdir('../../..')
    error_msg(errdir,"Corpus loading completed  ****************************************")
    print("Corpus loading completed  ****************************************")
    #******************************Sentence Begins
    error_msg(errdir,"Starting Sentence isolation  into single file*************************************")
    print("Starting Sentence isolation  into single file*************************************")
    os.chdir(compdir)
    for file in file_list:
        with open("All-Combined.txt", 'a') as f:
            f2 = open(file, 'r', errors='ignore').read()
            printable = set(string.printable)
            f.write(''.join(filter(lambda x: x in printable, f2)))
        f.close()



    print("Starting sentence tokenization *************************************")
    f3 = open("All-Combined.txt", 'r')
    slist = sent_tokenize(f3.read())
    f3.close()

    slist1 = [''.join(c for c in s if c not in string.punctuation) for s in slist]
    slist2 = [s for s in slist1 if s]
    #Remove instances of a single character
    slistready = [i for i in slist2 if len(i) > 1]
    #Remove empty items
    slistready = list(filter(None, slistready)) 

    #Remove combined text file
    if os.path.exists("All-Combined.txt"):
        os.remove("All-Combined.txt")
    #******************************Spacy Sentence Ends

    #******************************Very Important
    (os.chdir('../../..')) #Change the directory back based of dir tree diferences
    error_msg(errdir,"Sentence tokenization completed *************************************")
    print("Sentence tokenization completed *************************************")
    for files in file_list:
        if files.endswith(".txt"):
            shutil.copyfile(os.path.abspath(compdir+files),os.path.abspath(filedump+files))

    print("Moving files to "+filedump+" and cleaning clean_words **********************************")

    ss = {}
    ss2 = {}
    sentences_list = []
    frequencies0 = Counter([])
    frequencies = Counter([])
    frequencies3 = Counter([])
    frequencies4 = Counter([])
    Sentencecounter = Counter([])
    #totalcp = len(corpus)
    print("Building ngrams and quadgrams  ***********************************")
    error_msg(errdir,"Building ngrams and quadgrams  ***********************************")
    
    
    #Cycle through corpus to generate frequencies metrics
    for text in corpus:
        token = tknzr.tokenize(text.lower())
        #Frequency for words
        #Cleaning up any punctuation
        token0 = [''.join(c for c in s if c not in string.punctuation) for s in token]
        #Cleaning up any blank entries
        token1 = [''.join(c for c in s if c not in string.punctuation) for s in token0]
        tokenready1 = [s for s in token1 if s]
        #Remove instances of a single character
        tokenready = [i for i in tokenready1 if len(i) > 1]
        #Create single word NGRAM
        single = list(ngrams(list(filter(lambda x: x not in stopwords, tokenready)),term_len))
        #Assign to Counters
        frequencies0 += Counter(single)
        #Frequency for phrases
        doublegrams = list(ngrams(list(filter(lambda x: x not in stopwords, tokenready)),phrase_len))

        frequencies += Counter(doublegrams)
        triplegrams = list(ngrams(list(filter(lambda x: x not in stopwords, tokenready)),phrase_len3))

        frequencies3 += Counter(triplegrams)
        quadgrams = list(ngrams(list(filter(lambda x: x not in stopwords, tokenready)),phrase_len4))

        frequencies4 += Counter(quadgrams)



    error_msg(errdir,"Building ngrams and quadgrams completed  *************************")
    #---------------------Start of Sentence Analysis

    print("Assigning sentences to counter  ***********************************")
    error_msg(errdir,"Assigning sentences to counter  **********************************")
    Sentencecounter = Counter(slistready)
    error_msg(errdir,"Sentence assignment has completed  *******************************")
    print("Sentence assignment has completed  *******************************")
    #-----------------------------------------------
        #---------------------Start of Email Analysis
    # Find all email addresses
    error_msg(errdir,"Starting email isolation  ****************************************")
    print("Starting email isolation  *******************************")
    email = []
    for line in slistready:
        email.append(re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", line))

    email = [y for x in email for y in x]
    error_msg(errdir,"Email isolation completed ****************************************")
    print("Email isolation completed  *******************************")


    # #---------------------Start of URL Analysis
    # #Locate any URls
    error_msg(errdir,"Starting URL isolation  ******************************************")
    urls = []
    for line in slistready:
        urls.append(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', line))

    urls = [y for x in urls for y in x]
    error_msg(errdir,"URL isolation completed ******************************************")
    print("URL isolation completed  *******************************")
    #-----------------------------------------------

    error_msg(errdir,"Starting dictionary sorting **************************************")
    print("Starting dictionary sorting  *******************************")
    #Sort the dictionaries
    if (len(txt_files)> 1):
        odsimilar = collections.OrderedDict(d1.items())

    ods = collections.OrderedDict(filesentiment.items())
    od0 = collections.OrderedDict(frequencies0.most_common())
    od = collections.OrderedDict(frequencies.most_common())
    od3 = collections.OrderedDict(frequencies3.most_common())
    od4 = collections.OrderedDict(frequencies4.most_common())
    odsumm = collections.OrderedDict(dict_summ.items())
    SentenceOD = collections.OrderedDict(Sentencecounter.most_common())
    filewordcount = collections.OrderedDict(file_word_count.items())
    odnames = collections.OrderedDict(names_entities.items())



    error_msg(errdir,"Dictionary sorting completed *************************************")
    print("Dictionary sorting completed  *******************************")

    #Build dataframes
    print("Building dataframes  *********************************************")
    error_msg(errdir,"Building dataframes  *********************************************")
    if (len(txt_files)> 1):
        #Create output for fuzzy string compare as dataframe
        dffilecomp = pd.DataFrame(list(zip(filesim1, filesim2, bList)),
                      columns=['File #1','File #2', 'Similarity'])
        pd.set_option('display.max_colwidth',100)
        dffilecomp.sort_values(["Similarity"], inplace=True, ascending=False)
        dffilecomp.index = pd.RangeIndex(len(dffilecomp.index))



    #Create output for entities dataframe
    dffileentities = pd.DataFrame.from_dict(odnames, orient='index').reset_index()
    pd.set_option('display.max_colwidth', 100)
    dffileentities.style.set_table_styles([dict(selector="th",props=[('max-width', '50px')])])
    dffileentities = dffileentities.rename(columns={'index':'Filename', 0:'Named Entities in the file'})
    dffileentities = dffileentities.sort_values(by ='Filename')


    #Create output for word count dataframe
    dffilewordcount = pd.DataFrame.from_dict(filewordcount, orient='index').reset_index()
    dffilewordcount = dffilewordcount.rename(columns={'index':'Filename', 0:'Number_of_words'})
    dffilewordcount = dffilewordcount.sort_values(by ='Number_of_words', ascending=False)


    #Create output for summary dataframe
    dffilesummary = pd.DataFrame.from_dict(odsumm, orient='index').reset_index()
    dffilesummary = dffilesummary.rename(columns={'index':'File', 0:'Summary'})
    dffilesummary = dffilesummary.sort_values(by ='Summary' )


    #Create output for word frequency dataframe
    dfwordfreq = pd.DataFrame.from_dict(od0, orient='index').reset_index()
    dfwordfreq = dfwordfreq.rename(columns={'index':'Word', 0:'Count'})
    dfwordfreq = dfwordfreq[dfwordfreq.Count != 1]

    #Create output for Phrase frequency as dataframe
    dfphrasefreq = pd.DataFrame.from_dict(od, orient='index').reset_index()
    dfphrasefreq = dfphrasefreq.rename(columns={'index':'Phrase', 0:'Count'})
    dfphrasefreq = dfphrasefreq[dfphrasefreq.Count != 1]


    #Create output for Phrase frequency as dataframe
    dfphrasefreq3 = pd.DataFrame.from_dict(od3, orient='index').reset_index()
    dfphrasefreq3 = dfphrasefreq3.rename(columns={'index':'Phrase', 0:'Count'})
    dfphrasefreq3 = dfphrasefreq3[dfphrasefreq3.Count != 1]


    #Create output for Phrase frequency as dataframe
    dfphrasefreq4 = pd.DataFrame.from_dict(od4, orient='index').reset_index()
    dfphrasefreq4 = dfphrasefreq4.rename(columns={'index':'Phrase', 0:'Count'})
    dfphrasefreq4 = dfphrasefreq4[dfphrasefreq4.Count != 1]

    #Create output for Sentence frequency as dataframe
    dfSentences = pd.DataFrame.from_dict(SentenceOD, orient='index').reset_index()
    dfSentences = dfSentences.rename(columns={'index':'Sentence', 0:'Count'})
    dfSentences = dfSentences[dfSentences.Count != 1]
    filter2 = dfSentences["Sentence"] != "."
    dfSentences = dfSentences[filter2]


    #Create Sentiment Dataframe
    dfSentiment = pd.DataFrame.from_dict(ods, orient='index').reset_index()
    dfSentiment = dfSentiment.rename(columns={'index':'Files Analyzed', 0:'neg', 0:'neu', 0:'pos', 0:'compound'})
    dfSentiment.sort_values(["pos"], inplace=True, ascending=False)
    dfSentiment.at['Averages', 'neg'] = dfSentiment['neg'].mean()
    dfSentiment.at['Averages', 'neu'] = dfSentiment['neu'].mean()
    dfSentiment.at['Averages', 'pos'] = dfSentiment['pos'].mean()
    SentScore = dfSentiment['pos'].mean() + dfSentiment['neg'].mean()
    dfSentiment.at['Averages', 'compound'] = SentScore

    dfemail = pd.DataFrame({'Email Addresses':email})
    dfurl = pd.DataFrame({'Website Urls':urls})

    error_msg(errdir,"Getting counts for words, phrases and sentences ******************")
    print("Getting counts for words, phrases and sentences ******************")
    #Get a count of all words and phrases
    Count_Words=dfwordfreq.shape[0]
    Count_Phrase=dfphrasefreq.shape[0]
    Count_Sentences = dfSentences.shape[0]

    error_msg(errdir,"Dataframes build completed ***************************************")
    print("Dataframes build completed ***************************************")

    #Generate html files from dataframes
    print("Building web pages from dataframes *******************************")
    error_msg(errdir,"Building web pages from dataframes *******************************")


    dfwordfreq.to_html(open(outdir + wordfrequencyhtml, 'a'))
    dfphrasefreq.to_html(open(outdir + phrasefrequencyhtml, 'a'))
    dfphrasefreq3.to_html(open(outdir + phrasefrequencyhtml3, 'a'))
    dfphrasefreq4.to_html(open(outdir + phrasefrequencyhtml4, 'a'))
    dfSentences.to_html(open(outdir + sentencefrequencyhtml, 'a'))
    dfSentiment.to_html(open(outdir + sentimenthtml, 'w'))
    dfemail.to_html(open(outdir + 'email.html', 'w'))
    dfurl.to_html(open(outdir + 'url.html', 'w'))
    dffilesummary.to_html(open(outdir + 'dfsummary.html', 'w'))
    dffilewordcount.to_html(open(outdir + 'dffilewordcount.html', 'w'))
    dffileentities.to_html(open(outdir + 'dfNameEntities.html', 'w'))
    dfnames_entities.to_html(open(outdir + 'dfNameEntities2.html', 'w'))
    if (len(txt_files)> 1):
        dffilecomp.to_html(open(outdir + filesimilarityhtml, 'w'))


    error_msg(errdir,"Web pages from dataframes completed ******************************")
    print("Web pages from dataframes completed ******************************")

    #---------------------------Start TFIDF processing
    #--------------------------------------------------------------------------------------
    tfidf_calc(x)
    #--------------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------------


    error_msg(errdir,"Building Web pages for pdf structure *****************************")
    print("Building Web pages for pdf structure *****************************")

    #Write File list to File
    dir_name = gui.replace("job_queue/", "")
    with open (outdir+"index.html","a")as fp1:
       fp1.write("<!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 10%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;}</style></head><body><center>")
       fp1.write("<br/><br/><br/><table><tr><td class=section colspan=2> ")
       fp1.write("<h1>Document Analysis for task: " +dir_name+ "</h1>")
       fp1.write("</td></tr><tr><td colspan=2 align=center>")
       fp1.write("<h3>Contents: #1: Individual File Summary" + "<br/>")
       if (len(txt_files)> 2):
        fp1.write("Contents: #2: Comparison of the File similarities" + "<br/>")

       fp1.write("Contents: #3: TF-IDF" + "<br/>")
       fp1.write("Contents: #4: Email addresses" + "<br/>")
       fp1.write("Contents: #5: File Word Count" + "<br/>")
       fp1.write("Contents: #6: Website Urls" + "<br/>")
       fp1.write("Contents: #7: Word Frequency" + "<br/>")
       fp1.write("Contents: #8A (2) Word Phrase Frequency: " + "<br/>")
       fp1.write("Contents: #8B (3) Word Phrase Frequency: " + "<br/>")
       fp1.write("Contents: #9: Identified File Entities" + "<br/>")
       #fp1.write("Contents: #8C (4) Word Phrase Frequency: " + "<br/>")
       fp1.write("Contents: #10 Sentence Frequency: " + "<br/>")
       fp1.write("Contents: #11 Overall Content Sentiment Analysis: " + "</h3><br/><br/>")
       fp1.write("</td></tr><tr><td bgcolor='#ffff99' width='50%'>")
       fp1.write("<h3><u>Breakdown: SUMMARY:</u> " + "<br/><br/>")
       fp1.write("Execution time: {0:.5}".format(time.time() - start)+"s<br/>")
       fp1.write("With a total Word count of: "+str(Count_Words)+"<br/>")
       fp1.write("With a total Phrase count of: "+str(Count_Phrase)+"<br/>")
       fp1.write("With a total duplicate Sentence count of: "+str(Count_Sentences)+"<br/>")
       #fp1.write("With an overall Sentiment score of: {0:.3}".format(SentScore)+"</h3>")
       fp1.write("<u>Independent Analysis links</u><br/>")
       fp1.write("<ul>")
       fp1.write("<li><a href="+"dfsummary.html"+">Invividual File Summary</a></li>")
       fp1.write("<li><a href="+"datafile-comparison.txt"+">Comparison of the File similarities</a></li>")
       fp1.write("<li><a href="+"email.html"+">Email Extract</a></li>")
       fp1.write("<li><a href="+"dffilewordcount.html"+">File Word Count</a></li>")

       fp1.write("<li><a href="+"url.html"+">Website Address Extraction</a></li>")
       fp1.write("<li><a href="+tfidfhtml+">TF-IDF Calculation</a></li>")
       fp1.write("<li><a href="+wordfrequencyhtml+">Word Frequency</a></li>")
       fp1.write("<li><a href="+phrasefrequencyhtml+">(2) Word Phrase Frequency</a></li>")
       fp1.write("<li><a href="+phrasefrequencyhtml3+">(3) WordPhrase Frequency</a></li>")
       fp1.write("<li><a href="+"dfNameEntities.html"+">Identified Entities in Files</a></li>")
       #fp1.write("<li><a href="+phrasefrequencyhtml4+">(4) WordPhrase Frequency</a></li>")
       fp1.write("<li><a href="+sentencefrequencyhtml+">Sentence Frequency</a></li>")
       fp1.write("<li><a href="+sentimenthtml+">Overall Content Sentiment Analysis</a></li>")
       fp1.write("<li><a href="+"dfNameEntities.html"+">Identified Entities in Files</a></li>")

       fp1.write("</ul>")
       fp1.write("</td><td width='50%'>")
       fp1.write("<u>The following files ("+str(len(file_list))+") were processed:</u><br/><ul>")
       for line in uploaded_files:
           fp1.write("<li>"+line+"</li>")
       fp1.write("</ul></td></tr></table>")
       fp1.write("</center></body></html>")
       fp1.close()


    with open (outdir+"s10.html","a")as fp1:
       fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 15%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
       fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #1:</u><br/>Individual File Summary</h1></div></center></body></html>")
       fp1.close()


    if (len(txt_files)> 2):
        with open (outdir+"s1.html","a")as fp1:
            fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 10%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
            fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #2:</u><br/>Comparison of the File similarities</h1></div><p>Typically, the reason for a one-to-one file comparison is necessary for identifying duplicated documents that have name differencess but the contents are identical. The data below will show the overall similarity averages and the one to one comparison.</p><p>The ratings are as follows: <br/> A rating of 1.0 is a perfect match. As a rule of thumb, a rating over 0.6 means the documents are close matches. A rating of 0.001, you can be quite certain that the contents of the files are vastly different. The lower the score the less similarities there are between the comparison files.</p><p>Based on the number of files, the system executed "+ str(comparecount) +" file comparrisons.</p></center></body></html>")
            fp1.close()


    with open (outdir+"s1a.html","a")as fp1:
       fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 12%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
       fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #3:</u><br/>TF-IDF Calculation</h1></div><p>Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.</p><ul><li><p>TF: Term Frequency, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization:</p><p>TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).</p></li><li><p>IDF: Inverse Document Frequency, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as is, of, and that, may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following:</p><p>IDF(t) = log_e(Total number of documents / Number of documents with term t in it).</p></li><br/></center></body></html>")
       fp1.close()

    with open (outdir+"s1b.html","a")as fp1:
      fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 15%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
      fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #4:</u><br/>Email Addresses</h1></div><p>")
      fp1.write(str(len(email)) + " email addresses found in " + str(len(file_list))+ " total files. </p></center></body></html>")
      fp1.close()

    with open (outdir+"fwc.html","a")as fp1:
       fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 12%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
       fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #5:</u><br/>File Word Count</h1></div><p>The purpose of identifying the number of words in a file lends itself to discerning the depth and type of document. Whether it is taken from a blog post, email, online news article, more robust documentation, etc.</p></center></body></html>")
       fp1.close()

    with open (outdir+"s1d.html","a")as fp1:
       fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 15%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
       fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #6:</u><br/>Website Addresses</h1></div>")
       fp1.write(str(len(urls)) + " website url's found in " + str(len(file_list))+ " total files. </p></center></body></html>")
       fp1.close()

    with open (outdir+"s2.html","a")as fp1:
       fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 15%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
       fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #7:</u><br/>Word Frequency</h1></div></center></body></html>")
       fp1.close()

    with open (outdir+"s3.html","a")as fp1:
       fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 15%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
       fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #8A:</u><br/>2 Word Phrase Frequency</h1></div></center></body></html>")
       fp1.close()

    with open (outdir+"pf3.html","a")as fp1:
       fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 15%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
       fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #8B:</u><br/>3 Word Phrase Frequency</h1></div></center></body></html>")
       fp1.close()

    with open (outdir+"ide.html","a")as fp1:
       fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 15%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
       fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #9:</u><br/>Identified File Entities</h1></div><p>Named entity recognition (NER) is a key process in information extraction that seeks to locate and classify named entities in text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc. </p><p>NER is used in many fields in Natural Language Processing (NLP), and it can help answering many real-world questions, such as: Which companies were mentioned in the news article? Were specified products mentioned in complaints or reviews? Does the tweet contain the name of a person? Does the tweet contain this person’s location? </p></center></body></html>")
       fp1.close()


    with open (outdir+"s4.html","a")as fp1:
       fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 15%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
       fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #10:</u><br/>Sentence Frequency</h1></div>")
       fp1.write(str(Count_Sentences) + " sentences were repeated across " + str(len(file_list))+ " total files. </p></center></body></html>")
       fp1.close()

    with open (outdir+"s5.html","a")as fp1:
       fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 15%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
       fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #11:</u><br/>Overall Content Sentiment Analysis</h1></div><p> In the table below, the Files are sorted in order from most to least positive. </p><p>The score indicates how negative or positive the overall document analyzed is. The closer the score is to 1.0 in the neg column should be viewed as absolutley negative and anything closer to 1.0 in the pos column should be viewed as absolutely positive. The total score can never exceed 1.0 in either Positive, Negative or Neutral catergories. The neutral column is provided as a balance metric if the document is neither positive or negative.</p></center></body></html>")
       fp1.close()


    with open (outdir+"s7.html","a")as fp1:
       fp1.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>" +dir_name+ "</title><style> .section{width: 700px;height: 200px;z-index: 15;background:#33FFEC;text-align: center;}#watermark {  opacity: 0.20;    filter: alpha(opacity=20);   position: absolute;  width: 100%;  height: 100%;  top: 15%;  left : 40%;  z-index: 999999;     background-image: url('logo.png');     background-repeat: no-repeat; text-align: left;vertical-align: text-bottom;}</style></head><body><center>")
       fp1.write("<center><div id=watermark></div><div class='section'><br/><br/><h1><u>Section #12:</u><br/> Summarization </h1><br/><br/><br/><br/><br/><br/><br/><br/><p>This analysis has been done through the use of various Natural Language Processing, Fuzzy Logic, Geometric Mean, Text Summarization, Term Frequency/Inverse Document Frequency and Other Statistical formulas. The results are based soley of the information in the documents provided.<br/> If you have any questions or would like additional features, please contact "+ corp_url +" </p><p>Feel free to request this service for you Document Analysis needs. This can be used for  Venture Capital Investment Research, Content Weapons Utilization Analysis, Private Equity Investment Analysis, Competitive Intelligence Analysis, Research Aggregation Analysis, Legal Document Discovery Analysis, Pharmaceutical Testing Analysis, Political Campaigns, Marketing Effectiveness Research, Content Generation Analysis across various forms of media and much more.</p></div></center></body></html>")
       fp1.close()


    error_msg(errdir,"Web pages build for pdf structure completed***********************")
    #Generate Analysis pdf form files collection
    print("Generating Task-"+dir_name+"-Document-Analysis.pdf.....")
    error_msg(errdir,"Generating Task-"+dir_name+"-Document-Analysis.pdf  **************")

    options = {
        'page-size': 'Letter',
        'margin-top': '0.9in',
        'margin-right': '0.9in',
        'margin-bottom': '0.9in',
        'margin-left': '0.9in',
        'encoding': "UTF-8",
        'dpi': 200,
        'header-center': 'DocNovus file analysis for : '+dir_name,
        'footer-line':'',
        'header-line':'',
        'footer-right': '[page] of [topage]',
        'footer-left': corp_url,
    }


    if (len(txt_files)> 2):
        pdfkit.from_file([outdir+"index.html",outdir+"s10.html",outdir+'dfsummary.html',outdir+'s1.html' , outdir + 'similarity.html' , outdir + 's1a.html' , outdir + tfidfhtml , outdir + 's1b.html', outdir + 'email.html' , outdir + 'fwc.html', outdir + 'dffilewordcount.html', outdir + 's1d.html',outdir + 'url.html',outdir +'s2.html',outdir + wordfrequencyhtml,outdir+'s3.html',outdir + phrasefrequencyhtml,outdir+'pf3.html',outdir + phrasefrequencyhtml3, outdir+'ide.html',outdir + 'dfNameEntities.html',outdir+'s4.html',outdir + sentencefrequencyhtml,outdir+'s5.html',outdir + sentimenthtml,outdir+'s7.html'], outdir + dir_name + '-Document-Analysis.pdf', options=options)
    else:
        pdfkit.from_file([outdir+"index.html",outdir+'s1a.html',outdir + tfidfhtml,outdir+'s1b.html',outdir + 'email.html',outdir+'fwc.html',outdir + 'dffilewordcount.html',outdir + 's1d.html',outdir + 'url.html',outdir +'s2.html',outdir + wordfrequencyhtml,outdir+'s3.html',outdir + phrasefrequencyhtml,outdir+'pf3.html',outdir + phrasefrequencyhtml3,outdir+'ide.html',outdir + 'dfNameEntities.html',outdir+'s4.html',outdir + sentencefrequencyhtml,outdir+'s5.html',outdir + sentimenthtml,outdir+'s7.html'], outdir + dir_name + '-Document-Analysis.pdf', options=options)



    print("Cleaning up process files  ******************************************")
    error_msg(errdir,"Cleaning up process files  ******************************************")

    os.remove(outdir +'datafile-comparison.txt')

    print('File processing has completed for Task-'+ dir_name+' *************************')
    error_msg(errdir,"File processing has completed for Task-"+ dir_name+"  *************")
