# -*- coding: utf-8 -*-
def tfidf_calc(x):
    import os
    import glob
    import shutil
    import math
    import nltk
    #nltk.download('stopwords')
    from textblob import TextBlob as tb
    from nltk.corpus import stopwords
    #Imported Modules
    from error_log_writer import error_msg

    cachedStopWords = stopwords.words("english")

    compdir =  x + '/' 'forprocessing/'
    outdir = x + "/Analysis-Complete/"
    numofwords = 20

    def getpercent(num,num2):
        return (num/num2)*100

    def tf(word, blob):
        return blob.words.count(word) / len(blob.words)

    def n_containing(word, bloblist):
        return sum(1 for blob in bloblist if word in blob.words)

    def idf(word, bloblist):
        return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

    def tfidf(word, blob, bloblist):
        return tf(word, blob) * idf(word, bloblist)

    number_of_files = str(len([item for item in os.listdir(compdir) if os.path.isfile(os.path.join(compdir, item))]))
    print("Processing ("+ number_of_files + ") files for TFIDF.....")
    error_msg(x,"Processing ("+ number_of_files + ") files for TFIDF  *********************************")
    bloblist = []
    filename_listtf = []
    print("Building list and stream for TFIDF calculation......")
    error_msg(x,"Building list and stream for TFIDF calculation  ******************")
    for filename2tf in sorted(glob.glob(compdir+"*.txt")):
        with open(filename2tf, 'r') as myfile2tf:
            initstream = "".join(line.rstrip() for line in myfile2tf).upper()
            txtstreamtf =tb(" ".join([word for word in initstream.split() if word not in cachedStopWords]))
            filename_listtf.append(os.path.basename(filename2tf))
            bloblist.append(txtstreamtf)
            myfile2tf.close()


    print("Writing web page for TFIDF calculation......")
    error_msg(x,"Writing web page for TFIDF calculation  **************************")
    with open (outdir+"tfidf.html","a",encoding="utf-8")as fp1tfidf:
        fp1tfidf.write("<!DOCTYPE html><html><!DOCTYPE html><html lang='en'><head><title>TF/IDF Calculation</title></head><body>")
        #fp1tfidf.write("<h1><u>TF/IDF:</u></h1>")
        fp1tfidf.write("<table border=1>")
        total = len(bloblist)
        for i, blob in enumerate(bloblist):
            fp1tfidf.write("<tr><td colspan=2>")
            fp1tfidf.write("Top words "+ str(numofwords) +" in document {}".format(filename_listtf[i]))
            fp1tfidf.write("</td></tr>")
            #print("Running TFIDF for " + filename_listtf[i] + " *******************************")
            print('Process status:  [%d%%]\r'%getpercent(i,total), end="")
            scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:numofwords]:
                fp1tfidf.write("<tr><td>")
                fp1tfidf.write("\tWord: {}</td><td> TF-IDF: {} </td></tr>".format(word, round(score, 5)))
            fp1tfidf.write("</tr>")

        fp1tfidf.write("</table></body></html>")
    fp1tfidf.close()
    print("TF/IDF calculation completed on ("+ number_of_files + ") files******************************")
    error_msg(x,"TF/IDF calculation completed on ("+ number_of_files + ") files  **********************")
