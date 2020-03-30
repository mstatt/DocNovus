def extract_Text_pdf(xjob):
    #!/usr/bin/env python
    # coding: utf-8
    import glob
    import pdftotext
    import re
    import shutil
    import sys
    import os
    from error_log_writer import error_msg

    print("Starting Text Extraction for pdf files......")
    error_msg(xjob,"Starting Text Extraction for pdf files  **************************")
    compdir2 = xjob + "/" +'forprocessing/'
    pdfdir = xjob + "/" +'pdf/'
    number_of_files = str(len([item for item in os.listdir(pdfdir) if os.path.isfile(os.path.join(pdfdir, item))]))
    print("Processing ("+ number_of_files + ") .pdf files.....")
    error_msg(xjob,"Processing ("+ number_of_files + ") .pdf files **************************************")
    os.chdir(pdfdir)
    file_list2 = []
    for filename in glob.glob("*.pdf"):
        #Get the filename without the extension for nameing later
        base=os.path.basename(filename)
        filenameNoExt = os.path.splitext(base)[0]
        #Create a list of the text files
        file_list2.append("pdf_"+filenameNoExt+".txt")
        with open(filename, "rb") as f:
            pdf = pdftotext.PDF(f)

        filecontents = re.sub(' +', ' ', " ".join(pdf).replace("\n"," ").strip())

        # content_list = list(filter(None, content_list))
        with open ("pdf_"+filenameNoExt+".txt","a")as fp1:
            fp1.write(filecontents)
        fp1.close()

    #******************************Very Important
    os.chdir('../../..') #Change the directory back based of dir tree diferences
    for files in file_list2:
        if files.endswith(".txt"):
            shutil.move(os.path.abspath(pdfdir+files),os.path.abspath(compdir2+files))


    print("Text extraction completed for ("+ number_of_files + ") .pdf files ********************")
    error_msg(xjob,"Text extraction completed for ("+ number_of_files + ") .pdf files ********************")
