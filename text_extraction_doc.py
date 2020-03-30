def extract_Text_doc(x):
    # -*- coding: utf-8 -*-
    import os
    import glob
    import docx
    from docx import Document
    import shutil
    #Imported Modules
    from error_log_writer import error_msg


    print("Starting Text Extraction for word doc's......")
    error_msg(x,"Starting Text Extraction for word doc's  *************************")
    content_list = []
    compdir = x + "/" +'forprocessing/'
    docdir = x + "/" +"doc/"
    number_of_files = str(len([item for item in os.listdir(docdir) if os.path.isfile(os.path.join(docdir, item))]))
    print("Processing ("+ number_of_files + ") .docx files.....")
    error_msg(x,"Processing ("+ number_of_files + ") .docx files  **************************************")

    for filename3 in glob.glob(docdir+"*.docx"):
        #Get the filename without the extension for nameing later
        base=os.path.basename(filename3)
        filenameNoExt = os.path.splitext(base)[0]
        doc = docx.Document(filename3)
        fullText = []
        for para in doc.paragraphs:
            fullText.append(para.text)
        textstream = '\n'.join(fullText)
        # content_list = list(filter(None, content_list))
        with open (docdir+'doc_'+filenameNoExt+".txt","a")as fp11:
            fp11.write(textstream)
            fp11.close()
    print("Writing extracted doc output files ***************************************")
    error_msg(x,"Writing extracted doc output files  ******************************")
    for filename4 in glob.glob(docdir+"*.txt"):
        shutil.move(filename4,compdir+os.path.basename(filename4))


    print("Text extraction completed for ("+ number_of_files + ") .docx files  ******************************")
    error_msg(x,"Text extraction completed for ("+ number_of_files + ") .docx files *******************")
