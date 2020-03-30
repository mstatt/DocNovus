def post_run_cleaner(x):
    # -*- coding: utf-8 -*-
    import os
    import glob
    import shutil

    proc_completed = "Completed/"
    initialfiles = x + "/Analysis-Complete/" +"clientfiles/"
    htmlfiles = x + "/Analysis-Complete/" +"html_files/"
    compdir = x + "/" +'forprocessing/'
    txtdir = x + "/" +"txt/"
    pdfdir = x + "/" +"pdf/"
    docdir = x + "/" +"doc/"
    imgdir = x + "/" +"images/"
    csvdir = x + '/' +"csv/"

    print("Post .txt encoding cleanup **********************************")
    shutil.rmtree(txtdir)

    print("Post .pdf extraction cleanup **********************************")
    shutil.rmtree(pdfdir)

    print("Post .doc/.docx extraction cleanup **********************************")
    shutil.rmtree(docdir)

    print("Post .jpg extraction cleanup **********************************")
    shutil.rmtree(imgdir)

    print("Post .csv extraction cleanup **********************************")
    shutil.rmtree(csvdir)

    print("Post process cleanup **********************************")
    shutil.rmtree(compdir)

    if not os.path.exists(initialfiles):
        os.makedirs(initialfiles)

    for file in glob.glob(x+'/'+"*.*"):
        shutil.move(file, initialfiles)
        

    for file in glob.glob(x + '/Analysis-Complete/'+"*.html"):
        shutil.move(file, htmlfiles)

    if os.path.exists('__pycache__'):
        shutil.rmtree('__pycache__')

    shutil.move(x, proc_completed)
    print("Postflight activities complete******************************")
