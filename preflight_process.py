def file_manage(x):
    import os
    import glob
    import shutil
    import time
    from datetime import date, datetime
    #Imported Modules
    from error_log_writer import error_msg



    print("Starting preflight activities for ......"+x)
    


    content_list = []
    clientdir  = x + '/'
    docdir = x + '/' +"doc/"
    pdfdir = x + '/' +"pdf/"
    txtdir = x + '/' +"txt/"
    csvdir = x + '/' +"csv/"
    imgdir = x + '/' +"images/"
    initialfiles = x + "/Analysis-Complete/" +"clientfiles/"
    htmlfiles = x + "/Analysis-Complete/" +"html_files/"
    qafiles = x + "/Analysis-Complete/" +"qanda_files/"
    dir_name = x.replace("job_queue/", "")



    f= open(os.path.join(clientdir, dir_name + ".txt"),"w+")
    f.write("blank")
    f.close()

    number_of_files = str(len([item for item in os.listdir(clientdir) if os.path.isfile(os.path.join(clientdir, item))]))
    print("Processing "+ number_of_files + " files.....")
    tmstp = str(datetime.today().strftime("%m-%d-%Y"))
    with open (tmstp +"_process_log.txt","a")as fp1:
       fp1.write("Processing "+ number_of_files + " files for "+ dir_name + "\n")
    fp1.close()



    if not os.path.exists(initialfiles):
        os.makedirs(initialfiles)

    if not os.path.exists(htmlfiles):
        os.makedirs(htmlfiles)

    if not os.path.exists(qafiles):
        os.makedirs(qafiles)

    if not os.path.exists(qafiles):
        os.makedirs(qafiles)

        

    with open (x + "/Analysis-Complete/" + "error_process_log.txt","a")as fp3:
        fp3.write("*************************************Error Log*********************************************\n")
    fp3.close()
    error_msg(x,"Starting preflight activities for "+x)

    for setfiles in glob.glob(clientdir+"*.*"):
        error_msg(x,"Starting to process ["+ str(os.path.basename(setfiles)) +"] .***************************************")

    shutil.copyfile('logo.png',x + "/Analysis-Complete/logo.png")


    if not os.path.exists(clientdir + 'forprocessing/'):
        os.makedirs(clientdir + 'forprocessing/')

    print("Preflight directory set up ***************************************")
    error_msg(x,"Preflight directory set up.***************************************")

    if not os.path.exists(txtdir):
        os.makedirs(txtdir)
    print("Moving text files ***************************************")
    for filename2 in glob.glob(clientdir+"*.txt"):
        shutil.copyfile(clientdir+os.path.basename(filename2),txtdir+os.path.basename(filename2))
        #shutil.copyfile(clientdir+os.path.basename(filename2),txtdir+os.path.basename(filename2))

    if not os.path.exists(csvdir):
        os.makedirs(csvdir)
    print("Moving csv files ***************************************")
    for filename2 in glob.glob(clientdir+"*.c*"):
        shutil.copyfile(clientdir+os.path.basename(filename2),csvdir+os.path.basename(filename2))
        #shutil.copyfile(clientdir+os.path.basename(filename2),txtdir+os.path.basename(filename2))

    if not os.path.exists(pdfdir):
        os.makedirs(pdfdir)
    print("Moving .pdf files ***************************************")
    error_msg(x,"Moving .pdf files "+x)
    for filename1 in glob.glob(clientdir+"*.pdf"):
        shutil.copyfile(clientdir+os.path.basename(filename1),pdfdir+os.path.basename(filename1))
        #shutil.copyfile(clientdir+os.path.basename(filename1),pdfdir+os.path.basename(filename1))

    if not os.path.exists(docdir):
        os.makedirs(docdir)
    print("Moving word documents ***************************************")
    for filename in glob.glob(clientdir+"*.doc*"):
        shutil.copyfile(clientdir+os.path.basename(filename),docdir+os.path.basename(filename))
        #shutil.copyfile(clientdir+os.path.basename(filename),docdir+os.path.basename(filename))

    if not os.path.exists(imgdir):
        os.makedirs(imgdir)
    print("Moving image files ***************************************")
    for filename in glob.glob(clientdir+"*.jpg*"):
        shutil.copyfile(clientdir+os.path.basename(filename),imgdir+os.path.basename(filename))
    for filename in glob.glob(clientdir+"*.jpeg*"):
        shutil.copyfile(clientdir+os.path.basename(filename),imgdir+os.path.basename(filename))
    for filename in glob.glob(clientdir+"*.png*"):
        shutil.copyfile(clientdir+os.path.basename(filename),imgdir+os.path.basename(filename))

    print("Preflight activities completed ***********************************")
