def text_file_encoder(x):
    # -*- coding: utf-8 -*-
    import os
    import glob
    import shutil
    #Imported Modules
    from error_log_writer import error_msg

    compdir = x + '/' +'forprocessing/'
    txtdir = x + '/' +"txt/"
    print("Starting to encode txt files....")
    error_msg(x,"Starting to encode txt files  ************************************")

    print("Moving text files to processing directory......")
    for filename2 in glob.glob(txtdir+"*.*"):
        with open(filename2, 'r') as myfile2:
            data2="".join(line.rstrip() for line in myfile2)
            myfile2.close()
            with open (compdir+os.path.basename(filename2),"a")as fp1:
                fp1.write(data2)
            fp1.close()



    print("Text encoding completed for txt files.....")
    error_msg(x,"Text encoding completed for txt files  ***************************")
