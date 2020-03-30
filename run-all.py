import os
import sys
import time
from datetime import date, datetime



firstarg= 'job_queue' + '/' + sys.argv[1]
start = time.time()
start_time = datetime.now()
now = time.ctime(int(start))
tmstp = str(datetime.today().strftime("%m-%d-%Y"))
#---------------Create or Write to existing log file -------------------------------
with open (tmstp +"_process_log.txt","a")as fp1:
   fp1.write("Processing started for " + sys.argv[1] + " at " + str(now)+ "\n")
fp1.close()
#-----------------------------------------------------------------------------------
files = os.listdir(firstarg)
for f in files:
    os.rename(firstarg + '/' + os.path.basename(f), firstarg + '/' + os.path.basename(f).replace(' ', '_'))
#---------------------Load process scripts------------------------------------------
from preflight_process import file_manage
from text_file_manager import text_file_encoder
from text_extraction_doc import extract_Text_doc
from text_fromimage import extractimagetext
from csv_file_manager import csv_file_encoder
from text_extraction_pdf2 import extract_Text_pdf
from multi_file_processor import process_files
from post_flight import post_run_cleaner
#----------------Call file processor------------------------------------------------
file_manage(firstarg)
text_file_encoder(firstarg)
csv_file_encoder(firstarg)
extractimagetext(firstarg)
extract_Text_doc(firstarg)
extract_Text_pdf(firstarg)
process_files(firstarg)
post_run_cleaner(firstarg)
#----------------End file processor------------------------------------------------
#----------Write to existing log file ---------------------------------------------
end_time = datetime.now()
with open (tmstp +"_process_log.txt","a")as fp1:
   fp1.write('Execution time:  {} '.format(end_time - start_time) + " for " + sys.argv[1]+ "\n")
fp1.close()
#----------------------------------------------------------------------------------
