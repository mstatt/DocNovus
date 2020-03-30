import os
import subprocess
import sys
import time
from datetime import date, datetime



start = time.time()
start_time = datetime.now()
now = time.ctime(int(start))
tmstp = str(datetime.today().strftime("%m-%d-%Y"))
#----------Write to existing log file ---------------------------------------------
with open (tmstp +"_process_log.txt","a")as fp1:
    fp1.write("**************************************************************\n")
    fp1.write("Monitor script started at :" + str(now)+ "\n")
fp1.close()

#----------------------------------------------------------------------------------
#---------------------Set directory and call script on job folders-----------------
jobdir = 'job_queue' + '/'
fcount = 0
dlist = []
dlist = filter(lambda x: x, os.listdir(jobdir))
for x in sorted(dlist):
    fcount = fcount +1
    os.system('python run-all.py ' + x)
#----------------------------------------------------------------------------------
now = time.ctime(int(start))
end_time = datetime.now()
#----------Write to existing log file ---------------------------------------------
with open (tmstp +"_process_log.txt","a")as fp2:
    fp2.write("Processed " + str(fcount) + " jobs during last run. \n")
    fp2.write('Bulk process duration: {} \n'.format(end_time - start_time))
    fp2.write("Monitor script ended at " + str(now)+ "\n")
    fp2.write("**************************************************************\n")
fp2.close()
#----------------------------------------------------------------------------------
