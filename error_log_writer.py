def error_msg(x,msg): #Pass in Job folder and message
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
    with open (x + "/Analysis-Complete/" + "error_process_log.txt","a")as fp1:
        fp1.write(now + " | " + msg + "\n")
    fp1.close()
