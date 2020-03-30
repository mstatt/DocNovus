#!/usr/bin/env python
# coding: utf-8
def word_count(inputfilename):
    import os
    import re
	    
    #Imported Modules
    # from error_log_writer import error_msg

    
    # print("Executing word_counter for ("+ inputfilename + ")  *********************************")
    # error_msg(x,"Executing word_counter for ("+ inputfilename + ")  *********************************")
    num_words = 0

    with open(inputfilename, 'r') as f:
    	for line in f:
    		words = line.split()
    		num_words += len(words)

    return num_words