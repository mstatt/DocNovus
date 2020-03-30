#!/usr/bin/env python
# coding: utf-8
def extractimagetext(x):
	import os
	import re
	import glob
	import string
	from pytesseract import image_to_string
	from PIL import Image
	import shutil
	from error_log_writer import error_msg


	print("Starting Text Extraction for images's *************************")
	error_msg(x,"Starting Text Extraction for images's *************************")
	content_list = []
	compdir = x + "/" +'forprocessing/'
	imgdir = x + "/" +"images/"
	number_of_files = str(len([item for item in os.listdir(imgdir) if os.path.isfile(os.path.join(imgdir, item))]))
	print("Processing ("+ number_of_files + ") .jpg files **************************************.")
	error_msg(x,"Processing ("+ number_of_files + ") .jpg files **************************************")

	for filename3 in glob.glob(imgdir+"*.jpg"):
		#print("Processing ("+ filename3 + ") file **************************************.")
		txtstring = re.sub(r'[^\x00-\x7f]',r'', image_to_string(Image.open(filename3), lang='eng'))
		txtstring = str(txtstring).replace('\n', '')
		with open (os.path.splitext(compdir+os.path.basename(filename3))[0]+"_img_text.txt","w")as fimg1:
			fimg1.write(txtstring)
		fimg1.close()

	for filename3 in glob.glob(imgdir+"*.png"):
		#print("Processing ("+ filename3 + ") file **************************************.")
		txtstring = re.sub(r'[^\x00-\x7f]',r'', image_to_string(Image.open(filename3), lang='eng'))
		txtstring = str(txtstring).replace('\n', '')
		with open (os.path.splitext(compdir+os.path.basename(filename3))[0]+"_img_text.txt","w")as fimg1:
			fimg1.write(txtstring)
		fimg1.close()



	print("Writing extracted image output files ***************************************")
	error_msg(x,"Writing extracted image output files  ******************************")
	for filename4 in glob.glob(imgdir+"*.txt"):
		shutil.move(filename4,compdir+os.path.basename(filename4))

	print("Text extraction completed for ("+ number_of_files + ") .jpg files  ******************************")
	error_msg(x,"Text extraction completed for ("+ number_of_files + ") .jpg files *******************")
