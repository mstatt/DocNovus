# DocNovus
Multifile document analysis application. Python application that works on a variety of file types: (.txt, .doc, .docx, .pdf and .csv).

Created on Tue Dec 13 01:25:12 2017

Last Update on March 30 11:29:03 2020

@author: Michael Stattelman

#************************************************************************

    Complete multi-file document analysis:
    1) Fuzzy String compare for file similarity
    2) File Summarization
    3) Word frequency counter
    4) Phrase frequency counter
    5) File Sentiment Analyzer
    6) Name Entity Extraction
    7) Sentence Frequency counter
    
#************************************************************************

    #Instructions:
    ##---------------------------------------------------
    Prior to running ensure all .txt files are UTF-8 encoded

    ## Run these prior to running DocNovus:
    ##-------------------------------------------------------------------------
    curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
    sha256sum Anaconda3-2019.03-Linux-x86_64.sh
    bash Anaconda3-2019.03-Linux-x86_64.sh
    sudo apt install -y python3-pip
    conda update --all
    source ~/.bashrc
    conda update --all
    conda install -c anaconda ipython
    conda install -c anaconda jupyter
    conda install -c anaconda pandas
    conda install -c anaconda seaborn
    conda install -c anaconda spyder
    conda install -c anaconda tensorflow
    conda install -c conda-forge spacy
    conda install -c plotly plotly-orca psutil requests
    conda install -c conda-forge gensim
    conda install -c conda-forge pyteaser
    conda install -c conda-forge beautifulsoup4
    conda install -c conda-forge poppler
    conda install -c conda-forge textblob
    conda install -c conda-forge pytesseract
    conda install -c bioconda wkhtmltopdf
    conda install -c libgcc
    conda install -c conda-forge python-docx
    conda install -c conda-forge pdftotext
    conda update --all
    python -m nltk.downloader stopwords
    sudo apt-get install -y libpoppler-cpp-dev
    sudo pip install --upgrade pip
    sudo pip install wheel
    sudo pip install pandas
    sudo pip install nltk
    sudo pip install textract
    sudo pip install pdfkit
    sudo pip install pdftotext
    sudo pip install --upgrade tfBinaryURL
    sudo pip install pyteaser 
    sudo pip install fuzzy
    sudo pip install python-docx
    sudo pip install pdftotext
    sudo pip install --pre python-docx
    sudo pip install pypdf2
    sudo apt-get update
    sudo apt-get install build-essential libpoppler-cpp-dev pkg-config python-dev
    sudo apt-get install -y python-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev
    pip install pdfkit
    pip install pdftotext
    sudo pip install nltk
    sudo pip install vaderSentiment
    sudo pip install --upgrade nltk
    sudo pip install --upgrade keras
    
    For Linux:
    >> wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.3/wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
    >> tar vxf wkhtmltox-0.12.3_linux-generic-amd64.tar.xz
    >> cp wkhtmltox/bin/wk* /usr/local/bin/
    ************************************************************************
    Inital Run Uncomment the following lines (128-131) in multi_file_processor.py to download the NLTK resources, 
    After the 1st run you can remove or comment them out.
    ************************************************************************

************************************************************************
See Quick tutorial in the (DocNovus.mp4) file.

1) Place the Directory in the job_queue folder
2) Call the script as follows:
    python run-all.py <foldername>
3) Or to process all folders in the job_queue directory
    python monitor.py 
************************************************************************
Issues:
1) All .txt and .csv files need to be UTF8 encoded.
2) The process takes a extended amount of time can be optimized.
