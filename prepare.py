'''
Preapre public name datasets for reproducing experiments
'''
from urllib.request import urlretrieve
from zipfile import ZipFile
import os
from glob2 import glob
def maybe_download_baby_names(link="https://www.ssa.gov/oact/babynames/state/namesbystate.zip"):
    filename=link.split('/')[-1]
    folder_name=filename.split('.')[0]
    if not os.path.exists(filename):
        print("Downloading %s from %s"%(filename,link))
        urlretrieve(link,filename)
        with ZipFile(filename) as fhandler:
            fhandler.extractall(folder_name)
        print("Download complete. Text files are stored at %s"%folder_name)
    text_filenames=glob(folder_name+'/*.TXT')
    names=[] 
    for text_filename in text_filenames:
        with open(text_filename) as fhandler:
            names+=([name.split(',')[3] for name in fhandler])
    stored_all_names=folder_name+'.txt'
    with open(stored_all_names,'w') as fhandler:
        fhandler.write('\n'.join(names))
    print("All names stored at %s"%stored_all_names)


if __name__=='__main__':
    maybe_download_baby_names()



    