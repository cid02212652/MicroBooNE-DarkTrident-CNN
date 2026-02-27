import time
import os 
import numpy as np 


def timestr():
    t0 = time.time() + 60 * 60 * 2
    return time.strftime("%Y%m%d-%I_%M_%p",time.localtime(t0))

def get_fname(file_path):
    full_name = os.path.basename(file_path)
    file_name = os.path.splitext(full_name)
    return file_name[0]

def logit_transform(score):
    return np.log(score/(1-score))

def read_entry_list(filename):
    '''
    Function to read a set of entries 
    from a csv file. 

    filename: path to csv file

    returns an list containing the 
    entries stored in the csv file
    
    '''
    df = pd.read_csv(filename)
    entry_list = df['entry_number'].values.tolist()
    return entry_list