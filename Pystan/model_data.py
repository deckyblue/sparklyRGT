import pandas as pd
import numpy as np

def model_data(fnames,cue,condition,numsessions):
    #fnames is a list of strings with the full file locations for each dataset to be loaded in
    df_all = load_data(fnames)
    df = extract_data(df_all,cue,condition,numsessions)
    df = get_options(df)
    startSubject = start_subject(df)
    startSession = start_session(df)
    model_data = get_data_dict(df, startSubject, startSession)
    return model_data
    
    
def load_data(fnames): 
#load dataset from computer and redo subject numbers
    for i,file in enumerate(fnames):
        if i == 0:
            df = pd.read_csv(fnames[i])
            df['Subject'] += 100
        else:
            df2 = pd.read_csv(fnames[i])
            df2['Subject'] += 100 * (1+i)
            df = df.append(df2, ignore_index = True)
    return df


def extract_data(df, cue, condition, numsessions):
#take only columns needed
    df = df.loc[:, ['Subject','Session','Trial','Pellets','Chosen','Pun_Dur','Cue', 'Condition','Choice']]
#extract specified data
#extract the number of sessions specified above (numsessions)
    df_small = df.loc[df['Session'] < min(df.Session) + numsessions]

#extract the conditions specified above
    df = df_small.loc[np.logical_and(df_small['Cue'] == cue, df_small['Condition'] == condition)]
    
    #sort dataset by subject and then session
    df = df.sort_values(by=['Subject', 'Session'], ignore_index = True)
    
    return df



def get_options(df):
#Numerical options column
    df['Option'] = 0
    for i in range(len(df)):
        if isinstance(df['Choice'][i], str):
            df['Option'][i] = int(df['Choice'][i][1])
    return df


def start_subject(df):
#startSubject is a list the length of the df, where it equals the subject number at the start of that subject's data 
#and zero otherwise

#redo subject numbers in a new column going from 1 - N number of subjects

    subs = df.Subject.unique()
    sub_dict = {}
    for i, sub in enumerate(subs):
    	sub_dict[sub] = i + 1
    df['Sub'] = 0
    for i in range(len(df)):
    	df['Sub'][i] = sub_dict[df['Subject'][i]]
    
    #find the index for the row before where the subject number changes
    s = np.diff(df.Sub)
    s_idx = np.where(s != 0)

    startSubject = [0] * len(df)

    #set the startSubject list value to the subject number when the subject number changes (row after index)
    for index in s_idx[0]:
        startSubject[index + 1] = df['Sub'][index + 1]

    #set the first value to the first subject number
    startSubject[0] = min(df.Sub)
    return startSubject


def start_session(df):
#startSession is a list the length of the df, where it equals the session number at the start of that session 
#and zero otherwise

#find the index for the row before where the session number changes
    s = np.diff(df.Session)
    s_idx = np.where(s != 0)

    startSession = [0] * len(df)

#set the startSession list value to the session number when the session number changes (row after index)
    for index in s_idx[0]:
        startSession[index + 1] = df['Session'][index + 1]
    
#set the first value to the first session number
    startSession[0] = min(df.Session)
    return startSession
    



def get_data_dict(df,startSubject, startSession):
#extract variables from df
    N = len(df.Subject.unique())
    ntr = len(df)

    C = list(df.Chosen)
    R = list(df.Pellets)
    P = list(df.Pun_Dur)
    O = list(df.Option)

    #save them in dictionary
    model_data = {'N': N, 
                  'ntr': ntr,
                  'startSubject': startSubject,
                  'startSession': startSession,
                  'C': C, 
                  'R': R, 
                  'P': P, 
                  'O': O}
    
    return model_data
