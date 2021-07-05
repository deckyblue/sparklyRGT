import pandas as pd
import numpy as np

def get_model_data(fnames,numsessions, subjects, reset_sessions = True):
    #fnames is a list of strings with the full file locations for each dataset to be loaded in
    df_all = load_multiple_data(fnames, reset_sessions = reset_sessions)
    df = extract_data(df_all, numsessions, subjects)
    df = get_options(df) ## get_choices***
    startSubject = start_subject(df)
    startSession = start_session(df)
    model_data = get_data_dict(df, startSubject, startSession)
    return model_data
    
    
def load_multiple_data(fnames, reset_sessions=False): 
#load multiple datasets from computer and redo subject numbers (for multiple cohorts) 
    for i,file in enumerate(fnames):
        if i == 0:
            df = pd.read_excel(fnames[i])
            df['Subject'] += 100 #rat 1 becomes rat 101
            if reset_sessions:
                for i,session in enumerate(df.Session.unique()):
                    for j in range(len(df)):
                        if df.at[j,'Session'] == session:
                            df.at[j,'Session'] = i + 1
        else:
            df2 = pd.read_excel(fnames[i])
            df2['Subject'] += 100 * (1+i) #rat 1 becomes rat 201, 301, etc. 
            if reset_sessions:
                for i,session in enumerate(df2.Session.unique()):
                    for j in range(len(df2)):
                        if df2.at[j,'Session'] == session:
                            df2.at[j,'Session'] = i + 1 #makes sessions start from 1 
            df = df.append(df2, ignore_index = True)
    return df


def extract_data(df, numsessions, subjects):
    "take the data for subjects in subjects list from the same condition (task, drug, etc. - specified prior to this function) for the specified number of sessions"
#take only columns needed
    df = df.loc[:, ['Subject','Session','Trial','Pellets','Chosen','Pun_Dur','Option']]
#extract specified data
#extract the number of sessions specified above (numsessions)
    df_small = df.loc[df['Session'] < min(df.Session) + numsessions]

#extract the conditions based on given list of subject numbers
   #df = df_small.loc[np.logical_and(df_small['Cue'] == cue, df_small['Condition'] == condition)]
    df = df_small[df_small['Subjects'].isin(subjects)]
    
    #sort dataset by subject and then session
    df = df.sort_values(by=['Subject', 'Session'], ignore_index = True)
    
    return df

def extract_data_chosen(df, numsessions, subjects):
    "take the data for subjects in subjects list from the same condition (task, drug, etc.) for the specified number of sessions"
#take only columns needed
    df = df.loc[:, ['Subject','Session','Trial','Pellets','Chosen','Pun_Dur','Chosen']]
#extract specified data
#extract the number of sessions specified above (numsessions)
    df_small = df.loc[df['Session'] < min(df.Session) + numsessions]

#extract the conditions based on given list of subject numbers
   #df = df_small.loc[np.logical_and(df_small['Cue'] == cue, df_small['Condition'] == condition)]
    df = df_small[df_small['Subjects'].isin(subjects)]
    
    #sort dataset by subject and then session
    df = df.sort_values(by=['Subject', 'Session'], ignore_index = True)
    
    return df


def get_choices(df):
    configA = np.array([1, 4, 0, 2, 3]) #this is the order for version A - i.e., hole 1 corresponds to P1
    configB = np.array([4, 1, 0, 3, 2]) #this is the order for version B - i.e., hole 1 corresponds to P4

    #I took the following code from someone else, so honestly I'm not entirely sure how it works haha
    #the important thing is that it uses the configurations above to assign the correct option, 
    #based on whether the MSN name contains 'A' or 'B'
    df['Option'] = df['MSN'].str.contains("B").values*configB[df['Chosen'].astype('int').ravel()-1].astype('int') + \
        df['MSN'].str.contains("A").values*configA[df['Chosen'].astype('int').ravel()-1].astype('int')
    
    ###I can just take your word for it right? haha
    ####Yes

    #the above code changes any zero in the chosen column to a three in the option column - don't need to know why
    #so we need to fix that (zeros represent either a premature response or an omission)
    for i in range(len(df)): ##range gives me a list from 0 to the len(df), which should be all the indices
        if df['Chosen'][i] == 0: ###can we say this in English? 
            ##the same as df.at
            ##if the index of the 'Chosen' column gets 0 --> option equals 0 
            df['Option'][i] = 0 
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
