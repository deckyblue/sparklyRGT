import stan #install pystan using pip first
import model_data as md 
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

#arviz is for visualization of stan output
import arviz as az #install using pip first
import xarray as xr

#---------------------------

#number of sessions to do modelling on
numsessions = 5


fnames = ['BH09_raw-free_S1-5_corrected.xlsx','CH02_corrected.xlsx','NA01_raw_free-choice_S8-18.xlsx',"CH01_corrected.xlsx"]
#change these to the names of the excel files that you want to run modeling on 


#--------------------------

#load in excel files into one df, resetting the session numbers to start at 1
df = md.load_multiple_data(fnames, reset_sessions = True)


#creates lists of subjects run on each task (classic A, classic B, etc.)

#rename MSNs so that the rats on the outcome task don't have "loss" in the MSN
for i in range(len(df)):
    if df.at[i, 'MSN'] == 'LossrGT_A-losscue_v1':
        df.at[i,'MSN'] = 'outcomeRGT_A'
    if df.at[i, 'MSN'] == 'LossrGT_B-losscue_v1':
        df.at[i,'MSN'] = 'outcomeRGT_B'
        
#rename MSNs so that the rats on the random task don't have "loss" in the MSN
for i in range(len(df)):
    if df.at[i,'MSN'] == 'AnarchyrGT_B-losscue_v6':
        df.at[i,'MSN'] = 'RandomRGT_B'
    if df.at[i,'MSN'] == 'AnarchyrGT_A-losscue_v6':
        df.at[i,'MSN'] = 'RandomRGT_A'
        
        
task_list = df.groupby(['MSN'])['Subject'].unique()

#concatenate together the lists of subjects that run the same task (i.e., puts version A 
#and version B together) - based on unique string for each task name

##change between model runs
subs = np.concatenate(task_list[[task for task in df.MSN.unique() if 'outcome' in task]])

#---------------------------------------------


#extract model data from dataframe
model_data = md.get_model_data(df, numsessions, subs)

#stan code
##identify the model 
pindep_star_model = """
// rGT task -- basic RL model

data {

        // Metadata
        int N;          // number of subjects
        int ntr;        // total number of trials

        //Data
        int startSubject[ntr]; // subject number at start of each subject sequence
        int startSession[ntr]; // session number at start of each session

        int C[ntr]; // chosen stimulus 1-4
        int R[ntr]; // outcome in number of pellets
        int P[ntr]; // punishment in number of seconds
        int O[ntr]; // chosen option (i.e. P1,P2,P3,P4)

}   // data

parameters {

        // Group level parameters - etaPositive, beta are drawn from this...
        vector[6] mu_pr;
        vector<lower=0>[6] sigma;

        // Subject-level parameters (raw; these are unit normal for equal density sampling across each par)
        vector[N] beta_samp;
        vector[N] eta_samp;
        vector[N] p1_samp; // C1 weight on time outs
        vector[N] p2_samp; // C2 weight on time outs
        vector[N] p3_samp; // C3 weight on time outs
        vector[N] p4_samp; // C4 weight on time outs

}   // parameters

transformed parameters {

        // Subject-level parameters (transformed)
        vector<lower=0,upper=100>[N]    beta;
        vector<lower=0,upper=1>[N]     eta;
        vector[N]  p1;
        vector[N]  p2;
        vector[N]  p3;
        vector[N]  p4;

        for (i in 1:N) {
                beta[i]  = Phi_approx( mu_pr[1] + sigma[1] * beta_samp[i] ) * 100;
                eta[i] = Phi_approx( mu_pr[2] + sigma[2] * eta_samp[i] );
                p1[i] = mu_pr[3] + sigma[3] * p1_samp[i];
                p2[i] = mu_pr[4] + sigma[4] * p2_samp[i];
                p3[i] = mu_pr[5] + sigma[5] * p3_samp[i];
                p4[i] = mu_pr[6] + sigma[6] * p4_samp[i];
        }

}   // transformed parameters

model {

        // Group level priors
        mu_pr ~normal(0, 1);
        sigma ~gamma(1, 0.5);

        // Subject-level priors - these are for the sampled parameters
        beta_samp ~normal(0, 1);
        eta_samp ~normal(0, 1);
        p1_samp ~normal(0, 1);
        p2_samp ~normal(0, 1);
        p3_samp ~normal(0, 1);
        p4_samp ~normal(0, 1);


        // put this in a false loop (error message at vector...)
        for (d in 1:1) {

                //Initialize
                vector[4] Q;
                vector[4] pscale;
                real delta;
                int currentSubject;
                int currentSession;

                Q = rep_vector(0, 4);
                pscale = rep_vector(0, 4);
                delta = 0;
                currentSubject = 0;
                currentSession = 0;

                // Likelihood of all the data
                for (tr in 1:ntr) {

                        if (startSubject[tr]>0) { // if this is the start of a new subject
                                currentSubject = startSubject[tr];

                                // reset Q values
                                Q = rep_vector(0,4);

                                // reset p scale
                                pscale[1] = p1[currentSubject];
                                pscale[2] = p2[currentSubject];
                                pscale[3] = p3[currentSubject];
                                pscale[4] = p4[currentSubject];
                        }

                        if (startSession[tr]>0) { // if this is the start of a new session
                                currentSession = startSession[tr];
                        }

                        //if a choice was made
                        if ( O[tr]>0 ) {

                                // likelihood of observed choice
                                O[tr] ~categorical_logit( beta[currentSubject] * Q );

                                // calculate effective outcome
                                if ( R[tr]>0 ) {
                                        delta = R[tr] - Q[O[tr]]; // learn from positive
                                }
                                if ( P[tr]>0 ) {
                                        delta = -pscale[O[tr]]*P[tr] - Q[O[tr]];
                                }
                                // update based on outcome
                                Q[O[tr]] += eta[currentSubject] * delta;
                        } // if valid choice

                } // trials
        } // d

}     // model
"""

#compile stan code
##change model argument if needed 
model = stan.build(pindep_star_model, data = model_data, random_seed=1)

#do the sampling
fit = model.sample(num_chains=4)
fit_az = az.from_pystan(fit)
##change argument (task_model_fit.nc)
fit_az.to_netcdf('outcome_pindep-star_fit.nc')