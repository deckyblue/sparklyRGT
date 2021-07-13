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
pscale_model = """
  // rGT task -- basic RL model

  data {

    // Metadata
    int N;              // number of subjects
    int ntr;            // total number of trials

    //Data
    int startSubject[ntr];  // subject number at start of each subject sequence
    int startSession[ntr];  // session number at start of each session

    int C[ntr];     // chosen stimulus 1-4
    int R[ntr];     // outcome in number of pellets
    int P[ntr];     // punishment in number of seconds
    int O[ntr];   // chosen option (i.e. P1,P2,P3,P4)

  } // data

  parameters{

    // Group level parameters - beta, etaPos, etaNeg, pscale, pintercept
    vector[5] mu_pr;
    vector<lower=0>[5] sigma;

    // Subject-level parameters (raw; these are unit normal for equal density sampling across each par)
    vector[N] beta_samp;
    vector[N] etaPositive_samp;
    vector[N] etaNegative_samp;
    vector[N] m_samp;
    vector[N] b_samp;

  } // parameters

  transformed parameters {

      // Subject-level parameters (transformed)
      vector<lower=0,upper=100>[N]    beta;
      vector<lower=0,upper=1>[N]     etaPositive;
      vector<lower=0,upper=1>[N]     etaNegative;
      vector[N]    m;
      vector[N]    b;

      for (i in 1:N) {
          beta[i]  = Phi_approx( mu_pr[1] + sigma[1] * beta_samp[i] ) * 100;
          etaPositive[i] = Phi_approx( mu_pr[2] + sigma[2] * etaPositive_samp[i] );
          etaNegative[i] = Phi_approx( mu_pr[3] + sigma[3] * etaNegative_samp[i] );
          m[i]  = mu_pr[4] + sigma[4] * m_samp[i];
          b[i]  = mu_pr[5] + sigma[5] * b_samp[i];
      }

  } // transformed parameters

  model{

    // Group level priors
    mu_pr ~ normal(0, 1);
    sigma ~ gamma(1, 0.5); //normally 0.5, but changing this will change the strength of the group level regularization

    // Subject-level priors - these are for the sampled parameters
    beta_samp ~ normal(0, 1);
    etaPositive_samp ~ normal(0, 1);
    etaNegative_samp ~ normal(0, 1);
    m_samp ~ normal(0, 1);
    b_samp ~ normal(0, 1);


    // put this in a false loop (error message at vector...)
    for (d in 1:1) {

      //Initialize
      vector[4] Q;
      real delta;
      int currentSubject;
      int currentSession;

      Q = rep_vector(0, 4);
      delta = 0;
      currentSubject = 0;
      currentSession = 0;

      // Likelihood of all the data
      for (tr in 1:ntr){

        if (startSubject[tr]>0) { // if this is the start of a new subject
          currentSubject = startSubject[tr];

          // reset Q values
          Q = rep_vector(0,4);
        }

        if (startSession[tr]>0) { // if this is the start of a new session
          currentSession = startSession[tr];
        }

          //if a choice was made
          if ( O[tr]>0 ) {

            // likelihood of observed choice
            O[tr] ~ categorical_logit( beta[currentSubject] * Q );

            // if positive outcome
            if ( R[tr]>0 ) {
              delta = R[tr] - Q[O[tr]];
              // learn from the positive outcome
              Q[O[tr]] += etaPositive[currentSubject] * delta;
            }
            // if negative outcome
            if ( P[tr]>0 ) {
              delta = b[currentSubject] - m[currentSubject]*P[tr] - Q[O[tr]];
              // learn from the negative outcome
              Q[O[tr]] += etaNegative[currentSubject] * delta;
            }

          } // if valid choice

        } // trials
      } // d

    } // model
"""

#compile stan code
##change model argument if needed 
model = stan.build(pscale_model, data = model_data, random_seed=1)

#do the sampling
fit = model.sample(num_chains=4)
fit_az = az.from_pystan(fit)
##change argument (task_model_fit.nc)
fit_az.to_netcdf('outcome_pscale_fit.nc')