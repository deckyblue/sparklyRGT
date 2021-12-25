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
