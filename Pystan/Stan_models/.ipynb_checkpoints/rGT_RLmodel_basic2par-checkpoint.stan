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
    vector[3] mu_pr;
    vector<lower=0>[3] sigma;

    // Subject-level parameters (raw; these are unit normal for equal density sampling across each par)
    vector[N] beta_samp;
    vector[N] eta_samp;
    vector[N] m_samp;

  } // parameters

  transformed parameters {

      // Subject-level parameters (transformed)
      vector<lower=0,upper=100>[N]    beta;
      vector<lower=0,upper=1>[N]     eta;
      vector[N]    m;

      for (i in 1:N) {
          beta[i]  = Phi_approx( mu_pr[1] + sigma[1] * beta_samp[i] ) * 100;
          eta[i] = Phi_approx( mu_pr[2] + sigma[2] * eta_samp[i] );
          m[i]  = mu_pr[3] + sigma[3] * m_samp[i];
      }

  } // transformed parameters

  model{

    // Group level priors
    mu_pr ~ normal(0, 1);
    sigma ~ gamma(1, 0.5); //normally 0.5, but changing this will change the strength of the group level regularization

    // Subject-level priors - these are for the sampled parameters
    beta_samp ~ normal(0, 1);
    eta_samp ~ normal(0, 1);
    m_samp ~ normal(0, 1);


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
              Q[O[tr]] += eta[currentSubject] * delta;
            }
            // if negative outcome
            if ( P[tr]>0 ) {
              delta = -m[currentSubject]*P[tr] - Q[O[tr]];
              // learn from the negative outcome
              Q[O[tr]] += eta[currentSubject] * delta;
            }

          } // if valid choice

        } // trials
      } // d

    } // model
