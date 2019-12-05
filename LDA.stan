data{
  int K; // Number of topics
  int N; // Number of unique words
  int D; // Number of documents
  int ND; // Number of word-instances
  vector[N] eta; // Hyperprior for beta
  int<lower=1,upper=N> w[ND]; // Word correspondence vector for each instance
  int<lower=1,upper=D> doc[ND]; // Document correspondence vector
  int<lower=1> P; // Number of topic predictors
  vector[P] X[D]; // Model matrix for topic predictors
}

parameters{
  simplex[K] theta[D]; // Topic distribution for document d
  simplex[N] beta[K]; // Word distribution for topic k
  matrix[K,P] gamma; // Predictor coefs. (document level covariates)
}

model{
  for(k in 1:K){
    beta[k] ~ dirichlet(eta); // Prior for beta  
  }
  
  for(d in 1:D){
    theta[d] ~ dirichlet(exp(gamma * X[d]));
  }
  
  to_vector(gamma) ~ normal(0,1);
  
  for(i in 1:ND){
    real temp[K];
    for(k in 1:K){
      temp[k] = log(theta[doc[i], k]) + log(beta[k, w[i]]);
    }
    target += log_sum_exp(temp);  // likelihood;
  }
}
