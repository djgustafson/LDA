library(stm)

data("gadarian")

corpus <- textProcessor(documents = gadarian$open.ended.response,
                        metadata = gadarian,stem = FALSE
                        ,customstopwords=c("immigrant","immigration",
                                           "immigrants", "illegal"))

gadarian_stm <- prepDocuments(corpus$documents, corpus$vocab, corpus$meta)

word_data <- do.call(rbind, lapply(seq_along(gadarian_stm$documents),
                                   function(x){
                                     words <- rep(gadarian_stm$documents[[x]][1,]
                                                  ,times=gadarian_stm$documents[[x]][2,])
                                     cbind(words,x)
                                   }
))

X <- model.matrix(~treatment,data=gadarian_stm$meta)

stan_data <- list(K = 2, N = length(unique(word_data[,"words"])), 
                  D = length(unique(word_data[,"x"])), ND = length(word_data[,"words"]),
                  X = X, P = ncol(X), w = word_data[,"words"], doc = word_data[,"x"],
                  eta = rep(0.3,length(unique(word_data[,"words"]))))

## Obtain initial values for theta using k-means
docterm <- convertCorpus(gadarian_stm$documents,gadarian_stm$vocab,"Matrix")
kclust <- kmeans(docterm,stan_data$K)
topics <- as.factor(fitted(kclust,"class"))

inits <- model.matrix(~topics-1)

inits[inits==1] <- 0.999
inits[inits==0] <- (1-0.999)/(stan_data$K-1)

## Estimate model in Stan
library(rstan)
rstan_options(auto_write = TRUE)

lda_model <- stan("LDA.stan"
                   ,data = stan_data
                   ,chains = 1
                   ,init = list(list(theta=inits))
                   ,seed = 4
                   ,iter = 2.5e3)

## Obtain 10 highest probability words per topic
## This will be an array with samples x topics x words
beta_samp <- extract(lda_model,"beta")$beta

##Get median across samples
beta_median <- apply(beta_samp,c(2,3),mean)

## Get top 10 words per topic
library(knitr)
top_words <- apply(beta_median,1,function(x)order(x, decreasing=TRUE)[1:10])
kable(apply(top_words,2,function(x)gadarian_stm$vocab[x]), col.names = c("Topic1","Topic2"))

## Obtain estimates of predictor effects
exp(summary(lda_model,"gamma",probs=c(0.05,0.5,0.95))$summary[,c(4:6)])
