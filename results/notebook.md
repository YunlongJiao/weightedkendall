The Weighted Kendall Kernel for Permutations
================
Yunlong Jiao
June 12, 2018

-   [I. Eurobarometer (eubm) data and experimental setup](#i.-eurobarometer-eubm-data-and-experimental-setup)
-   [II. Classification of ranking data](#ii.-classification-of-ranking-data)
-   [III. Results](#iii.-results)
-   [IV. Session info](#iv.-session-info)

This notebook reproduces experiments of the paper:

> Yunlong Jiao, Jean-Philippe Vert. "The Weighted Kendall and High-order Kernels for Permutations." arXiv preprint arXiv:1802.08526, 2018. [arXiv:1802.08526](https://arxiv.org/abs/1802.08526)

``` r
knitr::opts_chunk$set(error = FALSE, warning = FALSE, message = FALSE)
options(stringsAsFactors = FALSE)
library(kernrank) # for fast kendall and weighted kendall
library(kernlab) # for svm and various kernels
library(ggplot2) # for general plots
library(corrplot) # for corr and heatmap plots
Rcpp::sourceCpp("../src/dots.cpp") # some dot functions
source("../R/func.R")
set.seed(70236562)
```

I. Eurobarometer (eubm) data and experimental setup
===================================================

Data are accessed, December 2017, online from the repository of the report:

> T. Christensen. "Eurobarometer 55.2: Science and tech- nology, agriculture, the euro, and internet access, may-june 2001." June, 2010. Cologne, Germany: GESIS/Ann Arbor, MI: Inter-university Consortium for Political and Social Research \[distributors\], 2010-06-30. [DOI:10.3886/ICPSR03341.v3](https://doi.org/10.3886/ICPSR03341.v3).

As part of the European Union survey Eurobarometer 55.2, participants were asked to rank, according to their opinion, the importance of six sources of information regarding scientific developments: TV, radio, newspapers and magazines, scientific magazines, the internet, school/university. The dataset also includes demographic information of the participants such as gender, nationality or age. The objective of this study is to predict the age group of participants from their ranking of 6 sources of news. Notably, this data was also studied in a similar supervised context in the following paper:

> H. Mania, et al. "On kernel methods for covariates that are rankings." arXiv preprint arXiv:1603.08035v2, 2017. [arXiv:1603.08035v2](https://arxiv.org/abs/1603.08035)

Raw data can be downloaded from the website [DOI:10.3886/ICPSR03341.v3](https://doi.org/10.3886/ICPSR03341.v3) where detailed description of the survey can also be found. Due to data protection policy, here we only provide the script below to process the raw data by anonymizing, randomizing, and subsampling the full dataset. Briefly, we removed all respondents who did not provide a complete ranking over all six sources, leaving a total of 12,216 participants. Then, we split the dataset across age groups, where 5,985 participants were 40 years old or younger, 6,231 were over 40. In order to perform classification, we chose to fit kernel SVMs with different kernels and compare.

``` r
# set seed
set.seed(70236562)

# eubm data
dname <- "eubm"
datapath <- paste0("../data/dat_", dname, ".RData")

# parameters to choose
nrep <- 50
ntr <- 400
ntst <- 100

# parameters specific to eubm data
p <- 6
ntot <- 12216

# process data
if (file.exists(datapath)) {
  dat <- get(load(datapath))
} else {
  # read full data
  d <- read.table("../data/ICPSR_03341/DS0001/03341-0001-Data.tsv", header = TRUE, sep = "\t")
  
  # get ranking data from survey no. Q5/V57-62 - x
  dx <- as.matrix(d[ , paste0("v", 57:62)])
  stopifnot(ncol(dx) == p)
  # keep full ranking only
  id.keep <- apply(dx, 1, function(u) all(sort(u) == seq(p)))
  stopifnot(sum(id.keep) == ntot)
  # further process such that larger values indicate higher preference
  dx <- t(apply(dx[id.keep, ], 1, function(u) return(p+1-u)))
  stopifnot(nrow(dx) == ntot)
  colnames(dx) <- c("TV", "RADIO", "NEWSP_MAGAZ", "SC_MAGAZINES", "INTERNET", "SCHOOL_UNIV")
  rownames(dx) <- paste0("ptcpt", 1:ntot)
  
  # get BINARY labels - y
  dy <- list(
    # age group from survey no. D11/V522
    "age" = factor(d[id.keep, "v522", drop=T] > 40, levels = c(TRUE, FALSE), labels = c("G40", "LE40"))
  )
  stopifnot(all(sapply(dy, function(u) sum(table(u)) == ntot)))
  dy <- lapply(dy, function(u){ names(u) <- rownames(dx); return(u) })
  
  # create data splits as a list structured as dat$`task`$`rep`$`xtrain, ytrain, xtest, ytest`
  dat <- lapply(seq(nrep), function(irep){
    lapply(dy, function(u){
      stopifnot(length(levels(u)) == 2)
      stopifnot(all(table(u) > (ntr+ntst)/2))
      stopifnot(length(u) == ntot)
      # evenly sample (ntr+ntst)/2 instances per group and get indices
      idx <- unlist(lapply(split(1:ntot, u), sample, size=(ntr+ntst)/2, replace=F))[sample(ntr+ntst, replace=F)]
      trainidx <- idx[1:ntr]
      testidx <- idx[(ntr+1):(ntr+ntst)]
      list(
        "trainidx" = trainidx,
        "testidx" = testidx,
        "xtrain" = dx[trainidx, , drop=F],
        "ytrain" = u[trainidx],
        "xtest" = dx[testidx, , drop=F],
        "ytest" = u[testidx]
      )
    })
  })
  names(dat) <- paste0("rep", seq_len(nrep))
  save(dat, file = datapath) # named just "dat"
}
# just to check
stopifnot(length(dat) == nrep)
stopifnot(length(dat[[1]][[1]]$ytrain) == ntr)
stopifnot(length(dat[[1]][[1]]$ytest) == ntst)

# parameters confirmed
# number of random repeats
nrep
```

    ## [1] 50

``` r
# binary classification tasks
tasklist <- names(dat[[1]])
tasklist
```

    ## [1] "age"

``` r
ntask <- length(tasklist)
```

Subsampled datasets are provided in ../data/dat\_eubm.RData. To summarize the preprocessing above,

-   First, we extracted information from the eubm data, where a total of 1.221610^{4} participants gave full ranks in the order of preference 6 sources of news regarding scientific developments: TV, radio, newspapers and magazines, scientific magazines, the internet, school/university.
-   Then, we created 1 binary classification task(s) predicting participant's demographic information that is the age group (over or under 40 years old) of the participants.
-   Finally, we randomly generate 50 times train-test splits of data, where for each fold there are 400 training instances vs 100 test instances.

In order to perform classification, we chose to fit kernel SVMs with different kernels and compare. Further, the parameters set to run experiments in `param` as follows:

``` r
# kernels defined in file R/func.R
# NOTE each kf has to START with ^dot_
kflist <- ls(pattern = "^dot_")
# add kendall at pos k ranging from 1 to (p-1)
key <- "dot_kenat"
if (key %in% kflist) {
  kflist <- c(setdiff(kflist, key), paste0(key, seq_len(p-1)))
}
# show kflist
kflist
```

    ##  [1] "dot_aken"          "dot_aosq1"         "dot_aosq2"        
    ##  [4] "dot_ken"           "dot_rr"            "dot_svdsq1"       
    ##  [7] "dot_svdsq2"        "dot_wken_dcg_add"  "dot_wken_dcg_mult"
    ## [10] "dot_wken_hb_add"   "dot_wken_hb_mult"  "dot_kenat1"       
    ## [13] "dot_kenat2"        "dot_kenat3"        "dot_kenat4"       
    ## [16] "dot_kenat5"

``` r
# parameters set to run experiments
param <- expand.grid(
  "dname" = as.character(dname), # data
  "task" = as.character(tasklist), # binary classification tasks
  "irep" = as.integer(seq_len(nrep)), # rep idx
  "kfname" = as.character(kflist), # kernel
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
)
head(param)
```

    ##   dname task irep   kfname
    ## 1  eubm  age    1 dot_aken
    ## 2  eubm  age    2 dot_aken
    ## 3  eubm  age    3 dot_aken
    ## 4  eubm  age    4 dot_aken
    ## 5  eubm  age    5 dot_aken
    ## 6  eubm  age    6 dot_aken

``` r
dim(param)
```

    ## [1] 800   4

II. Classification of ranking data
==================================

III. Results
============

IV. Session info
================

``` r
devtools::session_info()
```

    ##  setting  value                       
    ##  version  R version 3.4.3 (2017-11-30)
    ##  system   x86_64, darwin15.6.0        
    ##  ui       X11                         
    ##  language (EN)                        
    ##  collate  en_US.UTF-8                 
    ##  tz       Europe/London               
    ##  date     2018-06-27                  
    ## 
    ##  package    * version date       source         
    ##  backports    1.1.2   2017-12-13 CRAN (R 3.4.3) 
    ##  base       * 3.4.3   2017-12-07 local          
    ##  colorspace   1.3-2   2016-12-14 CRAN (R 3.4.0) 
    ##  combinat     0.0-8   2012-10-29 CRAN (R 3.4.0) 
    ##  compiler     3.4.3   2017-12-07 local          
    ##  corrplot   * 0.84    2017-10-16 CRAN (R 3.4.2) 
    ##  datasets   * 3.4.3   2017-12-07 local          
    ##  devtools     1.13.5  2018-02-18 CRAN (R 3.4.3) 
    ##  digest       0.6.15  2018-01-28 CRAN (R 3.4.3) 
    ##  evaluate     0.10.1  2017-06-24 CRAN (R 3.4.1) 
    ##  ggplot2    * 2.2.1   2016-12-30 CRAN (R 3.4.0) 
    ##  graphics   * 3.4.3   2017-12-07 local          
    ##  grDevices  * 3.4.3   2017-12-07 local          
    ##  grid         3.4.3   2017-12-07 local          
    ##  gtable       0.2.0   2016-02-26 CRAN (R 3.4.0) 
    ##  highr        0.6     2016-05-09 CRAN (R 3.4.0) 
    ##  htmltools    0.3.6   2017-04-28 CRAN (R 3.4.0) 
    ##  kernlab    * 0.9-26  2018-04-30 cran (@0.9-26) 
    ##  kernrank   * 1.1.0   2018-06-26 local          
    ##  knitr        1.20    2018-02-20 CRAN (R 3.4.3) 
    ##  lazyeval     0.2.1   2017-10-29 CRAN (R 3.4.2) 
    ##  magrittr     1.5     2014-11-22 CRAN (R 3.4.0) 
    ##  memoise      1.1.0   2017-04-21 CRAN (R 3.4.0) 
    ##  methods      3.4.3   2017-12-07 local          
    ##  munsell      0.4.3   2016-02-13 CRAN (R 3.4.0) 
    ##  pillar       1.2.1   2018-02-27 CRAN (R 3.4.3) 
    ##  plyr         1.8.4   2016-06-08 CRAN (R 3.4.0) 
    ##  Rcpp         0.12.17 2018-05-18 cran (@0.12.17)
    ##  rlang        0.2.0   2018-02-20 CRAN (R 3.4.3) 
    ##  rmarkdown    1.9     2018-03-01 CRAN (R 3.4.3) 
    ##  rprojroot    1.3-2   2018-01-03 CRAN (R 3.4.3) 
    ##  rstudioapi   0.7     2017-09-07 CRAN (R 3.4.1) 
    ##  scales       0.5.0   2017-08-24 CRAN (R 3.4.1) 
    ##  stats      * 3.4.3   2017-12-07 local          
    ##  stringi      1.1.7   2018-03-12 CRAN (R 3.4.4) 
    ##  stringr      1.3.0   2018-02-19 CRAN (R 3.4.3) 
    ##  tibble       1.4.2   2018-01-22 CRAN (R 3.4.3) 
    ##  tools        3.4.3   2017-12-07 local          
    ##  utils      * 3.4.3   2017-12-07 local          
    ##  withr        2.1.1   2017-12-19 CRAN (R 3.4.3) 
    ##  yaml         2.1.18  2018-03-08 CRAN (R 3.4.4)
