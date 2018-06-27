# file:
#   func.R
# description:
#   implements utile functions and classifiers for rank data

# Rcpp::sourceCpp("../src/dots.cpp") # some dot functions are implemented in cpp
# library(kernrank) # for fast kendall and weighted kendall
# library(kernlab) # for svm and kernels
# library(methods) # needed to run on cluster

# generic -----------------------------------------------------------------

perfSVM <- function(kfname, 
                    xtrain, 
                    ytrain, 
                    xtest, 
                    ytest, 
                    Cpara_list = 10^(-3:3), 
                    nfolds = 5, 
                    nrepeats = 1, 
                    seed = 99490066, 
                    ..., 
                    do.centerscale = FALSE, 
                    kernel.trick = FALSE)
{
  # wrap up function to perform specific kernel SVM
  # mostly SVM-based models plus C-independent-but-kernel-matrix-dependent KFD method
  # kfname indicates which kernel to use, show full list by \code{ls(pattern="^dot_")}
  # KFD (kernel fisher discriminant) accompanies SVM as simple reference kernel machines
  # cross validation for C-parameter tuning, default ranging in log scale 10^(-3:3)
  # NOTE for suquan-based models, the C parameter C are set equal for w and f
  # kernel.trick should always be set FALSE otherwise kernlab::ksvm would malfunction
  # SUQUAN will never use kernel trick anyway in alt opt
  
  ntr <- nrow(xtrain)
  p <- ncol(xtrain)
  if (is.null(names(Cpara_list)))
    names(Cpara_list) <- paste('C', seq_along(Cpara_list), sep='')
  
  # xtrain,xtest must be integers sampled from seq(ncol(xtrain)),seq(ncol(xtest)) indicating ranks in each row and no ties allowed
  stopifnot(all(apply(rbind(xtrain,xtest), 1, function(u) sort(u) == seq(p))))
  
  set.seed(seed)
  # make cv folds
  foldIndices <- lapply(1:nrepeats, function(i){
    test.fold <- split(sample(ntr, replace = FALSE), rep(1:nfolds, length = ntr))
    train.fold <- lapply(test.fold, function(testid){
      setdiff(1:ntr, testid)
    })
    return(train.fold)
  })
  foldIndices <- unlist(foldIndices, recursive = FALSE, use.names = FALSE)
  names(foldIndices) <- as.vector(outer(
    paste0("Fold", formatC(1:nfolds, width = nchar(nfolds), format = "d", flag = "0")), 
    paste0("Rep", formatC(1:nrepeats, width = nchar(nrepeats), format = "d", flag = "0")), 
    FUN = paste, sep = "."))
  
  if (grepl("^dot_.+sq[1-9]$", kfname)) {
    ##
    ## supervised kernel function
    ##
    
    # get sample order from sample rank
    r.xtrain <- xtrain
    r.xtest <- xtest
    o.xtrain <- t(apply(r.xtrain, 1, function(v) {match(seq(p),v)}))
    o.xtest <- t(apply(r.xtest, 1, function(v) {match(seq(p),v)}))
    
    message(" CV ", appendLF = FALSE)
    foldscore <- lapply(foldIndices, function(fold){
      message('+', appendLF = FALSE)
      sapply(Cpara_list, function(cpm){
        message('.', appendLF = FALSE)
        pred <- classifierSUQUAN(r.x = r.xtrain, o.x = o.xtrain, 
                                 idx = fold, y = ytrain[fold], cpm = cpm, 
                                 kfname = kfname, do.centerscale = do.centerscale, ...)$pred
        evaluateAcc(pred, ytrain[-fold])
      })
    })
    cvacc <- apply(as.matrix(as.data.frame(foldscore)), 1, mean, na.rm = TRUE)
    names(cvacc) <- names(Cpara_list) # SVM cv score (varying C)
    
    message(" IV ", appendLF = FALSE)
    # SVM
    best.cpm <- Cpara_list[which.max(cvacc)]
    res <- classifierSUQUAN(r.x = rbind(r.xtrain, r.xtest), o.x = rbind(o.xtrain, o.xtest), 
                            idx = 1:ntr, y = ytrain, cpm = best.cpm, 
                            kfname = kfname, do.centerscale = do.centerscale, ...)
    acc <- evaluateAcc(res$pred, ytest)
    
    # KFD with svm-learned kernel matrix
    res_kfd <- classifierKFD(km = res$kmat, trainidx = 1:ntr, traingrp = ytrain, ...)
    acc_kfd <- evaluateAcc(res_kfd$pred, ytest)
    
    if (!is.null(res$iter.res)) {
      iter.acc <- sapply(lapply(res$iter.res,"[[","pred"), evaluateAcc, observations = ytest)
      names(iter.acc) <- paste0("iter", sapply(res$iter.res,"[[","iter"))
    } else {
      iter.acc <- NULL
    }
    
    message(" DONE ! \n")
    
    return(list(
      "kfname" = kfname, 
      "Cpara_list" = Cpara_list, 
      "res" = res, 
      "cvacc" = cvacc, 
      "acc" = acc,
      "iter.acc" = iter.acc, 
      "res_kfd" = res_kfd, 
      "cvacc_kfd" = NULL, 
      "acc_kfd" = acc_kfd
    ))
  } else {
    ##
    ## deterministic kernel function
    ##
    
    message(" Computing kernel matrix ")
    if (grepl("^dot_kenat[0-9]+", kfname)) {
      k <- as.integer(gsub("^[^0-9]+", "", kfname))
      kf <- dot_kenat(k)
      mapf <- map_kenat(k)
    } else {
      kf <- get(kfname, mode = "function")
      mapf <- get(gsub("^dot_","map_",kfname), mode = "function")
    }
    
    if (kernel.trick) {
      kmat <- computeKernelMatrix(xdata = rbind(xtrain, xtest), kf = kf)
      xmat <- NULL
      kf <- NULL
    } else {
      kmat <- NULL
      xmat <- mapf(rbind(xtrain, xtest))
      kf <- 'vanilladot' # linear svm with explicit mapping
    }
    
    message(" CV ", appendLF = FALSE)
    foldscore <- lapply(foldIndices, function(fold){
      message('+', appendLF = FALSE)
      
      # SVM
      s <- sapply(Cpara_list, function(cpm){
        message('.', appendLF = FALSE)
        pred <- classifierSVM(km = kmat[1:ntr, 1:ntr], 
                              xm = xmat[1:ntr, , drop=F], kf = kf, 
                              trainidx = fold, traingrp = ytrain[fold], cpm = cpm, 
                              do.centerscale = do.centerscale, ...)$pred
        evaluateAcc(pred, ytrain[-fold])
      })
      
      s
    })
    cvacc <- apply(as.matrix(as.data.frame(foldscore)), 1, mean, na.rm = TRUE)
    names(cvacc) <- names(Cpara_list) # SVM cv score (varying C)
    
    message(" IV ", appendLF = FALSE)
    # SVM
    best.cpm <- Cpara_list[which.max(cvacc)]
    res <- classifierSVM(km = kmat, 
                         xm = xmat, kf = kf, 
                         trainidx = 1:ntr, traingrp = ytrain, cpm = best.cpm, 
                         do.centerscale = do.centerscale, ..., keep.km = TRUE)
    acc <- evaluateAcc(res$pred, ytest)
    
    # KFD
    res_kfd <- classifierKFD(km = res$kmat, trainidx = 1:ntr, traingrp = ytrain, ...)
    acc_kfd <- evaluateAcc(res_kfd$pred, ytest)
    
    message(" DONE ! \n")
    
    return(list(
      "kfname" = kfname, 
      "Cpara_list" = Cpara_list, 
      "res" = res, 
      "cvacc" = cvacc, 
      "acc" = acc, 
      "res_kfd" = res_kfd, 
      "cvacc_kfd" = NULL, 
      "acc_kfd" = acc_kfd
    ))
  }
}


# map and dot functions ---------------------------------------------------

# generic weighted kendall function
Rmap_wken <- function(x,
                      u)
{
  # u(x_i,x_j) defines deterministic weights over 1_{x_i<x_j}
  
  stopifnot(is.matrix(x))
  ind <- expand.grid(1:ncol(x), 1:ncol(x))
  ind <- subset(ind, Var1 != Var2)
  xwken <- t(u[cbind(as.vector(t(x[ ,ind[ ,1]])), as.vector(t(x[ ,ind[ ,2]])))] * t(x[ ,ind[ ,1]] < x[ ,ind[ ,2]])) # weighted kendall
  rownames(xwken) <- rownames(x)
  colnames(xwken) <- NULL
  xwken
}

# RANK-RANK
# dot_rr in dots.cpp OR worse implemented as follows
# dot_rr <- function(x, y)
# {
#   stopifnot(is.vector(x) && is.numeric(x) && 
#               is.vector(y) && is.numeric(y) && 
#               length(x) == length(y))
#   return(crossprod(rank(x), rank(y)))
# }
map_rr <- function(x)
{
  stopifnot(is.matrix(x))
  x
}

# KENDALL
# dot_ken in dots.cpp OR worse implemented as follows
# dot_ken <- function(x, y)
# {
#   stopifnot(is.vector(x) && is.numeric(x) && 
#               is.vector(y) && is.numeric(y) && 
#               length(x) == length(y))
#   return((cor(x, y, use = "everything", method = "kendall") + 1) / 2 * choose(length(x), 2))
# }
map_ken <- function(x)
{
  stopifnot(is.matrix(x))
  u <- rep(1, ncol(x))
  u <- tcrossprod(u)
  Rmap_wken(x, u)
}

# WEIGHTED KENDALL WITH ADDITIVE HYPERBOLIC WEIGHTS
# dot_wken_hb_add in dots.cpp
map_wken_hb_add <- function(x)
{
  stopifnot(is.matrix(x))
  u <- 1/(1+(ncol(x):1))
  u <- outer(u, u, FUN = "+")
  Rmap_wken(x, u)
}

# WEIGHTED KENDALL WITH MULTIPLICATIVE HYPERBOLIC WEIGHTS
# dot_wken_hb_mult in dots.cpp
map_wken_hb_mult <- function(x)
{
  stopifnot(is.matrix(x))
  u <- 1/(1+(ncol(x):1))
  u <- outer(u, u, FUN = "*")
  Rmap_wken(x, u)
}

# WEIGHTED KENDALL WITH ADDITIVE DCG WEIGHTS
# dot_wken_dcg_add in dots.cpp
map_wken_dcg_add <- function(x)
{
  stopifnot(is.matrix(x))
  u <- 1/log2(1+(ncol(x):1))
  u <- outer(u, u, FUN = "+")
  Rmap_wken(x, u)
}

# WEIGHTED KENDALL WITH MULTIPLICATIVE DCG WEIGHTS
# dot_wken_dcg_mult in dots.cpp
map_wken_dcg_mult <- function(x)
{
  stopifnot(is.matrix(x))
  u <- 1/log2(1+(ncol(x):1))
  u <- outer(u, u, FUN = "*")
  Rmap_wken(x, u)
}

# KENDALL AT TOP-K
dot_kenat <- function(k){
  # wrapper for Cdot_kenat
  function(x, y){
    Cdot_kenat(x, y, k)
  }
}
map_kenat <- function(k)
{
  function(x){
    stopifnot(is.matrix(x))
    u <- rep(1, ncol(x))
    if (k >= 2)
      u[seq(k-1)] <- 0
    u <- tcrossprod(u)
    Rmap_wken(x, u)
  }
}

# AVERAGE KENDALL
# dot_aken in dots.cpp
map_aken <- function(x)
{
  stopifnot(is.matrix(x))
  xatk <- lapply(seq(ncol(x)-1), function(k) map_kenat(k)(x))
  do.call("cbind", xatk)/sqrt(ncol(x))
}

# series of SUQUAN (Kendall) kernels
# note that the corresp map functions are nested in the implementation of \code{classifierSUQUAN}
# SUQUAN of order 1, Alternate Optimization (Le Morvan and Vert, 2017)
dot_aosq1 <- function(f){
  # wrapper for Cdot_sq1
  function(x, y){
    Cdot_sq1(x, y, f)
  }
}

# SUQUAN of order 1, SVD-based algorithm (Le Morvan and Vert, 2017)
dot_svdsq1 <- dot_aosq1

# SUQUAN of order 2, Alternate Optimization (Jiao and Vert, 2018)
dot_aosq2 <- function(f){
  # wrapper for Cdot_sq2
  function(x, y){
    Cdot_sq2(x, y, f)
  }
}

# SUQUAN of order 2, SVD-based algorithm (Jiao and Vert, 2018)
dot_svdsq2 <- dot_aosq2

# utiles ------------------------------------------------------------------

computeKernelMatrix <- function(xdata, 
                                kf,
                                ...)
{
  #' @param xdata n x p data matrix
  
  n <- nrow(xdata)
  samplenames <- rownames(xdata)
  
  stopifnot(is(kf, "function"))
  class(kf) <- 'kernel'
  kmat <- kernelMatrix(kf, as.matrix(xdata), ...)
  dimnames(kmat) <- list(samplenames, samplenames)
  
  return(as.kernelMatrix(kmat))
}

centerScaleKernelMatrix <- function(kmat, 
                                    trainidx)
{
  # NOTE centering is done feature-wise and scaling is done SAMPLE-wise
  n <- nrow(kmat)
  ntr <- length(trainidx)
  nts <- n - ntr
  
  #centering
  ed <- matrix(0, nrow = n, ncol = n)
  ed[trainidx,] <- 1
  kmcs <- kmat - t(ed) %*% kmat / ntr - kmat %*% ed / ntr + t(ed) %*% kmat %*% ed / (ntr^2)
  #scaling
  dsr <- sqrt(diag(kmcs))
  kmcs <- sweep(
    sweep(kmcs, 1, dsr, FUN = '/'),
    2, dsr, FUN = '/')
  rownames(kmcs) <- rownames(kmat)
  colnames(kmcs) <- colnames(kmat)
  
  return(as.kernelMatrix(kmcs))
}

centerScaleData <- function(xmat,
                            trainidx)
{
  # NOTE centering is done feature-wise and scaling is done SAMPLE-wise
  v.cen <- colMeans(xmat[trainidx, ,drop=F])
  xmcs <- t(t(xmat) - v.cen)
  xmcs <- xmcs / sqrt(apply(xmcs, 1, crossprod))
  xmcs
}

evaluateAcc <- function(predictions,
                        observations)
{
  stopifnot(is(predictions,"factor") && 
              all(levels(predictions) == levels(observations)) &&
              length(predictions) == length(observations))
  sum(predictions == observations, na.rm = TRUE) / length(observations)
}

# classifiers -------------------------------------------------------------

classifierKFD <- function(km, 
                          trainidx, 
                          traingrp, 
                          mu = 1e-3, 
                          ...)
{
  # wrap up function of Kernel Fisher Discriminant
  # mu is a small number added to diag(N) to avoid singularity
  # classifierKFD returns a ntst-vector of predicted binary classes
  # NOTE those indices not in trainidx is ready for test!
  # NOTE that (row/sample) NAMES of kernel matrix is necessary for naming predicted vector
  
  ntot <- nrow(km)
  ntrn <- length(trainidx)
  ntst <- ntot - ntrn
  traingrp <- factor(traingrp)
  classes <- levels(traingrp)
  if(length(classes) != 2)
    stop('Training response should be two classes!')
  
  tab <- table(traingrp)
  if(sum(tab) == 0)
    stop('Wrong input!')
  else if(tab[1] == 0){
    pred_response <- factor(rep(classes[2], ntst), levels = classes)
    return(pred_response)
  } else if(tab[2] == 0){
    pred_response <- factor(rep(classes[1], ntst), levels = classes)
    return(pred_response)
  }
  
  idx1 <- trainidx[traingrp == classes[1]]
  idx2 <- trainidx[traingrp == classes[2]]
  idxtst <- setdiff(1:ntot, trainidx)
  testsamples <- rownames(km)[idxtst]
  
  M1 <- apply(km[trainidx, idx1, drop = F], 1, mean)
  M2 <- apply(km[trainidx, idx2, drop = F], 1, mean)
  
  L1 <- matrix(1/length(idx1), nrow = length(idx1), ncol = length(idx1))
  L2 <- matrix(1/length(idx2), nrow = length(idx2), ncol = length(idx2))
  N <- km[trainidx, trainidx, drop = F] %*% km[trainidx, trainidx, drop = F] - 
    km[trainidx, idx1, drop = F] %*% L1 %*% km[idx1, trainidx, drop = F] - 
    km[trainidx, idx2, drop = F] %*% L2 %*% km[idx2, trainidx, drop = F]
  
  # case I: being adjusted by within-class variance i.e. KFD
  alpha <- solve(N + diag(mu, nrow = nrow(N), ncol = ncol(N))) %*% matrix(M2 - M1, ncol = 1)
  mo <- list(
    "M1" = M1,
    "M2" = M2,
    "N" = N
  )
  # # case II: without being  adjusted by within-class variance
  # alpha <- matrix(M2 - M1, ncol=1)
  
  pred <- as.vector(km[idxtst, trainidx, drop = F] %*% matrix(alpha, ncol = 1))
  pred_response <- as.vector(as.numeric(pred >= 0))
  pred_response <- factor(pred_response, levels = c(0, 1), labels = classes)
  names(pred_response) <- testsamples
  
  return(list(
    "model" = mo,
    "pred" = pred_response
  ))
}

classifierSVM <- function(km = NULL, 
                          xm = NULL, 
                          kf = 'vanilladot', 
                          trainidx, 
                          traingrp, 
                          cpm, 
                          ..., 
                          do.centerscale = FALSE, 
                          keep.km = FALSE)
{
  # wrap up function of SVM
  # classifierSVM returns a ntst-vector of predicted binary classes
  # NOTE if both data and kernel matrix are provided, kernel matrix will be used
  # NOTE center-scale are never done in here! provide pre-centerscaled data for training and test
  # NOTE those indices not in trainidx is ready for test!
  # NOTE that (row/sample) NAMES of kernel matrix is necessary for naming predicted vector
  
  if (is.null(km)) {
    stopifnot(!is.null(xm) && kf == 'vanilladot')
    if (is(kf, "function"))
      class(kf) <- "kernel"
    if (do.centerscale)
      xm <- centerScaleData(xmat = xm, trainidx = trainidx)
    mo <- ksvm(x = xm[trainidx, , drop=F], y = factor(traingrp), 
               kernel = kf, scaled = FALSE, C = cpm, type = "C-svc", ...)
    pred <- predict(object = mo, 
                    newdata = xm[-trainidx, , drop=F], 
                    type = "response")
    names(pred) <- rownames(xm)[-trainidx]
    if (keep.km)
      km <- tcrossprod(xm)
  } else {
    if (do.centerscale)
      km <- centerScaleKernelMatrix(kmat = km, trainidx = trainidx)
    mo <- ksvm(x = as.kernelMatrix(km[trainidx, trainidx, drop=F]), y = factor(traingrp), 
               kernel = "matrix", scaled = FALSE, C = cpm, type = "C-svc", ...)
    pred <- predict(object = mo, 
                    newdata = as.kernelMatrix(km[-trainidx,trainidx,drop=F][ ,SVindex(mo),drop=F]), 
                    type = "response")
    names(pred) <- rownames(km)[-trainidx]
  }
  
  return(list(
    "model" = mo,
    "pred" = pred,
    "kmat" = km
  ))
}


classifierSUQUAN <- function(r.x, # train and test rankx
                             o.x, # train and test orderx
                             idx, # trainidx in r.x and o.x and the rest are testidx
                             y, # train y
                             cpm,
                             kfname,
                             ...,
                             maxiter = 50, # only for ao suquan
                             init = c("classic", "svdsq"), # only for ao suquan
                             do.centerscale = FALSE,
                             eps = 1e-3,
                             track.iter = TRUE)
{
  # The weighted kernel SVM with supervised weights optimized jointly with prediction coefficients
  # Use the wrap up function \code{perfSVM} with "dot_[ao|svd]sq[1|2]" that calls this function
  # 
  # @param
  # r.x Matrix of rankings with seqs in rows, of size \code{(ntr+ntst)*p}
  # o.x Matrix of orders with seqs in rows, of size \code{(ntr+ntst)*p}
  # idx Vector of indices of training seqs, of size \code{ntr}
  # y Vector of binary responses, of size \code{ntr}
  # cpm C parameter in SVM
  # kfname See @param of \code{perfSVM}
  # 
  # @reference
  # Jiao and Vert. "The Weighted Kendall and High-order Kernels for Permutations." ICML, 2018.
  # 
  # @note
  # SUQUAN convergence is only guaranteed on running kernlab::ksvm(x = xdata, kernel='vanilladot', scaled=F), therefore alt opt
  # always run on unscaled data but set do.centerscale=TRUE will retrain a model with final output of the learned weights on CS data
  # 
  # @example
  # # source('../../src/func.R', chdir = T)
  # set.seed(2345)
  # n <- 2000
  # p <- 6
  # ntr <- 1000
  # x <- matrix(rnorm(n*p), n, p)
  # r.x <- t(apply(x, 1, rank))
  # o.x <- t(apply(x, 1, order))
  # idx <- sample(n,ntr)
  # cpm <- 1
  # y <- factor(rbinom(ntr,1,0.5), labels = c("n","y"))
  # res <- classifierSUQUAN(r.x, o.x, idx, y, cpm, "dot_aosq2", maxiter=50)
  # all(diff(sapply(res$iter.res, "[[", "objval")) < 1e-3)
  # max(abs(res$f - res$iter.res[[length(res$iter.res)]]$f))
  # max(abs(res$w - res$iter.res[[length(res$iter.res)]]$w))
  # 
  
  init <- match.arg(init)
  sqorder <- as.numeric(substr(kfname, nchar(kfname), nchar(kfname)))
  stopifnot(sqorder %in% 1:2)
  
  # always train on linear svm
  kmat <- NULL
  kf <- 'vanilladot'
  
  # reset param for svd
  svdflag <- grepl("svd", kfname)
  if (svdflag) {
    maxiter <- 1 # do NOT change
    init <- "svdsq" # do NOT change
    track.iter <- FALSE # no need to change
  }
  
  # number of training samples and features
  ntr <- length(idx)
  p <- ncol(r.x)
  
  # create bin +/- 1 from factor y
  classes <- levels(factor(y))
  stopifnot(length(classes) == 2)
  idx1 <- which(y == classes[1])
  idx2 <- which(y == classes[2])
  ybin <- numeric(length(y))
  ybin[idx1] <- -1
  ybin[idx2] <- 1
  
  # some useful functions
  dims <- function(p. = p, sqorder. = sqorder){
    if (sqorder. == 1) {
      NULL
    } else if (sqorder. == 2) {
      c(p., p.)
    }
  }
  
  f.init <- function(sqorder. = sqorder, svdflag. = svdflag, p. = p, 
                     r.x. = r.x[idx, , drop=F], y. = y){
    if (init == "classic") {
      if (sqorder. == 1) {
        # recovers rank-rank kernel
        f <- seq_len(p.)
      } else if (sqorder. == 2) {
        # recovers kendall kernel
        f <- matrix(0, p., p.)
        f[upper.tri(f)] <- 1
      }
    } else if (init == "svdsq") {
      classes <- levels(factor(y.))
      idx1 <- which(y. == classes[1])
      idx2 <- which(y. == classes[2])
      ybin.lda <- numeric(length(y.))
      ybin.lda[idx1] <- -1/length(idx1)
      ybin.lda[idx2] <- 1/length(idx2)
      # call mlda[1|2] from dots.cpp
      M <- get(paste0("mlda", sqorder.), mode = "function")(r.x., ybin.lda)
      # do SVD over M for left singular vector to init f
      f <- svd(M, nu = 1, nv = 0)$u
      dim(f) <- dims()
    }
    f
  }
  
  map <- function(x, qf, sqorder. = sqorder){
    if (sqorder. == 1) {
      t(apply(x, 1, function(perm) qf[perm]))
    } else if (sqorder. == 2) {
      t(apply(x, 1, function(perm) as.vector(qf[perm, perm])))
    }
  }
  
  coef.ksvm <- function(mo, x, qf, xf, jj = 1){
    # mo trained ksvm model
    # x train data needed for computing center
    # qf quantile vector or matrix
    # jj needs to be altered when training multiclass ksvm or changed when others
    # can be quite slow as no kernel trick is explored for sake of explicit feature map
    # return a list of two entries $`coef,b,alpha`
    
    # w (coef) and b in primal
    stopifnot(length(res$model@ymatrix) == nrow(x))
    svid <- SVindex(mo)[which(SVindex(mo) %in% alphaindex(mo)[[jj]])]
    stopifnot(length(svid) == length(coef(mo)[[jj]]))
    coef <- colSums(xf[svid, , drop=F] * coef(mo)[[jj]])
    dim(coef) <- dims()
    
    # alpha in dual
    a <- numeric(length = nrow(x))
    a[alphaindex(mo)[[jj]]] <- alpha(mo)[[jj]]
    
    list("coef" = coef, "b" = b(mo)[jj], "alpha" = a)
  }
  
  margin <- function(xf, coefs) {
    colSums(t(xf) * as.vector(coefs$coef)) - coefs$b
  }
  
  objective <- function(x, qf, coefs, xf, ybin. = ybin, cpm. = cpm){
    cpm. * sum(pmax(0, 1 - ybin. * margin(xf = xf, coefs = coefs))) + 
      sum(coefs$coef * coefs$coef)/2 + sum(qf * qf)/2
  }
  
  dual.objective <- function(x, qf, coefs, xf, ybin. = ybin){
    sum(coefs$alpha) - 
      sum(coefs$alpha * ybin. * t(coefs$alpha * ybin. * tcrossprod(xf))) / 2 + 
      sum(qf * qf)/2
  }
  
  # main loop
  f <- f.init()
  w <- 0
  objval <- Inf
  iter.res <- list()
  for (iter in seq_len(maxiter)) {
    message("iter = ", iter)
    f.old <- f
    w.old <- w
    objval.old <- objval
    
    # optimize w for fixed f
    x.feat <- map(x = r.x, qf = f)
    res <- classifierSVM(km = kmat, 
                         xm = x.feat, kf = kf, 
                         trainidx = idx, traingrp = y, cpm = cpm, ..., 
                         do.centerscale = FALSE)
    coefs <- coef.ksvm(mo = res$model, x = r.x[idx, , drop=F], qf = f, xf = x.feat[idx, , drop=F])
    w <- coefs$coef
    b <- coefs$b
    objval <- objective(x = r.x[idx, ], qf = f, coefs = coefs, xf = x.feat[idx, , drop=F])
    dualobjval <- dual.objective(x = r.x[idx, , drop=F], qf = f, coefs = coefs, xf = x.feat[idx, , drop=F])
    
    # update res
    res$iter <- iter
    res$f <- f
    res$w <- w
    res$objval <- objval
    res$kmat <- tcrossprod(x.feat)
    
    
    # # f check start ------------------------------------------------
    # # at this step while f (saved to res$f) fixed
    # # w.old (saved to res$w.old) -> w (saved to res$w)
    # 
    # res$w.old <- w.old
    # res$ntr <- ntr
    # res$ybin <- ybin
    # 
    # res$model.f <- res$model
    # res$kmat.f <- res$kmat[idx, idx]
    # res$x.feat.f <- map(r.x[idx, ], f)
    # res$coefs.f <- coefs
    # res$margin.f <- margin(x.feat[idx, ], coefs)
    # res$hinge.f <- cpm * sum(pmax(0, 1 - ybin * res$margin.f))
    # res$obj.f <- res$hinge.f + sum(w * w)/2
    # res$norm.f <- sum(f * f)/2
    # res$obj.total.f <- res$obj.f + res$norm.f
    # stopifnot(res$obj.total.f == objval)
    # res$dualobj.f <- dualobjval - res$norm.f
    # 
    # # check margin with kernlab implementation
    # res$kernlab.margin.f <- drop(predict(
    #   res$model,
    #   as.kernelMatrix(res$kmat.f[ , SVindex(res$model), drop = F]),
    #   type = "decision"))
    # res$check.margin.f <- sum(abs(res$kernlab.margin.f - res$margin.f))
    # 
    # # check obj and dualobj with kernlab implementation
    # res$kernlab.obj.f <- -obj(res$model)
    # res$check.obj.f <- abs(res$kernlab.obj.f - res$obj.f)
    # res$check.dualobj.f <- abs(res$kernlab.obj.f - res$dualobj.f)
    # 
    # # check that obj.total with (w.old, bnew) must be equal to that last computed in the other loop with f fixed
    # if (exists("bnew")) {
    #   res$other.obj.total.f <- objective(r.x[idx, ], f, list("coef"=w.old,"b"=bnew))
    #   res$check.other.obj.total.f <- abs(res$other.obj.total.f - iter.res[[length(iter.res)]]$obj.total.w)
    # } else {
    #   res$other.obj.total.f <- NA
    #   res$check.other.obj.total.f <- 0
    # }
    # 
    # # check that obj.total with (w.old, b) should NOT be smaller than (w, b) with f fixed
    # res$old.obj.total.f <- objective(r.x[idx, ], f, list("coef"=w.old,"b"=b))
    # res$check.old.obj.total.f <- (res$old.obj.total.f < res$obj.total.f - eps)
    # 
    # # f check end ------------------------------------------------
    
    
    # Optimize f for fixed w
    x.feat <- map(x = o.x, qf = w)
    resnew <- classifierSVM(km = kmat, 
                            xm = x.feat, kf = kf, 
                            trainidx = idx, traingrp = y, cpm = cpm, ..., 
                            do.centerscale = FALSE)
    coefs <- coef.ksvm(mo = resnew$model, x = o.x[idx, , drop=F], qf = w, xf = x.feat[idx, , drop=F])
    f <- coefs$coef
    bnew <- coefs$b
    objvalnew <- objective(x = o.x[idx, , drop=F], qf = w, coefs = coefs, xf = x.feat[idx, , drop=F])
    dualobjvalnew <- dual.objective(x = o.x[idx, , drop=F], qf = w, coefs = coefs, xf = x.feat[idx, , drop=F])
    
    
    # # w check start ------------------------------------------------
    # # at this step while w (saved to res$w) fixed
    # # f.old (saved to res$f) -> f (saved to res$f.new)
    # 
    # res$f.new <- f
    # 
    # res$model.w <- resnew$model
    # res$kmat.w <- tcrossprod(x.feat)[idx, idx]
    # res$x.feat.w <- map(o.x[idx, ], w)
    # res$coefs.w <- coefs
    # res$margin.w <- margin(x.feat[idx, ], coefs)
    # res$hinge.w <- cpm * sum(pmax(0, 1 - ybin * res$margin.w))
    # res$obj.w <- res$hinge.w + sum(f * f)/2
    # res$norm.w <- sum(w * w)/2
    # res$obj.total.w <- res$obj.w + res$norm.w
    # stopifnot(res$obj.total.w == objvalnew)
    # res$dualobj.w <- dualobjvalnew - res$norm.w
    # 
    # # check margin with kernlab implementation
    # res$kernlab.margin.w <- drop(predict(
    #   resnew$model,
    #   as.kernelMatrix(res$kmat.w[ , SVindex(resnew$model), drop = F]),
    #   type = "decision"))
    # res$check.margin.w <- sum(abs(res$kernlab.margin.w - res$margin.w))
    # 
    # # check obj with kernlab implementation
    # res$kernlab.obj.w <- -obj(resnew$model)
    # res$check.obj.w <- abs(res$kernlab.obj.w - res$obj.w)
    # res$check.dualobj.w <- abs(res$kernlab.obj.w - res$dualobj.w)
    # 
    # # check that obj.total with (f.old, b) must be equal to that last computed in the other loop with w fixed
    # res$other.obj.total.w <- objective(o.x[idx, ], w, list("coef"=f.old,"b"=b))
    # res$check.other.obj.total.w <- abs(res$other.obj.total.w - res$obj.total.f)
    # 
    # # check that obj.total with (f.old, bnew) should NOT be smaller than (f, bnew) with w fixed
    # res$old.obj.total.w <- objective(o.x[idx, ], w, list("coef"=f.old,"b"=bnew))
    # res$check.old.obj.total.w <- (res$old.obj.total.w < res$obj.total.w - eps)
    # 
    # # w check end ------------------------------------------------
    
    
    # save res to iter.res
    iter.res[[iter]] <- res
    
    # check condition
    if (objval > objval.old + eps)
      warning("obj value increases at iter = ", iter)
    else if (objval.old - objval < eps)
      break
    
    # do not converge flag
    if (iter == maxiter)
      warning("f did not converge!")
  }
  
  if (do.centerscale) {
    f <- res$f
    x.feat <- map(x = r.x, qf = f)
    x.feat <- centerScaleData(xmat = x.feat, trainidx = idx)
    res <- classifierSVM(km = kmat, 
                         xm = x.feat, kf = kf, 
                         trainidx = idx, traingrp = y, cpm = cpm, ..., 
                         do.centerscale = TRUE)
    coefs <- coef.ksvm(mo = res$model, x = r.x[idx, , drop=F], qf = f, xf = x.feat[idx, , drop=F])
    w <- coefs$coef
    b <- coefs$b
    objval <- objective(x = r.x[idx, ], qf = f, coefs = coefs, xf = x.feat[idx, , drop=F])
    dualobjval <- dual.objective(x = r.x[idx, , drop=F], qf = f, coefs = coefs, xf = x.feat[idx, , drop=F])
    
    # update res
    res$iter <- iter
    res$f <- f
    res$w <- w
    res$objval <- objval
    res$kmat <- tcrossprod(x.feat)
  }
  
  if (track.iter) {
    res$iter.res <- iter.res
    names(res$iter.res) <- seq(iter)
  }
  return(res)
}
