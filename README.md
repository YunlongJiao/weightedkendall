# weightedkendall

This repo contains R code for reproducing results from the paper:

> Yunlong Jiao, Jean-Philippe Vert. "The Weighted Kendall and High-order Kernels for Permutations." arXiv preprint arXiv:1802.08526, 2018. [arXiv:1802.08526](https://arxiv.org/abs/1802.08526)

## Quick Start

See the compiled `results/notebook.md` for data, code and results for the numerical experiments of this study.

## Directories

The top level structure is as follows:

* `data/` - static RData to be studied, including
  - `fulldat_eubm.RData` the full Eurobarometer 55.2 data (provided upon request, or freely accessible via [DOI:10.3886/ICPSR03341.v3](https://doi.org/10.3886/ICPSR03341.v3)).
  - `dat_eubm.RData` the anonymized, randomized, and subsampled rank data to run the binary classification (provided upon request).
* `results/` - notebook of experimental results, including
  - `notebook.[md|Rmd]` the project notebook of experiments, with code in `notebook.Rmd`, compiled version in `notebook.md` and figures saved in `notebook_files/`.
  - `scores.txt` table of performance scores per experiment
  - `weights.txt` matrix of learned weights in a weighted kernel (see paper and notebook for details)
* `R/` - R code and general purpose scripts, including
  - `func.R` implements utile functions and classifiers for rank data.
* `src/` - C++ code and general purpose scripts, including
  - `dots.cpp` implements some dot (inner product) function for rank data.

## Hands-on

_Note: If you would like to reproduce the experiments, first request data access to the Eurobarometer 55.2 survey through the website [DOI:10.3886/ICPSR03341.v3](https://doi.org/10.3886/ICPSR03341.v3), then you should be able to process the data locally. Or alternatively send me an email (with data access confirmation attached), I will be happy to provide you the processed dataset._

In order to build the project notebook `results/notebook.md`, make sure your local machine has the following R packages installed (or run the corresponding commands to install them in R console):

```r
> require(kernrank)   # devtools::install_github("YunlongJiao/kernrank")
> require(kernlab)    # install.packages("kernlab")
> require(caret)      # install.packages("caret")
> require(ggplot2)    # install.packages("ggplot2")
> require(corrplot)   # install.packages("corrplot")
> require(rmarkdown)  # install.packages("rmarkdown")
```

Then run in shell,

```sh
$ git clone git@github.com:YunlongJiao/weightedkendall.git
$ cd weightedkendall/results/
$ Rscript -e "rmarkdown::render('notebook.Rmd', output_format = 'html_document')"
```

## Authors

* **Yunlong Jiao** - main contributor
