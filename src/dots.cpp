// file:
//    dots.cpp
// description:
//    implements some dot (inner product) function for rank data

#include <Rcpp.h>
using namespace Rcpp;

//////////////////////////////////////// order-1

// [[Rcpp::export]]
double Cdot_sq1(IntegerVector x, IntegerVector y, NumericVector f) {
  /* suquan-1 */
  int n = x.size(), i = 0;
  double total = 0;
  
  /* for i */
  for (i = 0; i < n; ++i) {
    total += f[x[i] - 1] * f[y[i] - 1] ;
  }
  return total;
}

// [[Rcpp::export]]
double dot_rr(IntegerVector x, IntegerVector y) {
  /* rank-rank */
  IntegerVector t = x * y;
  double total = sum(t);
  return total;
}

//////////////////////////////////////// order-2

// [[Rcpp::export]]
double Cdot_sq2(IntegerVector x, IntegerVector y, NumericMatrix f) {
  /* suquan-2 */
  int n = x.size(), i = 0, j = 0;
  double total = 0;
  
  /* for i , j */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      total += f(x[i] - 1, x[j] - 1) * f(y[i] - 1, y[j] - 1);
    }
  }
  return total;
}

// [[Rcpp::export]]
double dot_ken(IntegerVector x, IntegerVector y) {
  /* kendall*/
  int n = x.size(), i = 0, j = 0;
  double total = 0;
  
  /* for i < j */
  for (i = 0; i < n - 1; ++i) {
    for (j = i + 1; j < n; ++j) {
      if ((x[i] < x[j] && y[i] < y[j]) || (x[i] > x[j] && y[i] > y[j]))
        total += 1;
    }
  }
  return total;
}

// [[Rcpp::export]]
double Cdot_kenat(IntegerVector x, IntegerVector y, int k) {
  /* kendall at pos k */
  int n = x.size(), i = 0, j = 0;
  double total = 0;
  
  /* for i < j */
  for (i = 0; i < n - 1; ++i) {
    for (j = i + 1; j < n; ++j) {
      if ((k <= x[i] && x[i] < x[j] && k <= y[i] && y[i] < y[j]) || (x[i] > x[j] && x[j] >= k && y[i] > y[j] && y[j] >= k))
        total += 1;
    }
  }
  return total;
}

// [[Rcpp::export]]
double dot_aken(IntegerVector x, IntegerVector y) {
  /* average kendall */
  int n = x.size(), i = 0, j = 0;
  double total = 0;
  
  /* for i < j */
  for (i = 0; i < n - 1; ++i) {
    for (j = i + 1; j < n; ++j) {
      if (x[i] < x[j] && y[i] < y[j])
        total += std::min(x[i], y[i]);
      if (x[i] > x[j] && y[i] > y[j])
        total += std::min(x[j], y[j]);
    }
  }
  return total/n;
}

// [[Rcpp::export]]
double dot_wken_hb_add(IntegerVector x, IntegerVector y) {
  /* hyperbolic weighted kendall (additive) */
  int n = x.size(), i = 0, j = 0;
  double total = 0, tx = 0, ty = 0;
  
  /* for i < j */
  for (i = 0; i < n - 1; ++i) {
    for (j = i + 1; j < n; ++j) {
      tx = 1./(n+2-x[i]) + 1./(n+2-x[j]);
      ty = 1./(n+2-y[i]) + 1./(n+2-y[j]);
      if ((x[i] < x[j] && y[i] < y[j]) || (x[i] > x[j] && y[i] > y[j]))
        total += tx * ty;
    }
  }
  return total;
}

// [[Rcpp::export]]
double dot_wken_hb_mult(IntegerVector x, IntegerVector y) {
  /* hyperbolic weighted kendall (multiplicative) */
  int n = x.size(), i = 0, j = 0;
  double total = 0, tx = 0, ty = 0;
  
  /* for i < j */
  for (i = 0; i < n - 1; ++i) {
    for (j = i + 1; j < n; ++j) {
      tx = 1./(n+2-x[i]) * 1./(n+2-x[j]);
      ty = 1./(n+2-y[i]) * 1./(n+2-y[j]);
      if ((x[i] < x[j] && y[i] < y[j]) || (x[i] > x[j] && y[i] > y[j]))
        total += tx * ty;
    }
  }
  return total;
}

// [[Rcpp::export]]
double dot_wken_dcg_add(IntegerVector x, IntegerVector y) {
  /* DCG weighted kendall (additive) */
  int n = x.size(), i = 0, j = 0;
  double total = 0, tx = 0, ty = 0;
  
  /* for i < j */
  for (i = 0; i < n - 1; ++i) {
    for (j = i + 1; j < n; ++j) {
      tx = 1./(log2(n+2-x[i])) + 1./(log2(n+2-x[j]));
      ty = 1./(log2(n+2-y[i])) + 1./(log2(n+2-y[j]));
      if ((x[i] < x[j] && y[i] < y[j]) || (x[i] > x[j] && y[i] > y[j]))
        total += tx * ty;
    }
  }
  return total;
}

// [[Rcpp::export]]
double dot_wken_dcg_mult(IntegerVector x, IntegerVector y) {
  /* DCG weighted kendall (multiplicative) */
  int n = x.size(), i = 0, j = 0;
  double total = 0, tx = 0, ty = 0;
  
  /* for i < j */
  for (i = 0; i < n - 1; ++i) {
    for (j = i + 1; j < n; ++j) {
      tx = 1./(log2(n+2-x[i])) * 1./(log2(n+2-x[j]));
      ty = 1./(log2(n+2-y[i])) * 1./(log2(n+2-y[j]));
      if ((x[i] < x[j] && y[i] < y[j]) || (x[i] > x[j] && y[i] > y[j]))
        total += tx * ty;
    }
  }
  return total;
}

//////////////////////////////////////// M matrix for suquan-svd

// [[Rcpp::export]]
NumericMatrix mlda1(IntegerMatrix rankx, NumericVector ylda) {
  int i = 0, j = 0, n = rankx.nrow(), p = rankx.ncol();
  NumericMatrix m(p, p);
  
  /* for i over 1:n ; j (and k == PI_{j}) over 1:p */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < p; ++j) {
      m(rankx(i,j) - 1, j) += ylda[i];
    }
  }
  return m;
}

// [[Rcpp::export]]
NumericMatrix mlda2(IntegerMatrix rankx, NumericVector ylda) {
  int i = 0, j = 0, k = 0, n = rankx.nrow(), p = rankx.ncol();
  NumericMatrix m(p*p, p*p);
  
  /* for i over 1:n ; j and k (and l == PI_{jk}) over 1:p */
  for (i = 0; i < n; ++i) {
    for (j = 0; j < p; ++j) {
      for (k = 0; k < p; ++k) {
        m(p*(rankx(i,j)-1)+(rankx(i,k)-1), p*j+k) += ylda[i];
      }
    }
  }
  return m;
}
