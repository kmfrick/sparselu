#include <suitesparse/umfpack.h>
#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <iostream>

// [[Rcpp::export]]
Rcpp::List sparseLU(const std::vector<int> &Ap, const std::vector<int> &Ai, const std::vector<double> &Ax) {
  int n = Ap.size() - 1;
  int m = *std::max_element(Ai.begin(), Ai.end()) + 1;
  double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO];
  umfpack_di_defaults(Control);
  Control[UMFPACK_PRL] = 2;
  Control[UMFPACK_SCALE] = 0;
  Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;

  void *symbolic, *numeric;
  int status;
  status = umfpack_di_symbolic(m, n, Ap.data(), Ai.data(), Ax.data(), &symbolic, Control, Info);
  if (status != UMFPACK_OK) {
    umfpack_di_report_status(Control, status);
    if (status < 0) {
      Rcpp::stop("umfpack_di_symbolic failed with status %d", status);
    }
  }

  status = umfpack_di_numeric(Ap.data(), Ai.data(), Ax.data(), symbolic, &numeric, Control, Info);
  if (status != UMFPACK_OK) {
    umfpack_di_report_status(Control, status);
    if (status < 0) {
      Rcpp::stop("umfpack_di_numeric failed with status %d", status);
      umfpack_di_free_symbolic(&symbolic);
    }
  }

  umfpack_di_free_symbolic(&symbolic);

  int lnz, unz, nz_udiag;
  status = umfpack_di_get_lunz(&lnz, &unz, &m, &n, &nz_udiag, numeric);
  if (status != UMFPACK_OK) {
    umfpack_di_report_status(Control, status);
    if (status < 0) {
      Rcpp::stop("umfpack_di_get_lunz failed with status %d", status);
      umfpack_di_free_numeric(&numeric);
    }
  }


  std::vector<int> Lp(m+1), Lj(lnz), Up(n+1), Ui(unz), P(m), Q(n);
  std::vector<double> Lx(lnz), Ux(unz), D(std::min(m, n)), Rs(m);
  int do_recip;

  status = umfpack_di_get_numeric(Lp.data(), Lj.data(), Lx.data(), Up.data(), Ui.data(), Ux.data(), P.data(), Q.data(), D.data(), &do_recip, Rs.data(), numeric);
  if (status != UMFPACK_OK) {
    umfpack_di_report_status(Control, status);
    if (status < 0) {
      umfpack_di_free_numeric(&numeric);
      Rcpp::stop("umfpack_di_get_numeric failed with status %d", status);
    }
  }

  umfpack_di_free_numeric(&numeric);

  Rcpp::List ret = Rcpp::List::create(
      Rcpp::Named("L") = Rcpp::List::create(Rcpp::Named("j") = Lj, Rcpp::Named("p") = Lp, Rcpp::Named("x") = Lx),
      Rcpp::Named("U") = Rcpp::List::create(Rcpp::Named("i") = Ui, Rcpp::Named("p") = Up, Rcpp::Named("x") = Ux),
      Rcpp::Named("P") = P,
      Rcpp::Named("Q") = Q
      );


  return ret;
}

//[[Rcpp::export]]
Rcpp::NumericVector sparseLU_solve(const std::vector<int> &Ap, const std::vector<int> &Ai, const std::vector<double> &Ax, const std::vector<double> &b) {
  int n = b.size();
  int m = b.size();
  double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO];
  umfpack_di_defaults(Control);
  Control[UMFPACK_PRL] = 2;

  void *symbolic, *numeric;
  int status;
  status = umfpack_di_symbolic(m, n, Ap.data(), Ai.data(), Ax.data(), &symbolic, Control, Info);
  if (status != UMFPACK_OK) {
    umfpack_di_report_status(Control, status);
    if (status < 0) {
      Rcpp::stop("umfpack_di_symbolic failed with status %d", status);
    }
  }

  status = umfpack_di_numeric(Ap.data(), Ai.data(), Ax.data(), symbolic, &numeric, Control, Info);
  if (status != UMFPACK_OK) {
    umfpack_di_report_status(Control, status);
    if (status < 0) {
      Rcpp::stop("umfpack_di_numeric failed with status %d", status);
      umfpack_di_free_symbolic(&symbolic);
    }
  }

  umfpack_di_free_symbolic(&symbolic);

  std::vector<double> x(n);
  status = umfpack_di_solve(UMFPACK_A, Ap.data(), Ai.data(), Ax.data(), x.data(), b.data(), numeric, Control, Info);
  if (status != UMFPACK_OK) {
    umfpack_di_report_status(Control, status);
    if (status < 0) {
      umfpack_di_free_numeric(&numeric);
      Rcpp::stop("umfpack_di_get_numeric failed with status %d", status);
    }
  }

  umfpack_di_free_numeric(&numeric);

  return Rcpp::wrap(x);

}

