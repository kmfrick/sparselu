#include <suitesparse/umfpack.h>
#include <Rcpp.h>
#include <algorithm>
#include <vector>

namespace {

struct SymbolicHandle {
  void *ptr = nullptr;

  ~SymbolicHandle() {
    if (ptr != nullptr) {
      umfpack_di_free_symbolic(&ptr);
    }
  }
};

struct NumericHandle {
  void *ptr = nullptr;

  ~NumericHandle() {
    if (ptr != nullptr) {
      umfpack_di_free_numeric(&ptr);
    }
  }
};

void stop_if_bad_status(int status, const double *control, const char *where) {
  (void) control;
  if (status != UMFPACK_OK) {
    Rcpp::stop("%s failed with status %d", where, status);
  }
}

void validate_ccs(const std::vector<int> &Ap,
                  const std::vector<int> &Ai,
                  const std::vector<double> &Ax,
                  bool enforce_square,
                  int expected_dim) {
  if (Ap.size() < 2) {
    Rcpp::stop("`Ap` must contain at least two elements.");
  }
  if (Ai.size() != Ax.size()) {
    Rcpp::stop("`Ai` and `Ax` must have the same length.");
  }
  if (Ai.empty()) {
    Rcpp::stop("`Ai` and `Ax` must contain at least one non-zero entry.");
  }
  if (Ap.front() != 0) {
    Rcpp::stop("`Ap` must start at 0.");
  }
  if (Ap.back() != static_cast<int>(Ai.size())) {
    Rcpp::stop("The last entry of `Ap` must equal length(`Ai`) and length(`Ax`).");
  }

  for (std::size_t idx = 1; idx < Ap.size(); ++idx) {
    if (Ap[idx] < Ap[idx - 1]) {
      Rcpp::stop("`Ap` must be a non-decreasing sequence of column pointers.");
    }
  }
  for (int row : Ai) {
    if (row < 0) {
      Rcpp::stop("`Ai` must contain non-negative row indices.");
    }
  }

  if (enforce_square) {
    const int n = static_cast<int>(Ap.size()) - 1;
    if (n != expected_dim) {
      Rcpp::stop("`length(b)` must equal `length(Ap) - 1`.");
    }
    for (int row : Ai) {
      if (row >= expected_dim) {
        Rcpp::stop("`Ai` must be strictly less than `length(b)` for `sparseLU_solve()`.");
      }
    }
  }
}

} // namespace

// [[Rcpp::export]]
Rcpp::List sparseLU(const std::vector<int> &Ap, const std::vector<int> &Ai, const std::vector<double> &Ax) {
  validate_ccs(Ap, Ai, Ax, false, 0);

  int n = static_cast<int>(Ap.size()) - 1;
  int m = n;
  if (!Ai.empty()) {
    m = std::max(n, *std::max_element(Ai.begin(), Ai.end()) + 1);
  }

  double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO];
  umfpack_di_defaults(Control);
  Control[UMFPACK_PRL] = 2;
  Control[UMFPACK_SCALE] = 0;
  Control[UMFPACK_STRATEGY] = UMFPACK_STRATEGY_SYMMETRIC;

  SymbolicHandle symbolic;
  NumericHandle numeric;

  int status = umfpack_di_symbolic(m, n, Ap.data(), Ai.data(), Ax.data(), &symbolic.ptr, Control, Info);
  stop_if_bad_status(status, Control, "umfpack_di_symbolic");

  status = umfpack_di_numeric(Ap.data(), Ai.data(), Ax.data(), symbolic.ptr, &numeric.ptr, Control, Info);
  stop_if_bad_status(status, Control, "umfpack_di_numeric");

  int lnz, unz, nz_udiag;
  status = umfpack_di_get_lunz(&lnz, &unz, &m, &n, &nz_udiag, numeric.ptr);
  stop_if_bad_status(status, Control, "umfpack_di_get_lunz");

  std::vector<int> Lp(static_cast<std::size_t>(m) + 1U);
  std::vector<int> Lj(static_cast<std::size_t>(lnz));
  std::vector<int> Up(static_cast<std::size_t>(n) + 1U);
  std::vector<int> Ui(static_cast<std::size_t>(unz));
  std::vector<int> P(static_cast<std::size_t>(m));
  std::vector<int> Q(static_cast<std::size_t>(n));
  std::vector<double> Lx(static_cast<std::size_t>(lnz));
  std::vector<double> Ux(static_cast<std::size_t>(unz));
  std::vector<double> D(static_cast<std::size_t>(std::min(m, n)));
  std::vector<double> Rs(static_cast<std::size_t>(m));
  int do_recip;

  status = umfpack_di_get_numeric(Lp.data(), Lj.data(), Lx.data(), Up.data(), Ui.data(), Ux.data(),
                                  P.data(), Q.data(), D.data(), &do_recip, Rs.data(), numeric.ptr);
  stop_if_bad_status(status, Control, "umfpack_di_get_numeric");

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
  const int n = static_cast<int>(b.size());
  const int m = n;
  validate_ccs(Ap, Ai, Ax, true, n);

  double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO];
  umfpack_di_defaults(Control);
  Control[UMFPACK_PRL] = 2;

  SymbolicHandle symbolic;
  NumericHandle numeric;

  int status = umfpack_di_symbolic(m, n, Ap.data(), Ai.data(), Ax.data(), &symbolic.ptr, Control, Info);
  stop_if_bad_status(status, Control, "umfpack_di_symbolic");

  status = umfpack_di_numeric(Ap.data(), Ai.data(), Ax.data(), symbolic.ptr, &numeric.ptr, Control, Info);
  stop_if_bad_status(status, Control, "umfpack_di_numeric");

  std::vector<double> x(static_cast<std::size_t>(n));
  status = umfpack_di_solve(UMFPACK_A, Ap.data(), Ai.data(), Ax.data(), x.data(), b.data(),
                            numeric.ptr, Control, Info);
  stop_if_bad_status(status, Control, "umfpack_di_solve");

  return Rcpp::wrap(x);

}
