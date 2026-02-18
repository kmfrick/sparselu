#' @useDynLib sparselu, .registration = TRUE
#' @import Rcpp
NULL

.validate_ccs <- function(Ap, Ai, Ax, b = NULL) {
  Ap <- as.integer(Ap)
  Ai <- as.integer(Ai)
  Ax <- as.numeric(Ax)

  if (length(Ap) < 2L) {
    stop("`Ap` must contain at least two elements.")
  }
  if (length(Ai) != length(Ax)) {
    stop("`Ai` and `Ax` must have the same length.")
  }
  if (length(Ai) == 0L) {
    stop("`Ai` and `Ax` must contain at least one non-zero entry.")
  }
  if (anyNA(Ap) || anyNA(Ai) || anyNA(Ax)) {
    stop("`Ap`, `Ai`, and `Ax` must not contain missing values.")
  }
  if (!all(is.finite(Ax))) {
    stop("`Ax` must contain only finite values.")
  }
  if (Ap[1L] != 0L) {
    stop("`Ap` must start at 0.")
  }
  if (any(Ap < 0L)) {
    stop("`Ap` must contain non-negative column pointers.")
  }
  if (any(diff(Ap) < 0L)) {
    stop("`Ap` must be a non-decreasing sequence of column pointers.")
  }
  if (Ap[length(Ap)] != length(Ai)) {
    stop("The last entry of `Ap` must equal length(`Ai`) and length(`Ax`).")
  }
  if (any(Ai < 0L)) {
    stop("`Ai` must contain non-negative row indices.")
  }

  n <- length(Ap) - 1L
  if (!is.null(b)) {
    b <- as.numeric(b)
    if (length(b) == 0L) {
      stop("`b` must contain at least one element.")
    }
    if (anyNA(b) || !all(is.finite(b))) {
      stop("`b` must contain only finite values and no missing values.")
    }
    if (length(b) != n) {
      stop("`length(b)` must equal `length(Ap) - 1`.")
    }
    if (any(Ai >= n)) {
      stop("`Ai` must be strictly less than `length(b)` for `sparseLU_solve()`.")
    }
  }

  list(Ap = Ap, Ai = Ai, Ax = Ax, b = b)
}

#' Sparse LU Decomposition
#'
#' This function performs LU decomposition on a sparse matrix represented
#' in compressed column storage (CCS) format.
#'
#' @param Ap Integer vector of column pointers for the sparse matrix.
#' @param Ai Integer vector of row indices for the sparse matrix.
#' @param Ax Numeric vector of non-zero values for the sparse matrix.
#' @return A list containing the LU decomposition.
#' @export
#' @examples
#' # Example usage
#' # Create a simple 3x3 sparse matrix:
#' # 1 0 2
#' # 0 3 0
#' # 4 0 5
#'
#' Ap <- c(0, 2, 3, 5)  # Column pointers
#' Ai <- c(0, 2, 1, 0, 2)  # Row indices
#' Ax <- c(1.0, 4.0, 3.0, 2.0, 5.0)  # Non-zero values
#'
#' # Perform sparse LU decomposition
#' result <- sparseLU(Ap, Ai, Ax)
#' print(result)
#'
sparseLU <- function(Ap, Ai, Ax) {
  args <- .validate_ccs(Ap = Ap, Ai = Ai, Ax = Ax)
  .Call('_sparselu_sparseLU', args$Ap, args$Ai, args$Ax)
}

#' Solve a Sparse Linear System
#'
#' Computes the solution to a sparse linear system \eqn{Ax = b} using
#' the SuiteSparse UMFPACK LU factorisation.
#'
#' @param Ap Integer vector of column pointers for the sparse matrix.
#' @param Ai Integer vector of row indices for the sparse matrix.
#' @param Ax Numeric vector of non-zero values for the sparse matrix.
#' @param b Numeric right-hand side vector.
#' @return Numeric vector containing the solution \eqn{x}.
#' @export
#' @examples
#' Ap <- c(0L, 2L, 3L, 5L)
#' Ai <- c(0L, 2L, 1L, 0L, 2L)
#' Ax <- c(1, 4, 3, 2, 5)
#' b <- c(1, 2, 3)
#' sparseLU_solve(Ap, Ai, Ax, b)
sparseLU_solve <- function(Ap, Ai, Ax, b) {
  args <- .validate_ccs(Ap = Ap, Ai = Ai, Ax = Ax, b = b)
  .Call('_sparselu_sparseLU_solve', args$Ap, args$Ai, args$Ax, args$b)
}
