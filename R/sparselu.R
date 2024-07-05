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
  .Call('_sparselu_sparseLU', Ap, Ai, Ax)
}

