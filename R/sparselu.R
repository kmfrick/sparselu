#' Sparse LU Decomposition
#'
#' @param Ap Column pointers for the sparse matrix
#' @param Ai Row indices for the sparse matrix
#' @param Ax Non-zero values for the sparse matrix
#' @return A list containing the LU decomposition [L,U,P,Q]
#' @export
sparseLU <- function(Ap, Ai, Ax) {
  .Call('_sparselu_sparseLU', Ap, Ai, Ax)
}

