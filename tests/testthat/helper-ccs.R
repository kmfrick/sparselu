dense_to_ccs <- function(A) {
  nz <- which(A != 0, arr.ind = TRUE)
  sparse <- Matrix::sparseMatrix(
    i = nz[, 1],
    j = nz[, 2],
    x = A[nz],
    dims = dim(A),
    giveCsparse = TRUE
  )
  list(
    Ap = as.integer(sparse@p),
    Ai = as.integer(sparse@i),
    Ax = as.numeric(sparse@x)
  )
}
