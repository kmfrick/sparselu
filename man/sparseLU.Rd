\name{sparseLU}
\alias{sparseLU}
\title{Sparse LU Decomposition}
\description{
Compute an LU factorisation of a sparse matrix stored in compressed column storage using the SuiteSparse UMFPACK routines.
}
\usage{
sparseLU(Ap, Ai, Ax)
}
\arguments{
\item{Ap}{Integer vector of column pointers indexing into \code{Ai} and \code{Ax}.}
\item{Ai}{Integer vector of row indices for each non-zero element.}
\item{Ax}{Numeric vector of the non-zero values.}
}
\value{
A named list with components \code{L}, \code{U}, \code{P}, and \code{Q} describing the LU factorisation returned by UMFPACK.
}
\details{
The column pointers \code{Ap} and row indices \code{Ai} must use zero-based indexing as required by the SuiteSparse UMFPACK interface.
}
\examples{
Ap <- c(0L, 2L, 3L, 5L)
Ai <- c(0L, 2L, 1L, 0L, 2L)
Ax <- c(1, 4, 3, 2, 5)

sparseLU(Ap, Ai, Ax)
}
