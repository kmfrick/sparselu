\name{sparseLU_solve}
\alias{sparseLU_solve}
\title{Solve a Sparse Linear System}
\description{
Solve the sparse linear system \eqn{Ax = b} using the SuiteSparse UMFPACK LU factorisation.
}
\usage{
sparseLU_solve(Ap, Ai, Ax, b)
}
\arguments{
\item{Ap}{Integer vector of column pointers indexing into \code{Ai} and \code{Ax}.}
\item{Ai}{Integer vector of row indices for each non-zero element.}
\item{Ax}{Numeric vector of the non-zero values.}
\item{b}{Numeric vector containing the right-hand side of the linear system.}
}
\value{
Numeric vector with the solution to the system.
}
\details{
The sparse matrix is provided in compressed column storage using zero-based indexing in \code{Ap} and \code{Ai}, matching the expectations of the SuiteSparse UMFPACK interface.
}
\examples{
Ap <- c(0L, 2L, 3L, 5L)
Ai <- c(0L, 2L, 1L, 0L, 2L)
Ax <- c(1, 4, 3, 2, 5)
b <- c(1, 2, 3)

sparseLU_solve(Ap, Ai, Ax, b)
}
