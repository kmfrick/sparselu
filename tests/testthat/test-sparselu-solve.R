test_that("sparseLU_solve matches dense solve for a deterministic system", {
  A <- matrix(
    c(1, 0, 4,
      0, 3, 0,
      2, 0, 5),
    nrow = 3,
    byrow = FALSE
  )
  Ap <- c(0L, 2L, 3L, 5L)
  Ai <- c(0L, 2L, 1L, 0L, 2L)
  Ax <- c(1, 4, 3, 2, 5)
  b <- c(1, 2, 3)

  expect_equal(sparseLU_solve(Ap, Ai, Ax, b), as.numeric(solve(A, b)), tolerance = 1e-10)
})

test_that("sparseLU_solve matches dense solve on randomized diagonally dominant systems", {
  set.seed(20260217)

  for (n in c(5L, 10L, 20L)) {
    for (iter in 1:5) {
      A <- matrix(0, n, n)
      mask <- matrix(runif(n * n) < 0.15, n, n)
      A[mask] <- rnorm(sum(mask))
      diag(A) <- diag(A) + rowSums(abs(A)) + 1

      ccs <- dense_to_ccs(A)
      b <- rnorm(n)

      x_sparse <- sparseLU_solve(ccs$Ap, ccs$Ai, ccs$Ax, b)
      x_dense <- as.numeric(solve(A, b))

      expect_equal(x_sparse, x_dense, tolerance = 1e-7)
    }
  }
})

test_that("sparseLU_solve errors on singular systems", {
  Ap <- c(0L, 2L, 4L)
  Ai <- c(0L, 1L, 0L, 1L)
  Ax <- c(1, 2, 2, 4)
  b <- c(1, 1)

  expect_error(sparseLU_solve(Ap, Ai, Ax, b), "umfpack_di_(numeric|solve) failed with status")
})
