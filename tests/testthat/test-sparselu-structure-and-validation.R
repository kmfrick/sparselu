test_that("sparseLU returns consistent factor structure", {
  A <- matrix(
    c(10, 0, 0, 2,
      3, 9, 0, 0,
      0, 7, 8, 0,
      0, 0, 1, 6),
    nrow = 4,
    byrow = TRUE
  )

  ccs <- dense_to_ccs(A)
  out <- sparseLU(ccs$Ap, ccs$Ai, ccs$Ax)

  expect_named(out, c("L", "U", "P", "Q"))
  expect_named(out$L, c("j", "p", "x"))
  expect_named(out$U, c("i", "p", "x"))

  expect_equal(length(out$L$p), nrow(A) + 1L)
  expect_equal(length(out$U$p), ncol(A) + 1L)
  expect_true(all(diff(out$L$p) >= 0L))
  expect_true(all(diff(out$U$p) >= 0L))
  expect_equal(out$L$p[length(out$L$p)], length(out$L$j))
  expect_equal(out$U$p[length(out$U$p)], length(out$U$i))

  expect_setequal(out$P, 0:(nrow(A) - 1L))
  expect_setequal(out$Q, 0:(ncol(A) - 1L))
})

test_that("R-layer validation rejects malformed CCS input", {
  expect_error(sparseLU(c(0L), 0L, 1), "at least two elements")
  expect_error(sparseLU(c(0L, 1L), c(0L, 1L), 1), "same length")
  expect_error(sparseLU(c(0L, 0L), integer(), numeric()), "at least one non-zero")
  expect_error(sparseLU(c(1L, 1L), 0L, 1), "must start at 0")
  expect_error(sparseLU(c(0L, 2L), 0L, 1), "last entry")
  expect_error(sparseLU(c(0L, 1L), -1L, 1), "non-negative row indices")
  expect_error(sparseLU(c(0L, 1L), 0L, Inf), "finite values")

  expect_error(sparseLU_solve(c(0L, 1L), 0L, 1, numeric()), "at least one element")
  expect_error(sparseLU_solve(c(0L, 1L, 1L), c(0L), 1, 1), "length\\(b\\)")
  expect_error(sparseLU_solve(c(0L, 1L), 1L, 1, 1), "strictly less than `length\\(b\\)`")
  expect_error(sparseLU_solve(c(0L, 1L), 0L, Inf, 1), "finite values")
  expect_error(sparseLU_solve(c(0L, 1L), 0L, 1, NA_real_), "no missing values")
})
