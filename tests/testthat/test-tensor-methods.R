library(torch)


test_that("abs works", {
  x <- tensor(-1)
  x$abs_()
  expect_equal(as.array(x), 1)
})

test_that("acos works", {
  x <- tensor(1)
  x$acos_()
  expect_equal(as.array(x), acos(1), tol = 1e-7)
})

