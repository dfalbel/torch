library(torch)

context("test-integer-tensor")

test_that("creation of 1d integer tensor", {
  expect_identical(as.array(tensor(1:10)), 1:10)
})

test_that("works even with gc", {
  x <- tensor(1:10)
  gc()
  expect_identical(as.array(x), 1:10)
})

test_that("creation of 2d integer tensor", {
  x <- matrix(1:100, ncol = 10)
  expect_identical(as.array(tensor(x)), x)
})

test_that("creation of 3d integer tensor", {
  x <- array(1:80, dim = c(20, 2, 2))
  expect_identical(as.array(tensor(x)), x)
})

context("tensor operations")

test_that("abs works", {
  x <- array((-80):(-1), dim = c(20, 2, 2))
  expect_identical(as.array(abs(tensor(x))), abs(x))

  x <- array(-runif(80), dim = c(20, 2, 2))
  expect_identical(as.array(abs(tensor(x))), abs(x))
})

context("test-numeric-tensor")

test_that("creation of 1d numeric tensor", {
  x <- runif(100)
  expect_identical(as.array(tensor(x)), x)
})

test_that("creation of 2d numeric tensor", {
  x <- matrix(runif(100), ncol = 10)
  expect_identical(as.array(tensor(x)), x)
})

test_that("creation of 3d numeric tensor", {
  x <- array(runif(80), dim = c(20, 2, 2))
  expect_identical(as.array(tensor(x)), x)
})

