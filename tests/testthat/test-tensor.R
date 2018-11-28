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

test_that("acos works", {
  x <- array(-runif(80), dim = c(20, 2, 2))
  expect_equal(as.array(acos(tensor(x))), acos(x))
})

test_that("add works", {
  x <- array((-80):(-1), dim = c(20, 2, 2))
  y <- array((-80):(-1), dim = c(20, 2, 2))
  expect_identical(as.array(tensor(x) + tensor(y)), x + y)

  x <- array(-runif(80), dim = c(20, 2, 2))
  y <- array(-runif(80), dim = c(20, 2, 2))
  expect_identical(as.array(tensor(x) + tensor(y)), x + y)
})

test_that("add does not modify in palce", {
  x <- array(1:80, dim = c(20, 2, 2))
  x_ <- as.array(tensor(x) + tensor(x))

  expect_identical(x, array(1:80, dim = c(20, 2, 2)))
  expect_identical(x_, x + x)
})

test_that("addbmm works", {

  x <- tensor(matrix(runif(15), nrow = 3, ncol = 5))
  b1 <- tensor(array(runif(120), dim = c(10, 3, 4)))
  b2 <- tensor(array(runif(200), dim = c(10, 4, 5)))

  res <- as.array(addbmm(x, b1, b2, 1, 1))

  expect_true(is.array(res))
  expect_identical(dim(res), c(3L, 5L))
})

test_that("addcdiv works", {

  x <- tensor(matrix(runif(3), nrow = 1, ncol = 3))
  t1 <- tensor(array(runif(3), dim = c(3, 1)))
  t2 <- tensor(array(runif(3), dim = c(1, 3)))

  res <- as.array(addcdiv(x, t1, t2, 0.1))

  expect_true(is.array(res))
  expect_identical(dim(res), c(3L, 3L))
})

test_that("addcmul works", {

  x <- tensor(matrix(runif(3), nrow = 1, ncol = 3))
  t1 <- tensor(array(runif(3), dim = c(3, 1)))
  t2 <- tensor(array(runif(3), dim = c(1, 3)))

  res <- as.array(addcmul(x, t1, t2, 0.1))

  expect_true(is.array(res))
  expect_identical(dim(res), c(3L, 3L))
})

test_that("addmm works", {

  x <- tensor(matrix(runif(6), nrow = 2, ncol = 3))
  mat1 <- tensor(array(runif(6), dim = c(2, 3)))
  mat2 <- tensor(array(runif(6), dim = c(3, 3)))

  res <- as.array(addmm(x, mat1, mat2, 1))

  expect_true(is.array(res))
  expect_identical(dim(res), c(2L, 3L))
})

test_that("addmv works", {

  x <- tensor(runif(2))
  mat <- tensor(array(runif(6), dim = c(2, 3)))
  vec <- tensor(runif(3))

  res <- as.array(addmv(x, mat, vec))

  expect_identical(length(res), 2L)
})

test_that("addr works", {
  vec1 <- tensor(c(1,2,3))
  vec2 <- tensor(c(1,2))
  x <- tensor(matrix(0, nrow = 3, ncol = 2))

  res <- as.array(addr(x, vec1, vec2))
  res_ <- matrix(c(1,2,3,2,4,6), ncol = 2)
  expect_identical(res, res_)
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

