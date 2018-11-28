library(torch)

context("integer tensors")

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

context("byte tensors")

test_that("creation of 1d byte tensor", {
  x <- tensor(c(TRUE, FALSE))
  expect_identical(class(x)[1], "tensor")
  expect_identical(as.array(x), c(TRUE, FALSE))

  x <- sample(c(TRUE, FALSE), 50, replace = TRUE)
  expect_identical(as.array(tensor(x)), x)
})

test_that("creation of 2d byte tensor", {
  x <- matrix(c(TRUE, FALSE), ncol = 10, nrow = 5)
  expect_identical(as.array(tensor(x)), x)
})

test_that("creation of 3d byte tensor", {
  x <- array(c(TRUE, FALSE), dim = c(20, 2, 2))
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

test_that("all works", {
  l <- array(TRUE, dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(all(x)), all(l))

  l <- array(FALSE, dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(all(x)), all(l))

  l <- array(c(TRUE, FALSE), dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(all(x)), all(l))

  l <- array(c(TRUE, FALSE, TRUE, TRUE), dim = c(2, 2))
  x <- tensor(l)

  expect_identical(
    as.array(all(x, dim = 1, keepdim = TRUE)),
    matrix(c(FALSE, TRUE), nrow = 1)
  )

  expect_identical(
    as.array(all(x, dim = 2, keepdim = TRUE)),
    matrix(c(TRUE, FALSE), ncol = 1)
  )

  expect_identical(
    as.array(all(x, dim = 1, keepdim = FALSE)),
    c(FALSE, TRUE)
  )

  expect_identical(
    as.array(all(x, dim = 2, keepdim = FALSE)),
    c(TRUE, FALSE)
  )

  expect_error(
    all(x, dim = 0)
  )

})

test_that("allclose works", {

  x <- tensor(c(1,2,3,4,5))
  y <- tensor(c(1,2,3,4,5) + 1e-6)
  a <- allclose(x, y)

  expect_identical(a, TRUE)

  x <- tensor(c(1,2,3,4,5))
  y <- tensor(c(1,2,3,4,5) + 1e-4)
  a <- allclose(x, y)

  expect_identical(a, FALSE)

  x <- tensor(c(1,2,3,4,5))
  y <- tensor(c(1,2,3,4,5) + 1e-5)
  a <- allclose(x, y)

  expect_identical(a, TRUE)
})

test_that("any works", {

  l <- array(TRUE, dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(any(x)), any(l))

  l <- array(FALSE, dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(any(x)), any(l))

  l <- array(c(TRUE, FALSE), dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(any(x)), any(l))

  l <- array(c(TRUE, FALSE, TRUE, TRUE), dim = c(2, 2))
  x <- tensor(l)

  expect_identical(
    as.array(any(x, dim = 1, keepdim = TRUE)),
    matrix(c(TRUE, TRUE), nrow = 1)
  )

  expect_identical(
    as.array(any(x, dim = 2, keepdim = TRUE)),
    matrix(c(TRUE, TRUE), ncol = 1)
  )

  expect_identical(
    as.array(any(x, dim = 1, keepdim = FALSE)),
    c(TRUE, TRUE)
  )

  expect_identical(
    as.array(any(x, dim = 2, keepdim = FALSE)),
    c(TRUE, TRUE)
  )

  expect_error(
    any(x, dim = 0)
  )


})

context("numeric tensors")

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

