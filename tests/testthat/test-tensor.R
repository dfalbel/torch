library(torch)

context("tensor options")

test_that("requires_grad", {
  x <- tensor(x = runif(10), requires_grad = TRUE)
  expect_identical(class(x)[1], "tensor")
})

test_that("dtype", {
  x <- tensor(1:10, dtype = "double")
  expect_identical(x$dtype(), "double")

  x <- tensor(1:10, dtype = "float32")
  expect_identical(x$dtype(), "float")

  x <- tensor(1:10)
  expect_identical(x$dtype(), "int")
})

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
  expect_identical(as.array(tch_abs(tensor(x))), abs(x))

  x <- array(-runif(80), dim = c(20, 2, 2))
  expect_equal(as.array(tch_abs(tensor(x))), abs(x), tol = 1e-7)

  x <- tensor(-1)
  x$abs_()
  expect_equal(as.array(x), 1)
})

test_that("acos works", {
  x <- array(-runif(80), dim = c(20, 2, 2))
  expect_equal(as.array(tch_acos(tensor(x))), acos(x), tol = 1e-7)

  x <- tensor(1)
  x$acos_()
  expect_equal(as.array(x), acos(1), tol = 1e-7)
})

test_that("add works", {
  x <- array((-80):(-1), dim = c(20, 2, 2))
  y <- array((-80):(-1), dim = c(20, 2, 2))
  expect_identical(as.array(tensor(x) + tensor(y)), x + y)

  x <- array(-runif(80), dim = c(20, 2, 2))
  y <- array(-runif(80), dim = c(20, 2, 2))
  expect_equal(as.array(tensor(x) + tensor(y)), x + y, tol = 1e-7)

  x <- runif(100)
  expect_equal(as.array(tensor(x) + 1), x + 1, tol = 1e-7)

  x <- tensor(1)
  x$add_(2)
  expect_equal(as.array(x), 3)

  x <- tensor(1)
  x$add_(tensor(2))
  expect_equal(as.array(x), 3)
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

  res <- as.array(tch_addbmm(x, b1, b2, 1, 1))

  expect_true(is.array(res))
  expect_identical(dim(res), c(3L, 5L))

  expect_silent(x$addbmm_(b1, b2, 1, 1))
  expect_identical(dim(as.array(x)), c(3L, 5L))
})

test_that("addcdiv works", {

  x <- tensor(matrix(runif(3), nrow = 1, ncol = 3))
  t1 <- tensor(array(runif(3), dim = c(3, 1)))
  t2 <- tensor(array(runif(3), dim = c(1, 3)))

  res <- as.array(tch_addcdiv(x, t1, t2, 0.1))

  expect_true(is.array(res))
  expect_identical(dim(res), c(3L, 3L))

  x <- matrix(runif(3), nrow = 1, ncol = 3)
  x_t <- tensor(x)
  expect_silent(x_t$addcdiv_(tensor(1), tensor(2), 1))
  expect_equal(as.array(x_t), x + 0.5, tol = 1e-6)
})

test_that("addcmul works", {

  x <- tensor(matrix(runif(3), nrow = 1, ncol = 3))
  t1 <- tensor(array(runif(3), dim = c(3, 1)))
  t2 <- tensor(array(runif(3), dim = c(1, 3)))

  res <- as.array(tch_addcmul(x, t1, t2, 0.1))

  expect_true(is.array(res))
  expect_identical(dim(res), c(3L, 3L))

  x <- matrix(runif(3), nrow = 1, ncol = 3)
  x_t <- tensor(x)
  expect_silent(x_t$addcmul_(tensor(2), tensor(2), 1))
  expect_equal(as.array(x_t), x + 4, tol = 1e-6)
})

test_that("addmm works", {

  x <- matrix(0, nrow = 2, ncol = 2)
  mat1 <- matrix(2, nrow = 2, ncol = 3)
  mat2 <- matrix(3, nrow = 3, ncol = 2)

  res <- as.array(tch_addmm(tensor(x), tensor(mat1), tensor(mat2), 1))
  expect_equal(res, x + mat1 %*% mat2)

  x <- matrix(0, nrow = 2, ncol = 2)
  mat1 <- matrix(2, nrow = 2, ncol = 3)
  mat2 <- matrix(3, nrow = 3, ncol = 2)

  x_t <- tensor(x)
  expect_silent(x_t$addmm_(tensor(mat1), tensor(mat2)))
  expect_equal(as.array(x_t), x + mat1 %*% mat2)
})

test_that("addmv works", {
  x <- c(1,0)
  mat <- matrix(1, nrow = 2, ncol = 4)
  vec <- c(1,2,3,4)

  res <- as.array(tch_addmv(tensor(x), tensor(mat), tensor(vec)))
  expect_equal(res, as.numeric(x + mat %*% vec))

  x_t <- tensor(x)
  x_t$addmv_(tensor(mat), tensor(vec))
  expect_equal(as.array(x_t), as.numeric(x + mat %*% vec))
})

test_that("addr works", {
  vec1 <- c(1,2,3)
  vec2 <- c(1,2)
  x <- matrix(runif(6), nrow = 3, ncol = 2)

  expect_equal(as.array(tch_addr(tensor(x), tensor(vec1), tensor(vec2))), x + vec1 %o% vec2, tol = 1e-6)

  x_t <- tensor(x)
  x_t$addr_(tensor(vec1), tensor(vec2))
  expect_equal(as.array(x_t), x + vec1 %o% vec2, tol = 1e-6)
})

test_that("all works", {
  l <- array(TRUE, dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(tch_all(x)), all(l))

  l <- array(FALSE, dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(tch_all(x)), all(l))

  l <- array(c(TRUE, FALSE), dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(tch_all(x)), all(l))

  l <- array(c(TRUE, FALSE, TRUE, TRUE), dim = c(2, 2))
  x <- tensor(l)

  expect_identical(
    as.array(tch_all(x, dim = 0, keepdim = TRUE)),
    matrix(c(FALSE, TRUE), nrow = 1)
  )

  expect_identical(
    as.array(tch_all(x, dim = 1, keepdim = TRUE)),
    matrix(c(TRUE, FALSE), ncol = 1)
  )

  expect_identical(
    as.array(tch_all(x, dim = 0, keepdim = FALSE)),
    c(FALSE, TRUE)
  )

  expect_identical(
    as.array(tch_all(x, dim = 1, keepdim = FALSE)),
    c(TRUE, FALSE)
  )

})

test_that("allclose works", {

  x <- tensor(c(1,2,3,4,5))
  y <- tensor(c(1,2,3,4,5) + 1e-6)
  a <- tch_allclose(x, y)

  expect_identical(a, TRUE)

  x <- tensor(c(1,2,3,4,5))
  y <- tensor(c(1,2,3,4,5) + 1e-4)
  a <- tch_allclose(x, y)

  expect_identical(a, FALSE)

  x <- tensor(c(1,2,3,4,5))
  y <- tensor(c(1,2,3,4,5) + 1e-6)
  a <- tch_allclose(x, y)

  expect_identical(a, TRUE)
})

test_that("any works", {

  l <- array(TRUE, dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(tch_any(x)), any(l))

  l <- array(FALSE, dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(tch_any(x)), any(l))

  l <- array(c(TRUE, FALSE), dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(tch_any(x)), any(l))

  l <- array(c(TRUE, FALSE, TRUE, TRUE), dim = c(2, 2))
  x <- tensor(l)

  expect_identical(
    as.array(tch_any(x, dim = 0, keepdim = TRUE)),
    matrix(c(TRUE, TRUE), nrow = 1)
  )

  expect_identical(
    as.array(tch_any(x, dim = 1, keepdim = TRUE)),
    matrix(c(TRUE, TRUE), ncol = 1)
  )

  expect_identical(
    as.array(tch_any(x, dim = 0, keepdim = FALSE)),
    c(TRUE, TRUE)
  )

  expect_identical(
    as.array(tch_any(x, dim = 1, keepdim = FALSE)),
    c(TRUE, TRUE)
  )

})

test_that("argmax works", {

  l <- array(1:6000, dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(tch_argmax(x)), which.max(l) - 1L)
  expect_identical(as.array(tch_argmax(x, 0)), apply(l, c(2,3), which.max) - 1L)
  expect_identical(as.array(tch_argmax(x, -1)), apply(l, c(1,2), which.max) - 1L)
})

test_that("argmin works", {

  l <- array(1:6000, dim = c(10, 20, 30))
  x <- tensor(l)
  expect_identical(as.array(tch_argmin(x)), which.min(l) - 1L)
  expect_identical(as.array(tch_argmin(x, 0)), apply(l, c(2,3), which.min) - 1L)
  expect_identical(as.array(tch_argmin(x, -1)), apply(l, c(1,2), which.min) - 1L)
})

test_that("asin works", {
  x <- runif(100)
  expect_equal(as.array(tch_asin(tensor(x))), asin(x), tol = 1e-7)

  x_t <- tensor(x)
  x_t$asin_()
  expect_equal(as.array(x_t), asin(x), tol = 1e-7)
})

test_that("atan works", {
  x <- runif(100)
  expect_equal(as.array(tch_atan(tensor(x))), atan(x), tol = 1e-7)

  x_t <- tensor(x)
  x_t$atan_()
  expect_equal(as.array(x_t), atan(x), tol = 1e-7)
})

test_that("atan2 works", {
  x <- runif(100)
  y <- runif(100)

  expect_equal(as.array(tch_atan2(tensor(x), tensor(y))), atan2(x, y), tol = 1e-7)

  x_t <- tensor(x)
  x_t$atan2_(tensor(y))
  expect_equal(as.array(x_t), atan2(x, y), tol = 1e-7)
})

test_that("as_strided works", {
  # TODO better testint as_strided - undocument in the python side too.
  l <- array(1:6000, dim = c(10, 20, 30))
  x <- tensor(l)

  k <- tch_as_strided(x, 0, 2)

  expect_identical(class(k)[1], "tensor")

  k <- tch_as_strided(x, 0, 2, 1)

  expect_identical(class(k)[1], "tensor")
})

test_that("backward works", {
  # TODO include tests for backward operation. for example python only accepts scalars, etc.
  x <- tensor(runif(10), requires_grad = TRUE)

  expect_silent(x$backward())
})

test_that("baddbmm works", {

  x <- tensor(array(runif(45), dim = c(3, 3, 5)))
  batch1 <- tensor(array(runif(36), dim = c(3, 3, 4)))
  batch2 <- tensor(array(runif(60), dim = c(3, 4, 5)))

  expect_silent(y <- tch_baddbmm(x, batch1, batch2))
  expect_equal(class(y)[1], "tensor")

  expect_silent(x$baddbmm_(batch1, batch2))
})

test_that("bernoulli works", {
  x <- tensor(runif(10))
  expect_silent(tch_bernoulli(x))

  x <- tensor(rep(0, 100))
  expect_equal(sum(as.array(tch_bernoulli(x))), 0)

  x <- tensor(rep(1, 100))
  expect_equal(sum(as.array(tch_bernoulli(x))), 100)

  x <- tch_empty(c(2,2))
  x$bernoulli_(p = 1)
  expect_equal(sum(as.array(x)), 4)

  x <- tch_empty(c(2,2))
  x$bernoulli_(p = tensor(matrix(c(0, 1), nrow = 2, ncol = 2)))
  expect_equal(sum(as.array(x)), 2)
})

test_that("bincount works", {
  x <- sample(0:9, 500, replace = TRUE)
  expect_equal(as.array(tch_bincount(tensor(x))), as.integer(table(x)))

  x <- sample(0:9, 500, replace = TRUE)
  weights <- runif(500)

  expect_equal(as.array(tch_bincount(tensor(x), tensor(weights))), as.numeric(tapply(weights, x, sum)), tol = 1e-6)
})

test_that("bmm works", {
  x <- tensor(array(runif(120), dim = c(10, 3, 4)))
  y <- tensor(array(runif(200), dim = c(10, 4, 5)))
  res <- as.array(tch_bmm(x, y))

  expect_equal(dim(res), c(10, 3, 5))
})

test_that("btrifact works and btrifact_with_info", {
  x <- tensor(array(runif(18), dim = c(2, 3, 3)))
  res <- tch_btrifact(x)

  a_lu <- as.array(res[[1]])
  pivot <- as.array(res[[2]])

  expect_equal(dim(a_lu), c(2, 3, 3))
  expect_equal(dim(pivot), c(2, 3))

  res <- x$btrifact_with_info()

  a_lu <- as.array(res[[1]])
  pivot <- as.array(res[[2]])

  expect_equal(dim(a_lu), c(2, 3, 3))
  expect_equal(dim(pivot), c(2, 3))
})

test_that("byte works", {
  x <- tensor(c(0,1,1,0,1))
  x <- x$byte()
  expect_equal(x$dtype(), "uint8")
  expect_equal(as.array(x), c(FALSE, TRUE, TRUE, FALSE, TRUE))
})

test_that("btrisolve works", {
  A <- tensor(array(runif(18), dim = c(2,3,3)))
  b <- tensor(matrix(runif(6), ncol = 3))
  A_LU <- tch_btrifact(A)
  x <- as.array(tch_btrisolve(b, A_LU[[1]], A_LU[[2]]))

  expect_equal(dim(x), c(2, 3))
})

test_that("cauchy works", {
  a <- matrix(runif(10), ncol = 2)
  x <- tensor(a)
  x$cauchy_(0, 1)
  expect_false(all(as.array(x) == a))

  b <- runif(10)
  x <- tensor(b)
  x$cauchy_(0, 1)

  expect_false(all(as.array(x) == b))
})

test_that("ceil works", {
  x <- tensor(runif(10))
  expect_equal(as.array(tch_ceil(x)), rep(1, 10))
})

test_that("ceil_ works", {
  x <- tensor(runif(10))
  x$ceil_()
  expect_equal(as.array(x), rep(1, 10))
})

test_that("char works", {
  x <- tensor(c(1,2,3))
  y <- x$char()
  expect_equal(y$dtype(), "signed char")
})

test_that("cholesky works", {
  m <- matrix(c(5,1,1,3),2,2)
  m_t <- tensor(m)
  res1 <- as.numeric(chol(m))[-2]
  res2 <- as.numeric(as.array(m_t$cholesky()))[-3]
  expect_equal(res1, res2, tol = 1e-6)
})

test_that("chunk works", {
  a <- array(runif(100), dim = c(4, 5, 5))
  x <- tensor(a)
  chunks <- tch_chunk(x, 2, 0)

  expect_equal(as.array(chunks[[1]]), a[1:2,,], tol = 1e-7)
  expect_equal(as.array(chunks[[2]]), a[3:4,,], tol = 1e-7)
})

test_that("clamp works", {
  x <- tensor(1:10)
  res <- as.array(tch_clamp(x, 5, 7))

  expect_equal(min(res), 5L)
  expect_equal(max(res), 7L)

  x <- tensor(runif(100))
  res <- as.array(tch_clamp(x, 0.5, 0.7))

  expect_equal(min(res), 0.5)
  expect_equal(max(res), 0.7)
})

test_that("clamp_ works", {
  x <- tensor(1:10)
  x$clamp_(5, 7)
  res <- as.array(x)

  expect_equal(min(res), 5L)
  expect_equal(max(res), 7L)

  x <- tensor(runif(100))
  x$clamp_(0.5, 0.7)
  res <- as.array(x)

  expect_equal(min(res), 0.5)
  expect_equal(max(res), 0.7)
})

test_that("clamp_max works", {
  x <- tensor(1:10)
  res <- as.array(tch_clamp_max(x, 7))

  expect_equal(max(res), 7L)

  x <- tensor(runif(100))
  res <- as.array(tch_clamp_max(x, 0.7))

  expect_equal(max(res), 0.7)
})

test_that("clamp_max_ works", {
  x <- tensor(1:10)
  x$clamp_max_(7)
  res <- as.array(x)

  expect_equal(max(res), 7L)

  x <- tensor(runif(100))
  x$clamp_max_(0.7)
  res <- as.array(x)

  expect_equal(max(res), 0.7)
})

test_that("clamp_min works", {
  x <- tensor(1:10)
  res <- as.array(tch_clamp_min(x, 7))

  expect_equal(min(res), 7L)

  x <- tensor(runif(100))
  res <- as.array(tch_clamp_min(x, 0.7))

  expect_equal(min(res), 0.7)
})

test_that("clamp_min_ works", {
  x <- tensor(1:10)
  x$clamp_min_(7)
  res <- as.array(x)

  expect_equal(min(res), 7L)

  x <- tensor(runif(100))
  x$clamp_min_(0.7)
  res <- as.array(x)

  expect_equal(min(res), 0.7)
})

test_that("clone works", {
  x <- tensor(1:10)
  y <- x$clone()
  x$clamp_max_(5)

  expect_false(all(as.array(x) == as.array(y)))
})

test_that("contiguous works", {
  # TODO better test if contiguous. it's important to test copies.
  expect_silent(tensor(1:10)$contiguous())
})

test_that("copy_ works", {
  x <- tensor(1:10)
  y <- tensor(11:20)
  x$copy_(y)
  expect_equal(as.array(x), as.array(y))

  x <- tensor(1:10)
  y <- tensor(11:20)
  x$copy_(y, non_blocking = TRUE)
  expect_equal(as.array(x), as.array(y))
})

test_that("cos works", {
  x <- tensor(c(pi, 2*pi))
  expect_equal(as.array(tch_cos(x)), cos(c(pi, 2*pi)))
})

test_that("cos_ works", {
  x <- tensor(c(pi, 2*pi))
  x$cos_()
  expect_equal(as.array(x), cos(c(pi, 2*pi)))
})

test_that("cosh works", {
  x <- tensor(c(pi, 2*pi))
  expect_equal(as.array(tch_cosh(x)), cosh(c(pi, 2*pi)), tol = 1e-5)
})

test_that("cosh_ works", {
  x <- tensor(c(pi, 2*pi))
  x$cosh_()
  expect_equal(as.array(x), cosh(c(pi, 2*pi)), tol = 1e-5)
})

test_that("cpu works", {
  # TODO better test if tensor is not on cpu. this requires allocating.
  # otherwise the function is doing nothing
  expect_silent(tensor(1:10)$cpu())
})

test_that("cpu works", {
  # TODO better test if tensor is not on cpu. this requires allocating.
  # otherwise the function is doing nothing
  expect_silent(tensor(1:10)$cpu())
})

test_that("cross works", {
  # TODO better test this. probly with R examples.
  a <- matrix(runif(12), ncol = 3)
  b <- matrix(runif(12), ncol = 3)

  expect_silent(
    as.array(tch_cross(tensor(a), tensor(b)))
  )
})

test_that("cuda works", {
  # TODO better test if tensor is not on cpu. this requires allocating.
  # otherwise the function is doing nothing
  # we expect an error since we are not testing on GPU's yet.
  expect_error(tensor(1:10)$cuda())
})

test_that("cumprod works", {
  expect_equal(
    as.array(tch_cumprod(tensor(1:10), dim = 0)),
    cumprod(1:10)
  )
})

test_that("cumsum works", {
  expect_equal(
    as.array(tch_cumsum(tensor(1:10), dim = 0)),
    cumsum(1:10)
  )
})

test_that("data_ptr works", {
  x <- tensor(c(1,2,3,4))
  y <- x$clone()

  expect_true(x$data_ptr() != y$data_ptr())
})

test_that("det works", {
  x <- matrix(runif(36), ncol = 6)

  expect_equal(
    as.array(tch_det(tensor(x))),
    det(x),
    tol = 1e-7
  )
})

test_that("detach works", {
  # TODO better testing this.
  x <- tensor(matrix(runif(36), ncol = 6))
  expect_silent(y <- x$detach())
})

test_that("detach_ works", {
  # TODO better testing this
  x <- tensor(matrix(runif(36), ncol = 6))
  expect_silent(x$detach_())
})

test_that("device works", {
  # TODO tests on the GPU
  x <- tensor(1:10)
  expect_equal(x$device(), "CPU")
})

test_that("diag works", {
  x <- tensor(1:10)
  expect_equal(
    as.array(tch_diag(x)),
    diag(1:10)
  )

  x <- tensor(matrix(1:25, 5, 5))
  expect_equal(
    as.array(tch_diag(x)),
    as.integer(c(1, 7, 13, 19, 25))
  )
})

test_that("diagembed works", {
  x <- tch_randn(c(4,4))
  expect_silent(x$diag_embed(dim1 = 0))
  expect_silent(x$diag_embed())
})

test_that("diagflat works", {

  x <- tensor(1:10)
  expect_equal(
    as.array(tch_diagflat(x)),
    diag(1:10)
  )

  x <- tensor(matrix(1:25, 5, 5))
  expect_equal(as.array(tch_diagflat(x)), diag(as.integer(t(matrix(1:25, 5)))))
})


test_that("diagonal works", {

  x <- tensor(1:10)
  expect_error(tch_diagonal(x))

  x <- tensor(matrix(1:25, 5, 5))
  expect_equal(
    as.array(tch_diagonal(x)),
    as.integer(c(1, 7, 13, 19, 25))
  )
})

test_that("digamma works", {
  x <- tensor(c(1, 0.5))
  expect_equal(as.array(tch_digamma(x)), digamma(c(1, 0.5)), tol = 1e-6)
})

test_that("digamma_ works", {
  x <- tensor(c(1, 0.5))
  x$digamma_()
  expect_equal(as.array(x), digamma(c(1, 0.5)), tol = 1e-6)
})

test_that("dim works", {
  x <- tensor(c(1, 0.5))
  expect_equal(x$dim(), 1L)

  x <- tensor(matrix(1:10, 2, 5))
  expect_equal(x$dim(), 2L)

  x <- tensor(array(0, dim = 1:5))
  expect_equal(x$dim(), 5L)
})

test_that("dist works", {
  x <- tensor(as.numeric(1:10))
  y <- tensor(as.numeric(10:1))
  expect_equal(
    as.array(tch_dist(x, y)),
    sqrt(sum((1:10 - 10:1)^2)),
    tol = 1e-7
  )
})

test_that("div works", {
  x <- tensor(as.numeric(1:10))
  y <- tensor(as.numeric(10:1))
  expect_equal(as.array(x / y), 1:10/10:1, tol = 1e-7)
  expect_equal(as.array(x / 2), 1:10/2, tol = 1e-7)

  x <- tensor(1:10)
  y <- tensor(10:1)
  expect_equal(as.array(x / y), as.integer(1:10/10:1))
  expect_equal(as.array(x / 2L), as.integer(1:10/2L))
})

test_that("div_ works", {
  x <- tensor(as.numeric(1:10))
  y <- tensor(as.numeric(10:1))
  x$div_(y)
  expect_equal(as.array(x), 1:10/10:1, tol = 1e-7)

  x <- tensor(as.numeric(1:10))
  x$div_(2)
  expect_equal(as.array(x), 1:10/2, tol = 1e-7)

  # works with integers too
  x <- tensor(1:10)
  y <- tensor(10:1)
  x$div_(y)
  expect_equal(as.array(x), as.integer(1:10/10:1))

  x <- tensor(1:10)
  x$div_(2L)
  expect_equal(as.array(x), as.integer(1:10/2))
})

test_that("dot works", {
  expect_equal(
    as.array(tch_dot(tensor(c(2,3)), tensor(c(2,1)))),
    7
  )
})

test_that("double", {
  x <- tensor(1:10)
  x <- x$double()
  expect_equal(x$dtype(), "double")
})

test_that("eig works", {
  x_r <- cbind(c(1,-1), c(-1,1))

  x <- tensor(x_r)
  out <- tch_eig(x, eigenvectors = TRUE)
  out_r <- eigen(x_r)

  expect_equal(as.array(out[[1]])[,1], out_r$values)
  expect_equal(as.array(out[[2]]), -out_r$vectors)
})

test_that("element size works", {
  x <- tensor(1)
  expect_equal(x$element_size(), 4)

  x <- x$to("double")
  expect_equal(x$element_size(), 8)
})

test_that("eq works", {
  x <- tensor(1:10)
  y <- tensor(c(1:5, 10:6))

  expect_equal(as.array(x == y), 1:10 == c(1:5, 10:6))
  expect_equal(as.array(x == 1), 1:10 == 1)
})

test_that("equal works", {
  x <- tensor(c(1,2))
  y <- tensor(c(1,2))
  expect_equal(tch_equal(x, y), TRUE)
  y <- tensor(c(1,1))
  expect_equal(tch_equal(x, y), FALSE)
  y <- tensor(c(1,2), dtype = "double")
  expect_error(tch_equal(x, y))
})

test_that("erf works", {
  x <- tensor(1)
  r <- 2 * pnorm(1 * sqrt(2)) - 1
  expect_equal(as.array(x$erf()), r)
  x$erf_()
  expect_equal(as.array(x), r)
})

test_that("erfc works", {
  x <- tensor(1)
  r <- 2 * pnorm(1 * sqrt(2)) - 1
  expect_equal(as.array(x$erfc()), 1 - r)
  x$erfc_()
  expect_equal(as.array(x),  1- r)
})

test_that("erfinv works", {
  x <- tensor(1)
  r <- qnorm((1 + 1)/2)/sqrt(2)
  expect_equal(as.array(x$erfinv()), r)
  x$erfinv_()
  expect_equal(as.array(x),  r)
})

test_that("exp works", {
  x <- tensor(1)
  r <- exp(1)
  expect_equal(as.array(x$exp()), r, tol = 1e-6)
  x$exp_()
  expect_equal(as.array(x),  r, tol = 1e-6)
})

test_that("expand works", {
  x <- tch_randn(c(2,2))
  y <- x$expand(c(1,2,2))
  expect_equal(dim(as.array(y)), c(1L,2L,2L))
  y <- x$expand(c(1,-1,-1))
  expect_equal(dim(as.array(y)), c(1L,2L,2L))
})

test_that("expand_as works", {
  x <- tch_randn(c(2,2))
  y <- tch_randn(c(1,2,2))
  z <- x$expand_as(y)
  expect_equal(dim(as.array(z)), c(1L,2L,2L))

  y <- tch_randn(c(1))
  expect_error(x$expand_as(y))
})

test_that("expm1 works", {
  x <- tensor(1)
  r <- expm1(1)
  expect_equal(as.array(x$expm1()), r, tol = 1e-6)
  x$expm1_()
  expect_equal(as.array(x),  r, tol = 1e-6)
})

test_that("exponential works", {
  x <- tch_empty(c(4,4))
  expect_silent(x$exponential_(1))
  expect_true(all(as.array(x) > 0))
})

test_that("fill works", {
  x <- tch_empty(c(2,2))
  x$fill_(2)
  expect_equal(as.array(x), matrix(2, nrow = 2, ncol = 2))
  y <- tensor(0)$sum()
  x$fill_(y)
  expect_equal(as.array(x), matrix(0, nrow = 2, ncol = 2))
})

test_that("floor works", {
  x <- tensor(pi)
  expect_equal(as.array(x$floor()), 3)
  x$floor_()
  expect_equal(as.array(x), 3)
})

test_that("fmod works", {
  x <- tensor(3)
  expect_equal(as.array(x$fmod(2)), 1)
  x$fmod_(2)
  expect_equal(as.array(x), 1)
})

test_that("lerp works", {
  start <- tch_arange(1, 5)
  end <- tch_empty(4)$fill_(10)

  expect_equal(as.array(tch_lerp(start, end, 0.5)), c(5.5,  6,  6.5,  7))

  start$lerp_(end, 0.5)
  expect_equal(as.array(start), c(5.5,  6,  6.5,  7))

})

test_that("flatten works", {
  x <- tch_randn(c(2,2,2))
  expect_equal(length(as.array(x$flatten())), 8)
  expect_equal(dim(as.array(x$flatten(start_dim = 1))), c(2,4))
  expect_equal(dim(as.array(x$flatten(end_dim = 1))), c(4,2))
})

test_that("flip works", {
  x <- tensor(1:10)
  expect_equal(as.array(x$flip(0)), 10:1)

  x <- tensor(matrix(1:10, ncol = 5))
  expect_equal((as.array(x$flip(1))), matrix(c(9L, 10L, 7L, 8L, 5L, 6L, 3L, 4L, 1L, 2L), nrow = 2))
})

test_that("float works", {
  x <- tensor(1:10)
  y <- x$float()
  expect_equal(y$dtype(), "float")

  # float does not copy
  x <- tensor(1)
  y <- x$float()
  y$sub_(1)
  expect_equal(as.array(x), 0)
})

test_that("frac works", {
  x <- tensor(c(2.5, 1.1))
  expect_equal(as.array(x$frac()), c(0.5, 0.1), tol = 1e-6)
  x$frac_()
  expect_equal(as.array(x), c(0.5, 0.1), tol = 1e-6)
})

test_that("gather works", {
  x <- tensor(c(1, 2))
  expect_equal(as.array(x$gather(0, tensor(c(1L, 0L), dtype = "long"))), c(2,1))
})

test_that("ge works", {
  x <- runif(100)
  y <- runif(100)

  x_t <- tensor(x)
  y_t <- tensor(y)

  expect_equal(as.array(x_t$ge(0.5)), x >= 0.5)
  expect_equal(as.array(x_t$ge(y_t)), x >= y)

  z <- tensor(1)
  expect_equal(as.array(z$ge(1)), TRUE)
  expect_equal(as.array(z$ge(0.99)), TRUE)

  x_t$ge_(0.5)
  expect_equal(as.array(x_t), as.numeric(x>= 0.5))
})

test_that("gels works", {
  y <- runif(10)
  X <- matrix(runif(100), ncol = 10)

  expect_silent(tch_gels(tensor(y), tensor(X)))

  # expect_equal(
  #   .lm.fit(X, y)$coefficients,
  #   tch_gels(tensor(y), tensor(X))[[1]] %>% as.array() %>% as.numeric(),
  #   tol = 1e-2
  # )
  # expect_equivalent(
  #   .lm.fit(X, y)$qr,
  #   tch_gels(tensor(y), tensor(X))[[2]] %>% as.array()
  # )
})

test_that("geometric works", {
  x <- tch_empty(100)
  expect_silent(x$geometric_(0.5))
  expect_true(all(as.array(x) > 0))
})

test_that("geqrf works", {
  x <- tch_randn(c(5,5))
  expect_silent(out <- x$geqrf())
  expect_equal(length(out), 2)
})

test_that("ger works", {
  x <- tch_randn(10)
  vec2 <- tch_randn(10)
  expect_silent(x$ger(vec2))
})

test_that("gesv works", {
  x <- tch_randn(c(5,5))
  A <- tch_randn(c(5,5))
  expect_silent(out <- x$gesv(A))
  expect_equal(length(out), 2)
  expect_equal(dim(as.array(out[[1]])), c(5,5))
})

test_that("get_device works", {
  if (tch_cuda_is_available()) {
    x <- tensor(1, device = "CUDA")
    expect_true(is.integer(x$get_device))
  } else {
    x <- tensor(1)
    expect_error(x$get_device())
  }
})

test_that("lt works", {
  x <- tensor(1)
  expect_equal(as.array(x$lt(2)), TRUE)
  expect_equal(as.array(x$lt(tensor(0))), FALSE)

  x <- tensor(1)
  x$lt_(0)
  expect_equal(as.array(x$to("uint8")), FALSE)
  x <- tensor(10)
  x$lt_(tensor(11))
  expect_equal(as.array(x$to("uint8")), TRUE)
})

test_that("gt works", {
  x <- tensor(1)
  expect_equal(as.array(x$gt(0)), TRUE)
  expect_equal(as.array(x$gt(tensor(2))), FALSE)

  x <- tensor(1)
  x$gt_(2)
  expect_equal(as.array(x$to("uint8")), FALSE)
  x <- tensor(10)
  x$gt_(tensor(9))
  expect_equal(as.array(x$to("uint8")), TRUE)
})

test_that("half works", {
  x <- tensor(1)
  x <- x$half()
  expect_equal(x$dtype(), "half")
})

test_that("histc works", {
  x <- tch_randn(1000)
  y <- x$histc(bins = 5)
  expect_equal(sum(as.array(y)), 1000)
})

test_that("index_add_ works", {
  x <- tch_ones(c(5,3))
  t <- tensor(matrix(1:9, nrow = 3), dtype = "float")
  index <- tensor(c(0L, 4L, 2L), dtype = "long")
  expect_silent(x$index_add_(0, index, t))
  expect_equal(as.array(x)[1,], c(2, 5, 8))
})

test_that("index_copy_ works", {
  x <- tch_ones(c(5,3))
  t <- tensor(matrix(1:9, nrow = 3), dtype = "float")
  index <- tensor(c(0L, 4L, 2L), dtype = "long")
  expect_silent(x$index_copy_(0, index, t))
  expect_equal(as.array(x)[1,], c(1, 4, 7))
})

test_that("index_fill_ works", {
  x <- tch_ones(c(5,3))
  index <- tensor(c(0L, 4L, 2L), dtype = "long")
  expect_silent(x$index_fill_(0, index, 0))
  expect_equal(as.array(x)[1,], c(0, 0, 0))
})

test_that("index_put_ works", {
  x <- tch_zeros(c(5))
  indices <- list(
    tensor(0:2, dtype = "long")
  )
  value <- tch_ones(c(1))
  expect_silent(x$index_put_(indices, value))
  expect_equal(sum(as.array(x)), 3)
})

test_that("index select", {
  x <- tch_randn(c(5, 5))
  a <- x$index_select(dim = 1, tensor(c(0,1), dtype = "long"))
  expect_equal(as.array(x)[, 1:2], as.array(a))
})

test_that("int", {
  x <- tensor(c(1,2,3,4))
  x <- x$int()
  expect_equal(x$dtype(), "int")
})

test_that("inverse works", {
  x <- tch_randn(c(5,5))
  expect_equal(as.array(x$inverse()), solve(as.array(x)), tol = 1e-6)
})

test_that("is_contiguous", {
  x <- tch_randn(10)
  expect_true(x$is_contiguous())
})

test_that("is_cuda", {
  x <- tch_randn(10)
  expect_true(x$is_cuda() == FALSE)
})

test_that("is_set_to works", {
  x <- tch_randn(10)
  y <- tch_randn(10)
  expect_true(x$is_set_to(x))
  expect_true(!x$is_set_to(y))
})

test_that("is_signed works", {
  x <- tch_randn(10)
  expect_true(x$is_signed())

  x <- tensor(1:10)
  expect_true(x$is_signed())

  x <- tensor(TRUE)
  expect_true(!x$is_signed())
})

test_that("item works", {
  x <- tch_rand(1)
  expect_equal(x$item(), as.array(x))
  x <- tch_rand(c(10,10))
  expect_error(x$item())
})

test_that("kthvalue", {
  x <- tch_rand(10)
  res <- lapply(x$kthvalue(1), as.array)

  expect_equal(res[[1]], min(as.array(x)))
  expect_equal(res[[2]], which.min(as.array(x)) - 1)
})

test_that("le works", {
  x <- tensor(c(1:10))
  expect_equal(as.array(x$le(5)), 1:10 <= 5)
  x$le_(5)
  expect_equal(as.array(x), as.integer(1:10 <= 5))

  x <- tensor(1:10)
  y <- tensor(c(1:5, 1:5))
  expect_equal(as.array(x$le(y)), 1:10 <= c(1:5, 1:5))
  x$le_(y)
  expect_equal(as.array(x), as.integer(1:10 <= c(1:5, 1:5)))
})


test_that("mean works", {
  x <- runif(100)
  expect_equal(as.array(tch_mean(tensor(x))), mean(x), tol = 1e-7)

  x <- matrix(runif(10), nrow = 2)
  expect_equal(as.array(tch_mean(tensor(x), dim = 0)), apply(x, 2, mean), tol = 1e-7)
  expect_equal(as.array(tch_mean(tensor(x), 1)), apply(x, 1, mean), tol = 1e-7)
  expect_equal(dim(as.array(tch_mean(tensor(x), dim = 0, keepdim = TRUE))), c(1,5))

})

test_that("var works", {
  n <- 10
  x <- runif(n)

  expect_equal(as.array(tch_var(tensor(x))), var(x), tol = 1e-7)
  expect_equal(as.array(tch_var(tensor(x), unbiased = FALSE)), var(x) * (n - 1)/n, tol = 1e-7)
})

test_that("std works", {
  n <- 10
  x <- runif(n)

  expect_equal(as.array(tch_std(tensor(x))), sd(x), tol = 1e-7)
  expect_equal(as.array(tch_std(tensor(x), unbiased = FALSE)), sd(x) * sqrt((n - 1)/n), tol = 1e-7)
})

test_that("min works", {
  x <- runif(10)
  expect_equal(as.array(tch_min(tensor(x))), min(x), tol = 1e-7)
})

test_that("mode works", {
  x <- array(c(1 , 1, 1, 2, 2, 3, 3, 4, 4, 4), c(2,5))
  res <- lapply(tch_mode(tensor(x)), as.matrix)
  expect_equivalent(res[[1]], array(c(1, 4), c(2, 1)))
  expect_equivalent(res[[2]], array(c(1, 4), c(2, 1)))

  res <- lapply(tch_mode(tensor(x), 0), as.matrix)
  expect_equivalent(res[[1]], array(c(1, 1, 2, 3, 4), c(5, 1)))
  expect_equivalent(res[[2]], array(c(1, 0, 0, 0, 1), c(5, 1)))

  res <- lapply(tch_mode(tensor(x), 0, TRUE), as.matrix)
  expect_equivalent(res[[1]], array(c(1, 1, 2, 3, 4), c(1, 5)))
  expect_equivalent(res[[2]], array(c(1, 0, 0, 0, 1), c(1, 5)))
})

test_that("median works", {
  x <- array(1:20, c(4,5))
  expect_equal(lapply(tch_median(tensor(x), 0), as.matrix), list(matrix(c(2, 6, 10, 14, 18), c(5,1)), array(1, c(5,1))), tol = 1e-7)
  expect_equal(lapply(tch_median(tensor(x), 0, TRUE), as.matrix), list(matrix(c(2, 6, 10, 14, 18), c(1,5)), array(1, c(1,5))), tol = 1e-7)
  expect_equal(lapply(tch_median(tensor(x), 1), as.matrix), list(matrix(9:12, c(4,1)), array(2, c(4,1))), tol = 1e-7)
  expect_equal(as.matrix(tch_median(tensor(x))), matrix(10, c(1,1)), tol = 1e-7)

})

test_that("max works", {
  x <- runif(10)
  expect_equal(as.array(tch_max(tensor(x))), max(x), tol = 1e-7)
})

test_that("prod works", {
  x <- array(c(1, 2, 3, -1, -2, -3), c(2, 3))
  x_t <- tensor(x)

  expect_equal(as.array(tch_prod(x_t)), prod(x), tol = 1e-6)
  expect_equal(as.array(tch_prod(x_t, 0)), apply(x, 2, prod), tol = 1e-6)
  expect_equal(as.array(tch_prod(x_t, 0)), as.array(tch_prod(x_t, 0, FALSE)), tol = 1e-6)
  expect_equal(as.array(tch_prod(x_t, 1)), apply(x, 1, prod), tol = 1e-6)
  expect_equal(as.array(tch_prod(x_t, 1)), as.array(tch_prod(x_t, 1, FALSE)), tol = 1e-6)
  expect_equal(as.array(tch_prod(x_t, 0, TRUE)), matrix(apply(x, 2, prod), 1, 3), tol = 1e-6)
  expect_equal(as.array(tch_prod(x_t, 1, TRUE)), matrix(apply(x, 1, prod), 2, 1), tol = 1e-6)
  expect_equal(tch_prod(x_t, 1, TRUE, "double")$dtype(), "double")
})

test_that("logsumexp works", {
  logsumexp <- function(x) log(sum(exp(x)))

  x <- array(runif(5*4*2), dim = c(5, 4, 2))
  t_x <- tensor(x)

  expect_equal(as.array(tch_logsumexp(t_x, 2)), apply(x, c(1, 2), logsumexp), tol = 1e-7)
  expect_equal(as.array(tch_logsumexp(t_x, 1)), apply(x, c(1, 3), logsumexp), tol = 1e-7)
  expect_equal(as.array(tch_logsumexp(t_x, 0)), apply(x, c(2, 3), logsumexp), tol = 1e-7)
  expect_error(as.array(tch_logsumexp(t_x))) # logsumexp() missing 1 required positional arguments: "dim"
})

test_that("mm works", {
  x <- matrix(runif(10), ncol = 5)
  y <- matrix(runif(10), nrow = 5)

  res_t <- as.array(tensor(x)$mm(tensor(y)))
  res_r <- x %*% y

  expect_equal(res_t, res_r, tol = 1e-7)

  res_t <- as.array(tch_mm(tensor(x), tensor(y)))
  expect_equal(res_t, res_r, tol = 1e-7)
})

test_that("mul works", {
  x <- tensor(2)
  y <- tensor(3)

  expect_equal(as.array(x*y), 6, tol = 1e-7)
  expect_equal(as.array(x*3), 6, tol = 1e-7)

  x <- runif(100)

  expect_equal(as.array(tensor(x)*3), x*3, tol = 1e-7)
})

test_that("permute works", {
  x <- tensor(array(1:100, dim = c(4, 5, 5)))
  y <- as.array(x$permute(c(2,1,0)))

  expect_equal(y, aperm(array(1:100, dim = c(4, 5, 5)), c(3,2,1)))
  expect_equal(as.array(tch_permute(x, c(2,1,0))), aperm(array(1:100, dim = c(4, 5, 5)), c(3,2,1)))
})

test_that("pow works", {
  x <- tensor(2)
  y <- tensor(3)

  expect_equal(as.array(x^y), 8, tol = 1e-7)
  expect_equal(as.array(x^3), 8, tol = 1e-7)

  x <- runif(100)
  expect_equal(as.array(tensor(x)^3), x^3, tol = 1e-7)
})

test_that("qr works", {

  x <- matrix(runif(100), ncol = 10)
  a <- qr(x)
  out <- tch_qr(tensor(x)) %>% lapply(as.array)

  expect_equal(qr.Q(a), out[[1]], tol = 1e-5)
  expect_equal(qr.R(a), out[[2]], tol = 1e-5)
})

test_that("sub works", {
  x <- tensor(2)
  y <- tensor(3)

  expect_equal(as.array(x$sub(y)), -1)
  expect_equal(as.array(x - y), -1)
  expect_equal(as.array(x - 1), 1)
})

test_that("sub_ works", {
  x <- tensor(2)
  x$sub_(1)

  expect_equal(as.array(x), 1)
})

test_that("sum works", {
  x <- 1:10
  expect_equal(as.array(tch_sum(tensor(x))), sum(x))

  x <- runif(100)
  expect_equal(as.array(tch_sum(tensor(x))), sum(x), tol = 1e-6)
})

test_that("transpose works", {
  x <- matrix(runif(6), ncol = 3)
  x_t <- tensor(x)
  expect_equal(as.array(x_t$transpose(0, 1)), t(x), tol = 1e-7)

  expect_error(x_t$transpose())

  x_t$transpose_(0, 1)
  expect_equal(as.array(x_t), t(x), tol = 1e-7)

})

test_that("t works", {
  x <- matrix(runif(6), ncol = 3)

  expect_equal(as.array(tch_t(tensor(x))), t(x), tol = 1e-7)

  x <- matrix(1:6, ncol = 3)

  expect_equal(as.array(tch_t(tensor(x))), t(x))

  expect_error(t(tensor(array(1:12, dim = c(2,2,3)))))
})

test_that("to works", {
  x <- matrix(runif(6), ncol = 3)

  expect_equal(as.array(tensor(x)$to(dtype = "int")), matrix(0L, ncol = 3, nrow = 2))
})

test_that("topk works", {
  x <- array(c(1, 2, 3, -1, -2, -3), c(2, 3))
  x_t <- tensor(x)
  expect_equal(length(x_t$topk(3)), 2)
  expect_equal(length(x_t$topk(3, 1, FALSE, FALSE)), 2)
  expect_equal(length(x_t$topk(3, 1, FALSE, TRUE)), 2)
  expect_equal(length(x_t$topk(3, 1, TRUE, FALSE)), 2)
  expect_equal(length(x_t$topk(3, 1, TRUE, TRUE)), 2)

  expect_error(x_t$topk())
  expect_error(x_t$topk(4))

})

test_that("log family works", {
  x <- runif(100)
  expect_equal(as.array(tch_log(tensor(x))), log(x), tol = 1e-7)
  expect_equal(as.array(tch_log2(tensor(x))), log2(x), tol = 1e-7)
  expect_equal(as.array(tch_log10(tensor(x))), log10(x), tol = 1e-7)
  expect_equal(as.array(tch_log1p(tensor(x))), log1p(x), tol = 1e-7)

  x_t <- tensor(x)
  x_t$log_()
  expect_equal(as.array(x_t), log(x), tol = 1e-7)


  x_t <- tensor(x)
  x_t$log2_()
  expect_equal(as.array(x_t), log2(x), tol = 1e-7)

  x_t <- tensor(x)
  x_t$log10_()
  expect_equal(as.array(x_t), log10(x), tol = 1e-7)

  x_t <- tensor(x)
  x_t$log1p_()
  expect_equal(as.array(x_t), log1p(x), tol = 1e-7)
})

test_that("tril works", {
  x <- array(1, c(3, 3))
  matrix_help <- function(x) matrix(x, 3, 3, byrow = TRUE)
  expect_equal(as.array(tch_tril(tensor(x))), matrix_help(c(1, 0, 0,
                                                            1, 1, 0,
                                                            1, 1, 1)))
  expect_equal(as.array(tch_tril(tensor(x), 1)), matrix_help(c(1, 1, 0,
                                                               1, 1, 1,
                                                               1, 1, 1)))
  expect_equal(as.array(tch_tril(tensor(x), 2)), matrix_help(c(1, 1, 1,
                                                               1, 1, 1,
                                                               1, 1, 1)))
  expect_equal(as.array(tch_tril(tensor(x), -1)), matrix_help(c(0, 0, 0,
                                                                1, 0, 0,
                                                                1, 1, 0)))
  expect_equal(as.array(tch_tril(tensor(x), -2)), matrix_help(c(0, 0, 0,
                                                                0, 0, 0,
                                                                1, 0, 0)))
})

test_that("triu works", {
  x <- array(1, c(3, 3))
  matrix_help <- function(x) matrix(x, 3, 3, byrow = TRUE)
  expect_equal(as.array(tch_triu(tensor(x))), matrix_help(c(1, 1, 1,
                                                            0, 1, 1,
                                                            0, 0, 1)))
  expect_equal(as.array(tch_triu(tensor(x), 1)), matrix_help(c(0, 1, 1,
                                                               0, 0, 1,
                                                               0, 0, 0)))
  expect_equal(as.array(tch_triu(tensor(x), 2)), matrix_help(c(0, 0, 1,
                                                               0, 0, 0,
                                                               0, 0, 0)))
  expect_equal(as.array(tch_triu(tensor(x), -1)), matrix_help(c(1, 1, 1,
                                                                1, 1, 1,
                                                                0, 1, 1)))
  expect_equal(as.array(tch_triu(tensor(x), -2)), matrix_help(c(1, 1, 1,
                                                                1, 1, 1,
                                                                1, 1, 1)))
})

test_that("rep (torch's repeat) works", {
  x <- tensor(array(1:6, c(1, 2, 3)))
  expect_equal(dim(as.array(x$rep(c(2, 2, 2)))), c(2, 4, 6))
  expect_equal(dim(as.array(x$rep(c(1, 1, 1, 2)))), c(1, 1, 2, 6))

  expect_error(x$rep(c(2, 2)))
  expect_error(x$rep(c(2, 2, 0.5)))
})

test_that("reciprocal works", {
  x <- tensor(array(c(-Inf, -10, -0.1, 0, 0.1, 10, Inf)))
  expect_equal(as.array(x$reciprocal()), c(0, -0.1, -10, Inf, 10, 0.1, 0), tol = 1e-7)

  x$reciprocal_()
  expect_equal(as.array(x), c(0, -0.1, -10, Inf, 10, 0.1, 0), tol = 1e-7)
})

test_that("round works", {
  x <- tensor(array(c(-1.1, -0.1, 0.1, 1.5, 1.51, 2.5, Inf)))
  expect_equal(as.array(tch_round(x)), c(-1, 0, 0, 2, 2, 2, Inf), tol = 1e-7)

  x$round_()
  expect_equal(as.array(x), c(-1, 0, 0, 2, 2, 2, Inf), tol = 1e-7)

  y <- tensor(array(c(0.5, 1.5, 2.5, 3.5, 4.5)))
  expect_equal(as.array(tch_round(y)), c(0, 2, 2, 4, 4), tol = 1e-7) #what??
})

test_that("rsqrt works", {
  x <- array(c(0.1, 1.5, 1.51, 2.5, Inf))
  expect_equivalent(as.array(tch_rsqrt(tensor(x))), 1/sqrt(x), tol = 1e-7)

  x_t <- tensor(x)
  x_t$rsqrt_()
  expect_equivalent(as.array(x_t), 1/sqrt(x), tol = 1e-7)
})

test_that("sigmoid works", {
  x <- array(c(rnorm(10), Inf))
  expect_equivalent(as.array(tch_sigmoid(tensor(x))), 1/(1 + exp(-x)), tol = 1e-7)
})

test_that("sign works", {
  x <- array(c(rnorm(10), Inf))
  expect_equivalent(as.array(tch_sign(tensor(x))), sign(x), tol = 1e-7)
})

test_that("sqrt works", {
  x <- array(c(0.1, 1.5, 1.51, 2.5, Inf))
  res <- as.array(tch_sqrt(tensor(x)))
  res2 <- as.numeric(sqrt(x))
  expect_equivalent(res, res2, tol = 1e-7)

  x_t <- tensor(x)
  x_t$sqrt_()

  res <- as.array(x_t)
  expect_equivalent(res, res2, tol = 1e-7)
})

test_that("trunc works", {
  x <- tensor(array(c(-1.1, -0.1, 0.1, 1.5, 1.51, 2.5, Inf)))
  expect_equal(as.array(tch_trunc(x)), c(-1, 0, 0, 1, 1, 2, Inf), tol = 1e-7)

  y <- tensor(array(c(0.5, 1.5, 2.5, 3.5, 4.5)))
  expect_equal(as.array(tch_trunc(y)), c(0, 1, 2, 3, 4), tol = 1e-7)

  x_t <- tensor(x)
  x_t$trunc_()
  expect_equal(as.array(x_t), c(-1, 0, 0, 1, 1, 2, Inf), tol = 1e-7)
})

test_that("zero_ works", {
  x <- tensor(1)
  x$zero_()
  expect_equal(as.array(x), 0)
})

context("numeric tensors")

test_that("creation of 1d numeric tensor", {
  x <- runif(100)
  expect_equal(as.array(tensor(x)), x, tol = 1e-7)
})

test_that("creation of 2d numeric tensor", {
  x <- matrix(runif(100), ncol = 10)
  expect_equal(as.array(tensor(x)), x, tol = 1e-7)
})

test_that("creation of 3d numeric tensor", {
  x <- array(runif(80), dim = c(20, 2, 2))
  expect_equal(as.array(tensor(x)), x, tol = 1e-7)
})

context("factory functions")

test_that("tensor from tensors", {
  x <- tensor(runif(10), requires_grad = TRUE)
  expect_silent(tensor(x))
})

test_that("tensor is really cloned in tensors", {
  x <- tensor(1, requires_grad = TRUE)
  w <- tensor(2, requires_grad = TRUE)
  b <- tensor(3, requires_grad = TRUE)
  a <- tensor(x, requires_grad = TRUE)
  y <- w * x + b
  y$backward()
  expect_error(as.array(a$grad)) # TODO handle undefined tensors in as.array.
})

test_that("randn", {
  x <- tch_randn(c(2,2))
  expect_equal(dim(as.array(x)), c(2L, 2L))
  expect_equal(x$dtype(), "float")

  expect_error(x <- tch_randn(c(2,2), dtype = "int"))

  x <- tch_randn(c(2,2), dtype = "double")
  expect_equal(x$dtype(), "double")
})

test_that("arange", {
  x <- tch_arange(5)
  expect_equal(as.array(x), c(0L, 1L, 2L, 3L, 4L))
  expect_null(dim(as.array(x)))
  expect_equal(x$dtype(), "float")

  y <- tch_arange(1, 4)
  expect_equal(as.array(y), c(1, 2, 3))
  expect_null(dim(as.array(y)))

  z <- tch_arange(1, 2.5, 0.5)
  expect_equal(as.array(z), c(1.0, 1.5, 2.0))
  expect_null(dim(as.array(z)))

  x <- tch_arange(5, dtype = "int")
  expect_equal(x$dtype(), "int")
})

test_that("empty", {
  x <- tch_empty(c(2, 4))
  expect_equal(dim(as.array(x)), c(2L, 4L))

  x <- tch_empty(c(2, 4), dtype = "int")
  expect_equal(x$dtype(), "int")
})

test_that("eye", {
  x <- tch_eye(2, 4)
  expect_equal(dim(as.array(x)), c(2L, 4L))
  expect_equal(as.array(x), diag(nrow = 2, ncol = 4))

  y <- tch_eye(3)
  expect_equal(dim(as.array(y)), c(3L, 3L))
  expect_equal(as.array(y), diag(3))
})

test_that("full", {
  x <- tch_full(c(2, 4), 10)
  expect_equal(dim(as.array(x)), c(2L, 4L))
  expect_equal(as.array(x), array(10, c(2, 4)))

  y <- tch_full(c(2, 4, 3), -1)
  expect_equal(dim(as.array(y)), c(2L, 4L, 3L))
  expect_equal(as.array(y), array(-1, c(2, 4, 3)))
})

test_that("linspace", {
  x <- tch_linspace(3, 10, steps = 5)
  expect_null(dim(as.array(x)))
  expect_equal(as.array(x), c(3, 4.75, 6.50, 8.25, 10))

  y <- tch_linspace(-10, 10, steps = 5)
  expect_null(dim(as.array(y)))
  expect_equal(as.array(y), c(-10, -5, 0, 5, 10))
})

test_that("logspace", {
  x <- tch_logspace(-10, 10, steps = 5)
  expect_null(dim(as.array(x)))
  expect_equal(as.array(x), c(1.0e-10,  1.0e-05,  1.0,  1.0e+05,  1.0e+10), tol = 1e-4)

  y <- tch_logspace(start=0.1, end=1.0, steps=5)
  expect_null(dim(as.array(y)))
  expect_equal(as.array(y), c(1.2589, 2.1135, 3.5481, 5.9566, 10.0000), tol = 1e-4)
})

test_that("ones", {
  x <- tch_ones(c(2, 4))
  expect_equal(dim(as.array(x)), c(2L, 4L))
  expect_equal(as.array(x), array(1, c(2, 4)))

  y <- tch_ones(5)
  expect_null(dim(as.array(y)))
  expect_equal(as.array(y), rep(1, 5))
})

test_that("rand", {
  x <- tch_rand(c(2,2))
  expect_equal(dim(as.array(x)), c(2L, 2L))
  expect_equal(x$dtype(), "float")

  expect_error(x <- tch_rand(c(2, 2), dtype = "int"))

  x <- tch_rand(c(2,2), dtype = "double")
  expect_equal(x$dtype(), "double")
})

test_that("randint", {
  x <- tch_randint(10, c(2, 2))
  expect_equal(dim(as.array(x)), c(2L, 2L))
  expect_equal(x$dtype(), "float")

  y <- tch_randint(3, 10, c(2, 2), dtype = "double")
  expect_equal(y$dtype(), "double")
})

test_that("randperm", {
  x <- tch_randperm(10)
  expect_null(dim(as.array(x)))
  expect_equal(x$dtype(), "float")
})

test_that("sin works", {
  x <- runif(100)
  expect_equal(as.array(tch_sin(tensor(x))), sin(x), tol = 1e-7)

  x_t <- tensor(x)
  x_t$sin_()
  expect_equal(as.array(x_t), sin(x), tol = 1e-7)
})

test_that("sinh works", {
  x <- runif(100)
  expect_equal(as.array(tch_sinh(tensor(x))), sinh(x), tol = 1e-7)

  x_t <- tensor(x)
  x_t$sinh_()
  expect_equal(as.array(x_t), sinh(x), tol = 1e-7)
})

test_that("tan works", {
  x <- runif(100)
  expect_equal(as.array(tch_tan(tensor(x))), tan(x), tol = 1e-7)

  x_t <- tensor(x)
  x_t$tan_()
  expect_equal(as.array(x_t), tan(x), tol = 1e-7)
})

test_that("tanh works", {
  x <- runif(100)
  expect_equal(as.array(tch_tanh(tensor(x))), tanh(x), tol = 1e-7)

  x_t <- tensor(x)
  x_t$tanh_()
  expect_equal(as.array(x_t), tanh(x), tol = 1e-7)
})

test_that("unfold works", {
  x <- array(c(1, 2, 3, -1, -2, -3), c(2, 3))
  x_t <- tensor(x)
  expect_equal(dim(as.array(x_t$unfold(0, 1, 1))), c(2, 3, 1), tol = 1e-7)
  expect_equal(dim(as.array(x_t$unfold(0, 2, 1))), c(1, 3, 2), tol = 1e-7)
  expect_equal(dim(as.array(x_t$unfold(0, 1, 2))), c(1, 3, 1), tol = 1e-7)
  expect_equal(dim(as.array(x_t$unfold(0, 2, 2))), c(1, 3, 2), tol = 1e-7)
  expect_equal(dim(as.array(x_t$unfold(1, 1, 1))), c(2, 3, 1), tol = 1e-7)
  expect_equal(dim(as.array(x_t$unfold(1, 2, 1))), c(2, 2, 2), tol = 1e-7)
  expect_equal(dim(as.array(x_t$unfold(1, 1, 2))), c(2, 2, 1), tol = 1e-7)
  expect_equal(dim(as.array(x_t$unfold(1, 2, 2))), c(2, 1, 2), tol = 1e-7)

  expect_error(as.array(x_t$unfold(0, 3, 1)))
  expect_error(as.array(x_t$unfold(2, 1, 1)))

})

test_that("unique works", {
  x <- matrix(c(c(0,0,0),
                c(0,0,1),
                c(0,0,1)), 3, byrow = TRUE)
  x_t <- tensor(x)

  # return_inverse = FALSE
  expect_equal(as.array(x_t$unique()), c(1, 0))
  expect_equal(as.array(x_t$unique(sorted = TRUE)), c(0, 1))
  expect_equal(as.array(x_t$unique(dim = 0)), matrix(c(c(0,0,0),
                                                       c(0,0,1)), 2, byrow = TRUE))
  expect_equal(as.array(x_t$unique(dim = 1)), matrix(c(c(0,0),
                                                       c(0,1),
                                                       c(0,1)), 3, byrow = TRUE))

  # return_inverse = TRUE
  expect_equal(class(x_t$unique(return_inverse = TRUE)), "list")
  expect_equal(lapply(x_t$unique(return_inverse = TRUE), as.array), list(c(1, 0), matrix(c(c(1,1,1),
                                                                                           c(1,1,0),
                                                                                           c(1,1,0)), 3, byrow = TRUE)))
  expect_equal(lapply(x_t$unique(sorted = TRUE, return_inverse = TRUE), as.array), list(c(0, 1), matrix(c(c(0,0,0),
                                                                                                          c(0,0,1),
                                                                                                          c(0,0,1)), 3, byrow = TRUE)))
  expect_equal(lapply(x_t$unique(return_inverse = TRUE, dim = 0), as.array), list(matrix(c(c(0,0,0),
                                                                                           c(0,0,1)), 2, byrow = TRUE),
                                                                                  c(0, 1, 1)))
  expect_equal(lapply(x_t$unique(return_inverse = TRUE, dim = 1), as.array), list(matrix(c(c(0,0),
                                                                                           c(0,1),
                                                                                           c(0,1)), 3, byrow = TRUE),
                                                                                  c(0, 0, 1)))
})

test_that("unsqueeze works", {
  x <- array(0, c(2, 2, 2))
  x_t <- tensor(x)
  expect_equal(dim(as.array(x_t$unsqueeze(0))), c(1, 2, 2, 2))
  expect_equal(dim(as.array(x_t$unsqueeze(1))), c(2, 1, 2, 2))
  expect_equal(dim(as.array(x_t$unsqueeze(2))), c(2, 2, 1, 2))
  expect_equal(dim(as.array(x_t$unsqueeze(3))), c(2, 2, 2, 1))

  x_t$unsqueeze_(3)
  expect_equal(dim(as.array(x_t)), c(2, 2, 2, 1))
})

test_that("zeros", {
  x <- tch_zeros(c(2, 4))
  expect_equal(dim(as.array(x)), c(2L, 4L))
  expect_equal(as.array(x), array(0, c(2, 4)))

  y <- tch_zeros(5)
  expect_null(dim(as.array(y)))
  expect_equal(as.array(y), rep(0, 5))
})

test_that("sort works", {
  x <- array(1:12, c(2, 2, 3))
  x_t <- tensor(x)
  expect_equal(length(as.array(x_t$sort())), 2)
  expect_equal(length(as.array(x_t$sort(0))), 2)
  expect_equal(length(as.array(x_t$sort(1, TRUE))), 2)
  expect_equal(length(as.array(x_t$sort(2, FALSE))), 2)

  expect_error(x_t$sort(3))
  expect_error(x_t$sort(-4))

})
