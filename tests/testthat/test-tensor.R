library(torch)

context("tensor options")

test_that("requires_grad", {
  x <- tensor(x = runif(10), requires_grad = TRUE)
  expect_identical(class(x)[1], "tensor")
})

test_that("dtype", {
  type <- typeof(as.array(tensor(1:10, dtype = "kDouble")))
  expect_identical(type, "double")
})

test_that("device", {
  # TODO can't test device without a gpu :(
  expect_identical(1, 1)
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
  expect_identical(as.array(tch_abs(tensor(x))), abs(x))
})

test_that("acos works", {
  x <- array(-runif(80), dim = c(20, 2, 2))
  expect_equal(as.array(tch_acos(tensor(x))), acos(x))
})

test_that("add works", {
  x <- array((-80):(-1), dim = c(20, 2, 2))
  y <- array((-80):(-1), dim = c(20, 2, 2))
  expect_identical(as.array(tensor(x) + tensor(y)), x + y)

  x <- array(-runif(80), dim = c(20, 2, 2))
  y <- array(-runif(80), dim = c(20, 2, 2))
  expect_identical(as.array(tensor(x) + tensor(y)), x + y)

  x <- runif(100)
  expect_equal(as.array(tensor(x) + 1), x + 1)
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
})

test_that("addcdiv works", {

  x <- tensor(matrix(runif(3), nrow = 1, ncol = 3))
  t1 <- tensor(array(runif(3), dim = c(3, 1)))
  t2 <- tensor(array(runif(3), dim = c(1, 3)))

  res <- as.array(tch_addcdiv(x, t1, t2, 0.1))

  expect_true(is.array(res))
  expect_identical(dim(res), c(3L, 3L))
})

test_that("addcmul works", {

  x <- tensor(matrix(runif(3), nrow = 1, ncol = 3))
  t1 <- tensor(array(runif(3), dim = c(3, 1)))
  t2 <- tensor(array(runif(3), dim = c(1, 3)))

  res <- as.array(tch_addcmul(x, t1, t2, 0.1))

  expect_true(is.array(res))
  expect_identical(dim(res), c(3L, 3L))
})

test_that("addmm works", {

  x <- tensor(matrix(runif(6), nrow = 2, ncol = 3))
  mat1 <- tensor(array(runif(6), dim = c(2, 3)))
  mat2 <- tensor(array(runif(6), dim = c(3, 3)))

  res <- as.array(tch_addmm(x, mat1, mat2, 1))

  expect_true(is.array(res))
  expect_identical(dim(res), c(2L, 3L))
})

test_that("addmv works", {

  x <- tensor(runif(2))
  mat <- tensor(array(runif(6), dim = c(2, 3)))
  vec <- tensor(runif(3))

  res <- as.array(tch_addmv(x, mat, vec))

  expect_identical(length(res), 2L)
})

test_that("addr works", {
  vec1 <- tensor(c(1,2,3))
  vec2 <- tensor(c(1,2))
  x <- tensor(matrix(0, nrow = 3, ncol = 2))

  res <- as.array(tch_addr(x, vec1, vec2))
  res_ <- matrix(c(1,2,3,2,4,6), ncol = 2)
  expect_identical(res, res_)
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
  y <- tensor(c(1,2,3,4,5) + 1e-5)
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

test_that("as_strided works", {
  # TODO better testint as_strided - undocument in the python side too.
  l <- array(1:6000, dim = c(10, 20, 30))
  x <- tensor(l)

  k <- tch_as_strided(x, 0, 2)

  expect_identical(class(k)[1], "tensor")

  k <- tch_as_strided(x, 0, 2, 1)

  expect_identical(class(k)[1], "tensor")
})

test_that("asin works", {
  x <- runif(100)
  expect_equal(as.array(tch_asin(tensor(x))), asin(x))
})

test_that("atan works", {
  x <- runif(100)
  expect_equal(as.array(tch_atan(tensor(x))), atan(x))
})

test_that("atan2 works", {
  x <- runif(100)
  y <- runif(100)

  expect_equal(as.array(tch_atan2(tensor(x), tensor(y))), atan2(x, y))
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
})

test_that("bernoulli works", {
  x <- tensor(runif(10))
  expect_silent(tch_bernoulli(x))

  x <- tensor(rep(0, 100))
  expect_equal(sum(as.array(tch_bernoulli(x))), 0)

  x <- tensor(rep(1, 100))
  expect_equal(sum(as.array(tch_bernoulli(x))), 100)
})

test_that("bincount works", {
  x <- sample(0:9, 500, replace = TRUE)
  expect_equal(as.array(tch_bincount(tensor(x))), as.integer(table(x)))

  x <- sample(0:9, 500, replace = TRUE)
  weights <- runif(500)

  expect_equal(as.array(tch_bincount(tensor(x), tensor(weights))), as.numeric(tapply(weights, x, sum)))
})

test_that("bmm works", {
  x <- tensor(array(runif(120), dim = c(10, 3, 4)))
  y <- tensor(array(runif(200), dim = c(10, 4, 5)))
  res <- as.array(tch_bmm(x, y))

  expect_equal(dim(res), c(10, 3, 5))
})

test_that("btrifact works", {
  x <- tensor(array(runif(18), dim = c(2, 3, 3)))
  res <- tch_btrifact(x)

  a_lu <- as.array(res[[1]])
  pivot <- as.array(res[[2]])

  expect_equal(dim(a_lu), c(2, 3, 3))
  expect_equal(dim(pivot), c(2, 3))
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

test_that("chunk works", {
  a <- array(runif(100), dim = c(4, 5, 5))
  x <- tensor(a)
  chunks <- tch_chunk(x, 2, 0)

  expect_equal(as.array(chunks[[1]]), a[1:2,,])
  expect_equal(as.array(chunks[[2]]), a[3:4,,])
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

test_that("clone_ works", {
  x <- tensor(1:10)
  y <- x$clone_()
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
  expect_equal(as.array(tch_cosh(x)), cosh(c(pi, 2*pi)))
})

test_that("cosh_ works", {
  x <- tensor(c(pi, 2*pi))
  x$cosh_()
  expect_equal(as.array(x), cosh(c(pi, 2*pi)))
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

test_that("det works", {
  x <- matrix(runif(36), ncol = 6)

  expect_equal(
    as.array(tch_det(tensor(x))),
    det(x)
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
  expect_equal(as.array(tch_digamma(x)), digamma(c(1, 0.5)))
})

test_that("digamma_ works", {
  x <- tensor(c(1, 0.5))
  x$digamma_()
  expect_equal(as.array(x), digamma(c(1, 0.5)))
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
    sqrt(sum((1:10 - 10:1)^2))
  )
})

test_that("div works", {
  x <- tensor(as.numeric(1:10))
  y <- tensor(as.numeric(10:1))
  expect_equal(as.array(x / y), 1:10/10:1)
  expect_equal(as.array(x / 2), 1:10/2)

  x <- tensor(1:10)
  y <- tensor(10:1)
  expect_equal(as.array(x / y), as.integer(1:10/10:1))
  expect_equal(as.array(x / 2L), as.integer(1:10/2L))
})

test_that("div_ works", {
  x <- tensor(as.numeric(1:10))
  y <- tensor(as.numeric(10:1))
  x$div_(y)
  expect_equal(as.array(x), 1:10/10:1)

  x <- tensor(as.numeric(1:10))
  x$div_(2)
  expect_equal(as.array(x), 1:10/2)

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

# TODO dtype
# test_that("dtype works", {
#   x <- tensor(1:10)
#   expect_equal(x$dtype(), "kInt")
#   x <- tensor(runif(10))
#   expect_equal(x$dtype(), "kDouble")
#   # test for other tensor types.
# })

test_that("eig works", {
  x_r <- cbind(c(1,-1), c(-1,1))

  x <- tensor(x_r)
  out <- tch_eig(x, eigenvectors = TRUE)
  out_r <- eigen(x_r)

  expect_equal(as.array(out[[1]])[,1], out_r$values)
  expect_equal(as.array(out[[2]]), -out_r$vectors)
})

test_that("gels works", {
  y <- runif(10)
  X <- matrix(runif(100), ncol = 10)

  expect_equal(
    .lm.fit(X, y)$coefficients,
    tch_gels(tensor(y), tensor(X))[[1]] %>% as.array() %>% as.numeric()
  )
  # expect_equivalent(
  #   .lm.fit(X, y)$qr,
  #   tch_gels(tensor(y), tensor(X))[[2]] %>% as.array()
  # )
})


test_that("mean works", {
  x <- runif(100)
  expect_equal(as.array(tch_mean(tensor(x))), mean(x))
})

test_that("mm works", {
  x <- matrix(runif(10), ncol = 5)
  y <- matrix(runif(10), nrow = 5)

  res_t <- as.array(tensor(x)$mm(tensor(y)))
  res_r <- x %*% y

  expect_equal(res_t, res_r)

  res_t <- as.array(tch_mm(tensor(x), tensor(y)))
  expect_equal(res_t, res_r)
})

test_that("mul works", {
  x <- tensor(2)
  y <- tensor(3)

  expect_equal(as.array(x*y), 6)
  expect_equal(as.array(x*3), 6)

  x <- runif(100)

  expect_equal(as.array(tensor(x)*3), x*3)
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

  expect_equal(as.array(x^y), 8)
  expect_equal(as.array(x^3), 8)

  x <- runif(100)
  expect_equal(as.array(tensor(x)^3), x^3)
})

test_that("qr works", {

  x <- matrix(runif(100), ncol = 10)
  a <- qr(x)
  out <- tch_qr(tensor(x)) %>% lapply(as.array)

  expect_equal(qr.Q(a), out[[1]])
  expect_equal(qr.R(a), out[[2]])
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
  expect_equal(as.array(tch_sum(tensor(x))), sum(x))
})

test_that("t works", {
  x <- matrix(runif(6), ncol = 3)

  expect_equal(as.array(tch_t(tensor(x))), t(x))

  x <- matrix(1:6, ncol = 3)

  expect_equal(as.array(tch_t(tensor(x))), t(x))

  expect_error(t(tensor(array(1:12, dim = c(2,2,3)))))
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
