library(torch)

context("grad mode")

test_that("grad_mode works", {

  x <- tensor(runif(100), requires_grad = TRUE)
  s <- tensor(runif(100))

  expect_error(x$sub_(s))
  expect_silent(with_no_grad(x$sub_(s)))
})
