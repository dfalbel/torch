#' @useDynLib torch
NULL

#' Create torch Tensor from R object
#'
#' @param x an R vector, matrix or array.
#'
#' @export
tensor <- function(x) {

  dimension <- dim(x)

  if (is.null(dimension)) {
    dimension <- length(x)
  }

  if (!is.null(dim(x))) {
    x <- aperm(x, perm = seq(length(dim(x)), 1))
  }

  `torch::Tensor`$dispatch(tensor_(x, dimension))
}

#' Tensor casting
#'
#' Casts an object with class [tensor] to an R atomic vector, matrix or array.
#'
#' @param x tensor object to be casted to an R array.
#'
#' @examples
#' x <- tensor(array(1:8, dim = c(2, 2, 2)))
#' as.array(x)
#'
#' @export
as.array.tensor <- function(x) {
  x$as_vector()
}

#' @export
as.matrix.tensor <- function(x) {
  as.matrix(x$as_vector())
}

#' @export
abs.tensor <- function(x) {
  x$abs()
}

#' @export
acos.tensor <- function(x) {
  x$acos()
}

#' @export
`+.tensor` <- function(x, y) {
  x$add(y)
}

#' @export
addbmm.tensor <- function(x, batch1, batch2, beta = 1, alpha = 1) {
  x$addbmm(batch1, batch2, beta, alpha)
}

#' @export
addcdiv.tensor <- function(x, tensor1, tensor2, value = 1) {
  x$addcdiv(tensor1, tensor2, value)
}

#' @export
addcmul.tensor <- function(x, tensor1, tensor2, value = 1) {
  x$addcmul(tensor1, tensor2, value)
}

#' @export
addmm.tensor <- function(x, mat1, mat2, beta = 1, alpha = 1) {
  x$addmm(mat1, mat2, beta, alpha)
}

#' @export
addmv.tensor <- function(x, mat, vec, alpha = 1, beta = 1) {
  x$addmv(mat, vec, alpha, beta)
}
