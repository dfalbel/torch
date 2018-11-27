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

#' as.array
#'
#' Casts a `torch::Tensor` to an R array
#'
#' @param x torch::Tensor object to be casted to an R array.
#'
#' @examples
#' x <- tensor(array(1:8, dim = c(2, 2, 2)))
#' as.array(x)
#'
#' @export
`as.array.torch::Tensor` <- function(x) {
  x$as_vector()
}

#' @export
`as.matrix.torch::Tensor` <- function(x) {
  as.matrix(x$as_vector())
}

#' @export
`abs.torch::Tensor` <- function(x) {
  x$abs()
}

#' @export
`acos.torch::Tensor` <- function(x) {
  x$acos()
}

#' @export
`+.torch::Tensor` <- function(x, y) {
  x$add(y)
}

#' @export
`addbmm.torch::Tensor` <- function(x, batch1, batch2, beta = 1, alpha = 1) {
  x$addbmm(batch1, batch2, beta, alpha)
}

#' @export
`addcdiv.torch::Tensor` <- function(x, tensor1, tensor2, value = 1) {
  x$addcdiv(tensor1, tensor2, value)
}

#' @export
`addcmul.torch::Tensor` <- function(x, tensor1, tensor2, value = 1) {
  x$addcmul(tensor1, tensor2, value)
}

#' @export
`addmm.torch::Tensor` <- function(x, mat1, mat2, beta = 1, alpha = 1) {
  x$addmm(mat1, mat2, beta, alpha)
}
