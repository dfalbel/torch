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
