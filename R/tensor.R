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

  ten <- tensor_(x, dimension)
  class(ten) <- "tensor"

  ten
}

#' @export
print.tensor <- function(x, ...) {
  print_tensor_(x)
}

#' @useDynLib torch
NULL

#' @export
as.array.tensor <- function(x) {

  a <- as_array_tensor_(x)

  if (length(a$dim) == 1) {
    out <- a$vec
  } else if (length(a$dim) == 2L) {
    out <- t(matrix(a$vec, ncol = a$dim[1], nrow = a$dim[2]))
  } else {
    out <- aperm(array(a$vec, dim = rev(a$dim)), seq(length(a$dim), 1))
  }

  out
}

#' @export
as.matrix.tensor <- function(x) {
  as.matrix(as.array(x))
}
