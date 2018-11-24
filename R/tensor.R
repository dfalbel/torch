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
