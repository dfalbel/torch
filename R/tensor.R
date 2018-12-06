#' @useDynLib torch
NULL

#' Create torch Tensor from R object
#'
#' @param x an R vector, matrix or array.
#'
#' @note it uses the R type when creating the tensor.
#'
#' @examples
#' tensor(1:10)
#' tensor(array(runif(8), dim = c(2, 2, 2)))
#' tensor(matrix(c(TRUE, FALSE), nrow = 3, ncol = 4))
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
addmv.tensor <- function(x, mat, vec, beta = 1, alpha = 1) {
  x$addmv(mat, vec, beta, alpha)
}

#' @export
addr.tensor <- function(x, vec1, vec2, beta = 1, alpha = 1) {
  x$addr(vec1, vec2, beta, alpha)
}

#' all
#'
#' @param x tensor object
#' @param dim if negative (the default) will reduce to a scalar. Otherwise it will
#' return TRUE if all elements in each row of the tensor in the given dimension
#' `dim` are TRUE, FALSE otherwise.
#' @param keepdim If keepdim is TRUE, the output tensor is of the same size as
#' input except in the dimension dim where it is of size 1. Otherwise, dim is
#' squeezed [squeeze()], resulting in the output tensor having 1 fewer
#' dimension than input.
#' @param na.rm won't be used by the function. Only there to be compatible with
#' [all] generic.
#'
#' @examples
#' x <- tensor(array(c(TRUE, FALSE, TRUE, TRUE), dim = c(2, 2)))
#' all(x)
#' all(x, dim = 0)
#' all(x, dim = 1, keepdim = FALSE)
#' @export
all.tensor <- function(x, dim = -1, keepdim = FALSE, na.rm = FALSE) {
  if (na.rm) warning("tensor's don't use the na.rm argument!")
  x$all(dim, keepdim)
}

#' allclose
#'
#' similiar to [all.equal()]
#'
#' @param other tensor to comparte
#' @param rtol tolerance
#' @param atol tolerance
#' @param equal_nan compare nans?
#'
#' @examples
#' x <- tensor(c(1,2,3,4,5))
#' y <- tensor(1:5 + 1e-6)
#' allclose(x, y)
#' @export
allclose.tensor <- function(x, other, rtol = 1e-05, atol = 1e-08, equal_nan = FALSE) {
  x$allclose(other, rtol, atol, equal_nan)
}

#' any
#'
#' @param x tensor object
#' @param dim if negative (the default) will reduce to a scalar. Otherwise it will
#' return TRUE if all elements in each row of the tensor in the given dimension
#' `dim` are TRUE, FALSE otherwise.
#' @param keepdim If keepdim is TRUE, the output tensor is of the same size as
#' input except in the dimension dim where it is of size 1. Otherwise, dim is
#' squeezed [squeeze()], resulting in the output tensor having 1 fewer
#' dimension than input.
#' @param na.rm won't be used by the function. Only there to be compatible with
#' [all] generic.
#'
#' @examples
#' x <- tensor(array(c(TRUE, FALSE, TRUE, TRUE), dim = c(2, 2)))
#' any(x)
#' any(x, dim = 0)
#' any(x, dim = 1, keepdim = FALSE)
#' @export
any.tensor <- function(x, dim = -1, keepdim = FALSE, na.rm = FALSE) {
  if (na.rm) warning("tensor's don't use the na.rm argument!")
  x$any(dim, keepdim)
}

#' argmax
#'
#' @param x tensor object
#' @param dim if negative (the default) will reduce to a scalar. Otherwise it will
#' return TRUE if all elements in each row of the tensor in the given dimension
#' `dim` are TRUE, FALSE otherwise.
#' @param keepdim If keepdim is TRUE, the output tensor is of the same size as
#' input except in the dimension dim where it is of size 1. Otherwise, dim is
#' squeezed [squeeze()], resulting in the output tensor having 1 fewer
#' dimension than input.
#' @param na.rm won't be used by the function. Only there to be compatible with
#' [all] generic.
#'
#' @examples
#' x <- tensor(array(runif(8), dim = c(2,2,2)))
#' x
#' argmax(x)
#' argmax(x, dim = 0)
#' argmax(x, dim = 1, keepdim = FALSE)
#' @export
argmax.tensor <- function(x, dim = -1, keepdim = FALSE, na.rm = FALSE) {
  if (na.rm) warning("tensor's don't use the na.rm argument!")
  x$argmax(dim, keepdim)
}

#' argmin
#'
#' @param x tensor object
#' @param dim if negative (the default) will reduce to a scalar. Otherwise it will
#' return TRUE if all elements in each row of the tensor in the given dimension
#' `dim` are TRUE, FALSE otherwise.
#' @param keepdim If keepdim is TRUE, the output tensor is of the same size as
#' input except in the dimension dim where it is of size 1. Otherwise, dim is
#' squeezed [squeeze()], resulting in the output tensor having 1 fewer
#' dimension than input.
#' @param na.rm won't be used by the function. Only there to be compatible with
#' [all] generic.
#'
#' @examples
#' x <- tensor(array(runif(8), dim = c(2,2,2)))
#' x
#' argmin(x)
#' argmin(x, dim = 0)
#' argmin(x, dim = 1, keepdim = FALSE)
#' @export
argmin.tensor <- function(x, dim = -1, keepdim = FALSE, na.rm = FALSE) {
  if (na.rm) warning("tensor's don't use the na.rm argument!")
  x$argmin(dim, keepdim)
}

#' @export
as_strided.tensor <- function(x, size, stride, storage_offset = NULL) {
  x$as_strided(size, stride, storage_offset)
}
