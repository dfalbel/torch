#' @useDynLib torch
NULL

#' Create torch Tensor from R object
#'
#' @param x an R vector, matrix or array.
#'
#' @note it uses the R type when creating the tensor.
#'
#' @examples
#' tensor_from_r(1:10)
#' tensor_from_r(array(runif(8), dim = c(2, 2, 2)))
#' tensor_from_r(matrix(c(TRUE, FALSE), nrow = 3, ncol = 4))
#' @export
tensor_from_r <- function(x) {

  dimension <- dim(x)

  if (is.null(dimension)) {
    dimension <- length(x)
  }

  if (!is.null(dim(x))) {
    x <- aperm(x, perm = seq(length(dim(x)), 1))
  }

  `torch::Tensor`$dispatch(tensor_from_r_(x, dimension))
}

#' Creates a torch tensor.
#'
#' @param x an R object or a torch tensor.
#' @param dtype a string with torch types
#' @param device a device type
#' @param requires_grad boolean indicating if tensor requires grad.
#'
#' @examples
#' x <- tensor(1:10)
#' x
#'
#' y <- tensor(x, dtype = "kDouble")
#' y
#' @export
tensor <- function(x, ...) {
  UseMethod("tensor", x)
}

#' @export
tensor.default <- function(x, dtype = NULL, device = NULL, requires_grad = FALSE) {
  tensor(tensor_from_r(x), dtype, device, requires_grad)
}

#' @export
tensor.tensor <- function(x, dtype = NULL, device = NULL, requires_grad = FALSE) {
  `torch::Tensor`$dispatch(tensor_(x$pointer, dtype, device, requires_grad))
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


#' elementwise addition
#'
#' @param x tensor object
#' @param y tensor object
#'
#' @examples
#' x <- tensor(1)
#' y <- tensor(2)
#' x + y
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
#' @param dim if NULL (the default) will reduce to a scalar. Otherwise it will
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
all.tensor <- function(x, dim = NULL, keepdim = FALSE, na.rm = FALSE) {
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
#' @param dim if NULL (the default) will reduce to a scalar. Otherwise it will
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
any.tensor <- function(x, dim = NULL, keepdim = FALSE, na.rm = FALSE) {
  if (na.rm) warning("tensor's don't use the na.rm argument!")
  x$any(dim, keepdim)
}

#' argmax
#'
#' @param x tensor object
#' @param dim if NULL (the default) will reduce to a scalar. Otherwise it will
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
argmax.tensor <- function(x, dim = NULL, keepdim = FALSE, na.rm = FALSE) {
  if (na.rm) warning("tensor's don't use the na.rm argument!")
  x$argmax(dim, keepdim)
}

#' argmin
#'
#' @param x tensor object
#' @param dim if NULL (the default) will reduce to a scalar. Otherwise it will
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
argmin.tensor <- function(x, dim = NULL, keepdim = FALSE, na.rm = FALSE) {
  if (na.rm) warning("tensor's don't use the na.rm argument!")
  x$argmin(dim, keepdim)
}

#' as_strided
#'
#' TODO: create better docs.
#'
#' @param x tensor object
#' @param size size
#' @param stride stride
#' @param storage_offset (optional) storage_offset
#'
#' @examples
#' x <- tensor(array(runif(8), dim = c(2,2,2)))
#' as_strided(x, 0, 1)
#' @export
as_strided.tensor <- function(x, size, stride, storage_offset = NULL) {
  x$as_strided(size, stride, storage_offset)
}

#' asin
#'
#' Returns a new tensor with the arcsine of the elements of input.
#'
#' @param x tensor object
#' @examples
#' x <- tensor(array(runif(8), dim = c(2,2,2)))
#' asin(x)
#' @export
asin.tensor <- function(x) {
  x$asin()
}

#' atan
#'
#' Returns a new tensor with the arctangent of the elements of input.
#'
#' @param x tensor object
#' @examples
#' x <- tensor(array(runif(8), dim = c(2,2,2)))
#' atan(x)
#' @export
atan.tensor <- function(x) {
  x$atan()
}

#' atan2
#'
#' Returns a new tensor with the arctangent of the elements of input1 and input2.
#'
#' @param x tensor object
#' @param other also a tensor object
#'
#' @examples
#' x <- tensor(array(runif(8), dim = c(2,2,2)))
#' y <- tensor(array(runif(8), dim = c(2,2,2)))
#' atan2(x, y)
#' @export
atan2.tensor <- function(x, other) {
  x$atan2(other)
}

#' baddbmm
#'
#' Performs a batch matrix-matrix product of matrices in batch1 and batch2.
#' x is added to the final result.
#'
#' @param x tensor object
#' @param batch1 the first batch of matrices to be multiplied
#' @param batch2 the second batch of matrices to be multiplied
#' @param beta  multiplier for x (β)
#' @param alpha multiplier for batch1 * batch2 (α)
#'
#' @examples
#' x <- tensor(array(runif(45), dim = c(3, 3, 5)))
#' batch1 <- tensor(array(runif(36), dim = c(3, 3, 4)))
#' batch2 <- tensor(array(runif(60), dim = c(3, 4, 5)))
#' baddbmm(x, batch1, batch2)
#' @export
baddbmm.tensor <- function(x, batch1, batch2, beta = 1, alpha = 1) {
  x$baddbmm(batch1, batch2, beta, alpha)
}

#' bernoulli
#'
#' @param x tensor object
#' @param p probability (if null uses tensor values)
#'
#' @examples
#' x <- tensor(runif(10))
#' bernoulli(x)
#'
#' x <- tensor(rep(0, 10))
#' bernoulli(x)
#' @export
bernoulli.tensor <- function(x, p = NULL) {
  x$bernoulli(p)
}

#' mean
#'
#' @param x tensor object
#' @param dim dimension in which to sum
#' @param keepdim wether to keep or not the dim
#' @param dtype optionaly cast the sum result
#'
#' @examples
#' x <- tensor(1:10)
#' mean(x)
#' @export
mean.tensor <- function(x, dim = NULL, keepdim = NULL, dtype = NULL, na.rm = FALSE) {
  x$mean(dim, keepdim, dtype)
}

#' matrix multiplication
#'
#' Performs matrix multiplication for 2 tensors.
#'
#' @param x tensor object
#' @param mat2 second tensor object
#'
#' @examples
#' x <- tensor(matrix(runif(10), ncol = 5))
#' y <- tensor(matrix(runif(10), nrow = 5))
#' mm(x, y)
#' @export
mm.tensor <- function(x, mat2) {
  x$mm(mat2)
}

#' elementwise multiplciation
#'
#' @param x tensor object
#' @param other tensor object
#'
#' @examples
#' x <- tensor(2)
#' y <- tensor(3)
#' x * y
#' @export
`*.tensor` <- function(x, y) {
  x$mul(y)
}


#' sum
#'
#' @param x tensor object
#' @param dim dimension in which to sum
#' @param keepdim wether to keep or not the dim
#' @param dtype optionaly cast the sum result
#'
#' @examples
#' x <- tensor(1:10)
#' sum(x)
#' @export
sum.tensor <- function(x, dim = NULL, keepdim = NULL, dtype = NULL, na.rm = FALSE) {
  x$sum(dim, keepdim, dtype)
}

#' transpose
#'
#' @param x tensor object
#'
#' @examples
#' x <- tensor(matrix(runif(6), nrow = 3))
#' t(x)
#' @export
t.tensor <- function(x) {
  x$t()
}
