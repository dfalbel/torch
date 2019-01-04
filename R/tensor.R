#' @useDynLib torch
#' @importFrom Rcpp sourceCpp
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

  `torch::Tensor`$dispatch(tensor_from_r_(x, rev(dimension)))
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
#' @seealso [as.matrix.tensor()]
#' @examples
#' x <- tensor(array(1:8, dim = c(2, 2, 2)))
#' as.array(x)
#' @export
as.array.tensor <- function(x) {
  x$as_vector()
}

#' Casts a 2d tensor to a matrix.
#'
#' @param x tensor object
#' @seealso [as.array.tensor()]
#'
#' @examples
#' x <- tensor(as.matrix(mtcars))
#' as.matrix(x)
#' @export
as.matrix.tensor <- function(x) {
  as.matrix(x$as_vector())
}

#' Abs
#'
#' Returns absolute values of tensor elements.
#'
#' @param x tensor object
#' @examples
#' x <- tensor(c(-1,1))
#' tch_abs(x)
#' @export
tch_abs <- function(x) {
  x$abs()
}

#' Arc-cos
#'
#' @param x tensor object
#' @examples
#' x <- tensor(runif(10))
#' tch_acos(x)
#' @export
tch_acos <- function(x) {
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
tch_addbmm <- function(x, batch1, batch2, beta = 1, alpha = 1) {
  x$addbmm(batch1, batch2, beta, alpha)
}

#' @export
tch_addcdiv <- function(x, tensor1, tensor2, value = 1) {
  x$addcdiv(tensor1, tensor2, value)
}

#' @export
tch_addcmul <- function(x, tensor1, tensor2, value = 1) {
  x$addcmul(tensor1, tensor2, value)
}

#' @export
tch_addmm <- function(x, mat1, mat2, beta = 1, alpha = 1) {
  x$addmm(mat1, mat2, beta, alpha)
}

#' @export
tch_addmv <- function(x, mat, vec, beta = 1, alpha = 1) {
  x$addmv(mat, vec, beta, alpha)
}

#' @export
tch_addr <- function(x, vec1, vec2, beta = 1, alpha = 1) {
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
#' tch_all(x)
#' tch_all(x, dim = 0)
#' tch_all(x, dim = 1, keepdim = FALSE)
#' @export
tch_all <- function(x, dim = NULL, keepdim = FALSE, na.rm = FALSE) {
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
#' tch_allclose(x, y)
#' @export
tch_allclose <- function(x, other, rtol = 1e-05, atol = 1e-08, equal_nan = FALSE) {
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
#' tch_any(x)
#' tch_any(x, dim = 0)
#' tch_any(x, dim = 1, keepdim = FALSE)
#' @export
tch_any <- function(x, dim = NULL, keepdim = FALSE, na.rm = FALSE) {
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
#' tch_argmax(x)
#' tch_argmax(x, dim = 0)
#' tch_argmax(x, dim = 1, keepdim = FALSE)
#' @export
tch_argmax <- function(x, dim = NULL, keepdim = FALSE, na.rm = FALSE) {
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
#' tch_argmin(x)
#' tch_argmin(x, dim = 0)
#' tch_argmin(x, dim = 1, keepdim = FALSE)
#' @export
tch_argmin <- function(x, dim = NULL, keepdim = FALSE, na.rm = FALSE) {
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
#' tch_as_strided(x, 0, 1)
#' @export
tch_as_strided <- function(x, size, stride, storage_offset = NULL) {
  x$as_strided(size, stride, storage_offset)
}

#' asin
#'
#' Returns a new tensor with the arcsine of the elements of input.
#'
#' @param x tensor object
#' @examples
#' x <- tensor(array(runif(8), dim = c(2,2,2)))
#' tch_asin(x)
#' @export
tch_asin <- function(x) {
  x$asin()
}

#' atan
#'
#' Returns a new tensor with the arctangent of the elements of input.
#'
#' @param x tensor object
#' @examples
#' x <- tensor(array(runif(8), dim = c(2,2,2)))
#' tch_atan(x)
#' @export
tch_atan <- function(x) {
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
#' tch_atan2(x, y)
#' @export
tch_atan2 <- function(x, other) {
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
#' @param beta  multiplier for x
#' @param alpha multiplier for batch1 * batch2
#'
#' @examples
#' x <- tensor(array(runif(45), dim = c(3, 3, 5)))
#' batch1 <- tensor(array(runif(36), dim = c(3, 3, 4)))
#' batch2 <- tensor(array(runif(60), dim = c(3, 4, 5)))
#' tch_baddbmm(x, batch1, batch2)
#' @export
tch_baddbmm <- function(x, batch1, batch2, beta = 1, alpha = 1) {
  x$baddbmm(batch1, batch2, beta, alpha)
}

#' bernoulli
#'
#' @param x tensor object
#' @param p probability (if null uses tensor values)
#'
#' @examples
#' x <- tensor(runif(10))
#' tch_bernoulli(x)
#'
#' x <- tensor(rep(0, 10))
#' tch_bernoulli(x)
#' @export
tch_bernoulli <- function(x, p = NULL) {
  x$bernoulli(p)
}

#' bincount
#'
#' Count the frequency of each value in an array of non-negative ints.
#'
#' @param x tensor object
#' @param weights optional, weight for each value in the input tensor. Should be of same size as input tensor.
#' @param minlength optional, minimum number of bins. Should be non-negative.
#' @examples
#' x <- tensor(sample(0:9, 50, replace = TRUE))
#' tch_bincount(x)
#' @export
tch_bincount <- function(x, weights = NULL, minlength = 0) {
  x$bincount(weights, minlength)
}

#' bmm
#'
#' Performs a batch matrix-matrix product of matrices stored in batch1 and
#' batch2.
#'
#' @examples
#' x <- tensor(array(runif(120), dim = c(10, 3, 4)))
#' y <- tensor(array(runif(200), dim = c(10, 4, 5)))
#' tch_bmm(x, y)
#' @export
tch_bmm <- function(x, mat2) {
  x$bmm(mat2)
}

#' Batch LU factorization
#'
#' Returns a tuple containing the LU factorization and pivots.
#' Pivoting is done if pivot is set.
#'
#' @param x tensor object
#' @param pivot controls whether pivoting is done
#' @examples
#' x <- tensor(array(runif(18), dim = c(2, 3, 3)))
#' tch_btrifact(x)
#' @export
tch_btrifact <- function(x, pivot = TRUE) {
  x$btrifact(pivot)
}

#' Batch LU solve
#'
#' Returns the LU solve of the linear system $Ax = b$.
#'
#' @param x tensor object
#' @param LU_data the pivoted LU factorization of A from [tch_btrifact()]
#' @param LU_pivots the pivots of the LU factorization
#'
#' @examples
#' A <- tensor(array(runif(18), dim = c(2,3,3)))
#' b <- tensor(matrix(runif(6), ncol = 3))
#' A_LU <- tch_btrifact(A)
#' tch_btrisolve(b, A_LU[[1]], A_LU[[2]])
#' @export
tch_btrisolve <- function(x, LU_data, LU_pivots) {
  x$btrisolve(LU_data, LU_pivots)
}

#' Ceil
#'
#' Returns a new tensor with the ceil of the elements of input,
#' the smallest integer greater than or equal to each element.
#'
#' @param x tensor object
#' @examples
#' x <- tensor(runif(10))
#' x
#' tch_ceil(x)
#' @export
tch_ceil <- function(x) {
  x$ceil()
}

#' Chunk
#'
#' Splits a tensor into a specific number of chunks.
#' Last chunk will be smaller if the tensor size along the given dimension
#' `dim` is not divisible by chunks.
#'
#' @param x tensor object
#' @param chunks number of chunks to return
#' @param dim dimension along which to split the tensor
#'
#' @examples
#' x <- tensor(array(runif(100), dim = c(4, 5, 5)))
#' tch_chunk(x, 2, 0)
#' @export
tch_chunk <- function(x, chunks, dim) {
  x$chunk(chunks, dim)
}

#' Clamp
#'
#' Clamp all elements in x into the range min, max and return a
#' resulting tensor.
#'
#' @param x tensor object
#' @param min lower-bound of the range to be clamped to
#' @param max upper-bound of the range to be clamped to
#' @examples
#' x <- tensor(1:10)
#' tch_clamp(x, 5, 7)
#' @export
tch_clamp <- function(x, min, max) {
  x$clamp(min, max)
}

#' Clamp max
#'
#' Clamp all elements in x into the range -Inf, max and return a
#' resulting tensor.
#'
#' @param x tensor object
#' @param max upper-bound of the range to be clamped to
#' @examples
#' x <- tensor(1:10)
#' tch_clamp_max(x, 5)
#' @export
tch_clamp_max <- function(x, max) {
  x$clamp_max(max)
}

#' Clamp min
#'
#' Clamp all elements in x into the range min, Inf and return a
#' resulting tensor.
#'
#' @param x tensor object
#' @param min lower-bound of the range to be clamped to
#' @examples
#' x <- tensor(1:10)
#' tch_clamp_min(x, 5)
#' @export
tch_clamp_min <- function(x, min) {
  x$clamp_min(min)
}

#' Cosine
#'
#' Returns a new tensor with the cosine of the elements of input
#'
#' @param x tensor object
#' @examples
#' tch_cos(tensor(pi))
#' @export
tch_cos <- function(x) {
  x$cos()
}

#' Hyperbolic Cosine
#'
#' Returns a new tensor with the hyperbolic cosine of the elements of input.
#' @param x tensor object
#' @examples
#' tch_cosh(tensor(pi))
#' @export
tch_cosh <- function(x) {
  x$cosh()
}

#' Cross
#'
#' Returns the cross product of vectors in dimension `dim` of `x` and `other`.
#' `x` and `other` must have the same size, and the size of their `dim` dimension
#' should be 3.
#'
#' If dim is not given, it defaults to the first dimension found with the size 3.
#'
#' @param x tensor object
#' @param other the second input tensor
#' @param dim the dimension to take the cross-product in.
#'
#' @examples
#'
#' a <- tensor(matrix(runif(12), ncol = 3))
#' b <- tensor(matrix(runif(12), ncol = 3))
#' tch_cross(a, b, dim=1)
#' tch_cross(a, b)
#' @export
tch_cross <- function(x, other, dim = -1) {
  x$cross(other, dim)
}

#' Cumprod
#'
#' Returns the cumulative product of elements of input in the dimension dim.
#'
#' @param x tensor object
#' @param dim the dimension to do the operation over
#'
#' @examples
#' x <- tensor(1:10)
#' tch_cumprod(x, dim = 0)
#' @export
tch_cumprod <- function(x, dim) {
  x$cumprod(dim)
}

#' Cumsum
#'
#' Returns the cumulative sum of elements of input in the dimension dim.
#'
#' @param x tensor object
#' @param dim the dimension to do the operation over
#'
#' @examples
#' x <- tensor(1:10)
#' tch_cumsum(x, dim = 0)
#' @export
tch_cumsum <- function(x, dim) {
  x$cumsum(dim)
}

#' Determinant
#'
#' Calculates determinant of a 2D square tensor.
#'
#' @param x tensor object
#' @examples
#' x <- tensor(matrix(runif(36), ncol = 6))
#' tch_det(x)
#' @export
tch_det <- function(x) {
  x$det()
}

#' Diag
#'
#' If input is a vector (1-D tensor), then returns a 2-D square tensor with the
#' elements of input as the diagonal.
#'
#' If input is a matrix (2-D tensor), then returns a 1-D tensor with the
#' diagonal elements of input.
#'
#' * If diagonal = 0, it is the main diagonal.
#' * If diagonal > 0, it is above the main diagonal.
#' * If diagonal < 0, it is below the main diagonal.
#'
#' @param x tensor object
#' @param diagonal the diagonal to consider
#'
#' @seealso [tch_diagflat()] [tch_diagonal()]
#'
#' @examples
#' tch_diag(tensor(1:10))
#' tch_diag(tensor(matrix(4, 2, 2)))
#' @export
tch_diag <- function(x, diagonal = 0) {
  x$diag(diagonal)
}

#' Diagflat
#'
#' If input is a vector (1-D tensor), then returns a 2-D square tensor with the
#' elements of input as the diagonal.
#' If input is a tensor with more than one dimension, then returns a 2-D tensor
#' with diagonal elements equal to a flattened input.
#' The argument offset controls which diagonal to consider:
#'
#' * If offset = 0, it is the main diagonal.
#' * If offset > 0, it is above the main diagonal.
#' * If offset < 0, it is below the main diagonal.
#'
#' @param x tensor object
#' @param offset the offset to consider
#'
#' @seealso [tch_diag()] [tch_diagonal()]
#'
#' @examples
#' tch_diagflat(tensor(1:10))
#' @export
tch_diagflat <- function(x, offset = 0) {
  x$diagflat(offset)
}


#' Diagonal
#'
#' Returns a partial view of input with the its diagonal elements with respect
#' to dim1 and dim2 appended as a dimension at the end of the shape.
#'
#' The argument offset controls which diagonal to consider:
#'
#' If offset = 0, it is the main diagonal.
#' If offset > 0, it is above the main diagonal.
#' If offset < 0, it is below the main diagonal.
#'
#' @param x tensor object
#' @param offset which diagonal to consider
#' @param dim1 first dimension with respect to which to take diagonal
#' @param dim2 second dimension with respect to which to take diagonal
#'
#' @seealso [tch_diag()] [tch_diagonal()]
#'
#' @examples
#' tch_diagonal(tensor(matrix(1:4, 2, 2)))
#' @export
tch_diagonal <- function(x, offset = 0, dim1 = 0, dim2 = 1) {
  x$diagonal(offset, dim1, dim2)
}

#' Digamma
#'
#' Computes the logarithmic derivative of the gamma function on input.
#'
#' @param x tensor object
#'
#' @examples
#' tch_digamma(tensor(c(1, 0.5)))
#' @export
tch_digamma <- function(x){
  x$digamma()
}

#' Distance
#'
#' Returns the p-norm of (input - other)
#' The shapes of input and other must be broadcastable.
#'
#' @param x tensor object
#' @param other the Right-hand-side input tensor
#' @param p the norm to be computed
#'
#' @examples
#' x <- tensor(as.numeric(1:10))
#' y <- tensor(as.numeric(10:1))
#' tch_dist(x, y)
#' @export
tch_dist <- function(x, other, p = 2) {
  x$dist(other, p)
}

#' Div
#'
#' Divides each element of the input input with the scalar value and returns a
#' new resulting tensor or,
#'
#' Each element of the tensor input is divided by each element of the tensor other.
#' The resulting tensor is returned. The shapes of input and other must be broadcastable.
#'
#' @param x tensor object
#' @param other scalar or tensor to divide by.
#'
#' @examples
#' tensor(1:10)/2
#' tensor(1:10)/tensor(10:1)
#' @export
`/.tensor` <- function(x, other) {
  x$div(other)
}

#' Gels
#'
#' Computes the solution to the least squares and least norm problems for a full
#' rank matrix A of size (m,n) and a matrix B of size (m,k).
#'
#' @param x tensor object
#' @param A the m,b by nn matrix A
#' @examples
#' A <- tensor(matrix(runif(100), ncol = 10))
#' y <- tensor(matrix(runif(10), ncol = 1))
#' tch_gels(y, A)
#' @export
tch_gels <- function(x, A) {
  x$gels(A)
}

#' mean
#'
#' @param x tensor object
#' @param dim dimension in which to sum
#' @param keepdim wether to keep or not the dim
#' @param dtype optionaly cast the sum result
#'
#' @examples
#' x <- tensor(runif(100))
#' tch_mean(x)
#' @export
tch_mean <- function(x, dim = NULL, keepdim = NULL, dtype = NULL, na.rm = FALSE) {
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
#' tch_mm(x, y)
#' @export
tch_mm <- function(x, mat2) {
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

#' permute
#'
#' Permute the dimensions of this tensor.
#'
#' @param x tensor object
#' @param dims the desired ordering of dimensions (0 based).
#'
#' @examples
#' x <- tensor(array(1:10, dim = c(2,5,1)))
#' x
#' tch_permute(x, c(2,1,0))
#' @export
tch_permute <- function(x, dims) {
  x$permute(dims)
}

#' pow
#'
#' @param x tensor object
#' @param y exponent (a tensor)
#' @examples
#' x <- tensor(c(1,2,3,4))
#' y <- tensor(2)
#' x^y
#' @export
`^.tensor` <- function(x, y) {
  x$pow(y)
}

#' QR decomposition
#'
#' Computes the QR decomposition of a matrix input, and returns matrices Q and R
#' such that $input = QR$, with QQ being an orthogonal matrix and RR being an
#' upper triangular matrix.
#'
#' This returns the thin (reduced) QR factorization.
#'
#' @param x tensor object
#' @examples
#' x <- tensor(matrix(runif(16), ncol = 4))
#' tch_qr(x)
#' @export
tch_qr <- function(x) {
  x$qr()
}

#' substraction
#'
#' @param x tensor object
#' @param y tensor to substract
#' @examples
#' x <- tensor(c(1,2,3,4))
#' y <- tensor(2)
#' x - y
#' @export
`-.tensor` <- function(x, y) {
  x$sub(y)
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
#' tch_sum(x)
#' @export
tch_sum <- function(x, dim = NULL, keepdim = NULL, dtype = NULL, na.rm = FALSE) {
  x$sum(dim, keepdim, dtype)
}

#' transpose
#'
#' @param x tensor object
#'
#' @examples
#' x <- tensor(matrix(runif(6), nrow = 3))
#' tch_t(x)
#' @export
tch_t <- function(x) {
  x$t()
}
