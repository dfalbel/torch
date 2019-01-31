#' Random normal
#'
#' Returns a tensor filled with random numbers
#' from a normal distribution with mean 0 and variance 1
#' (also called the standard normal distribution).
#'
#' The shape of the tensor is defined by the variable argument sizes.
#'
#' @param sizes a sequence of integers defining the shape of the output tensor.
#' @param dtype the desired data type of returned tensor. Default: if `NULL`, infers
#' data type from `x`.
#' @param the desired layout of returned Tensor. Default: 'strided'
#' @param device  the desired device of returned tensor. Default: if `NULL`, uses
#' the current device for the default tensor type (see `tch_set_default_tensor_type()`).
#' device will be the CPU for CPU tensor types and the current CUDA device for
#' CUDA tensor types.
#' @param requires_grad If autograd should record operations on the
#' returned tensor. Default: `FALSE`.
#'
#' @examples
#' tch_randn(c(2,2))
#' tch_randn(c(2,2), dtype = "double")
#'
#' @export
tch_randn <- function(sizes, dtype = NULL, layout = NULL, device = NULL, requires_grad = FALSE) {
  `torch::Tensor`$dispatch(torch_randn_(sizes, dtype, layout, device, requires_grad))
}

#' arange
#'
#' Returns a 1-D tensor of size \code{floor((end - start)/end)} with values from the interval [start, end) taken with common difference step beginning from start.
#' Note that non-integer step is subject to floating point rounding errors when comparing against end; to avoid inconsistency, we advise adding a small epsilon to
#' end in such cases.
#'
#' @param start the starting value for the set of points
#' @param end the ending value for the set of points
#' @param step the gap between each pair of adjacent points
#' @param out (optional) the output tensor
#' @param dtype the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_tensor_type()). If dtype is not given,
#' infer the data type from the other input arguments. If any of start, end, or stop are floating-point, the dtype is inferred to be the default dtype, see
#' get_default_dtype(). Otherwise, the dtype is inferred to be torch.int64.
#' @param layout the desired layout of returned Tensor
#' @param device the desired device of returned tensor. Default: if None, uses the current device for the default tensor type (see torch.set_default_tensor_type()).
#' device will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types
#' @param requires_grad boolean. If autograd should record operations on the returned tensor
#'
#' @examples
#' tch_arange(5)
#' tch_arange(1, 4)
#' tch_arange(1, 2.5, 0.5)
#'
#' @export
#'
tch_arange <- function(start = 0, end = NULL, step = 1, out = NULL, dtype = NULL, layout = NULL, device = NULL, requires_grad = FALSE) {
  # this is necessary to make the call tch_arange(2) works because the first argument is start instead of end.
  if(is.null(end)) {
    end <- start
    start <- 0
  }

  `torch::Tensor`$dispatch(torch_arange_(start, end, step, dtype, layout, device, requires_grad))
}


#' Empty tensor
#'
#' Returns a tensor filled with uninitialized data.
#'
#' The shape of the tensor is defined by the variable argument sizes.
#'
#' @param sizes a sequence of integers defining the shape of the output tensor.
#' @param dtype the desired data type of returned tensor. Default: if `NULL`, infers
#' data type from `x`.
#' @param the desired layout of returned Tensor. Default: 'strided'
#' @param device  the desired device of returned tensor. Default: if `NULL`, uses
#' the current device for the default tensor type (see `tch_set_default_tensor_type()`).
#' device will be the CPU for CPU tensor types and the current CUDA device for
#' CUDA tensor types.
#' @param requires_grad If autograd should record operations on the
#' returned tensor. Default: `FALSE`.
#'
#' @examples
#' tch_empty(c(2, 2))
#'
#' @export
tch_empty <- function(sizes, dtype = NULL, layout = NULL, device = NULL, requires_grad = FALSE) {
  `torch::Tensor`$dispatch(torch_empty_(sizes, dtype, layout, device, requires_grad))
}

#' Eye matrix (identity matrix)
#'
#' Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
#'
#' @param n integer. The number of rows.
#' @param m (optional) integer. The number of columns with default being n.
#' @param dtype the desired data type of returned tensor. Default: if `NULL`, infers
#' data type from `x`.
#' @param the desired layout of returned Tensor. Default: 'strided'
#' @param device  the desired device of returned tensor. Default: if `NULL`, uses
#' the current device for the default tensor type (see `tch_set_default_tensor_type()`).
#' device will be the CPU for CPU tensor types and the current CUDA device for
#' CUDA tensor types.
#' @param requires_grad If autograd should record operations on the
#' returned tensor. Default: `FALSE`.
#'
#' @examples
#' tch_eye(3)
#' tch_eye(2, 4)
#'
#'
#' @export
tch_eye <- function(n, m = n, dtype = NULL, layout = NULL, device = NULL, requires_grad = FALSE) {
  `torch::Tensor`$dispatch(torch_eye_(n, m, dtype, layout, device, requires_grad))
}


#' Full tensor (one value filled tensor)
#'
#' Returns a tensor of given size filled with fill_value.
#'
#' @param size a sequence of integers defining the shape of the output tensor.
#' @param fill_value the number to fill the output tensor with.
#' @param dtype the desired data type of returned tensor. Default: if `NULL`, infers
#' data type from `x`.
#' @param the desired layout of returned Tensor. Default: 'strided'
#' @param device  the desired device of returned tensor. Default: if `NULL`, uses
#' the current device for the default tensor type (see `tch_set_default_tensor_type()`).
#' device will be the CPU for CPU tensor types and the current CUDA device for
#' CUDA tensor types.
#' @param requires_grad If autograd should record operations on the
#' returned tensor. Default: `FALSE`.
#'
#' @examples
#' tch_full(c(2, 3), 3.141592)
#' tch_full(c(2, 3, 4), 0)
#'
#' @export
tch_full <- function(size, fill_value, dtype = NULL, layout = NULL, device = NULL, requires_grad = FALSE) {
  `torch::Tensor`$dispatch(torch_full_(size, fill_value, dtype, layout, device, requires_grad))
}


#' Linear spaced tensor
#'
#' Returns a one-dimensional tensor of steps equally spaced points between start and end.
#'
#' The output tensor is 1-D of size steps.
#'
#' @param start the starting value for the set of points.
#' @param end the ending value for the set of points.
#' @param steps number of points to sample between start and end. Default: 100.
#' @param fill_value the number to fill the output tensor with.
#' @param dtype the desired data type of returned tensor. Default: if `NULL`, infers
#' data type from `x`.
#' @param the desired layout of returned Tensor. Default: 'strided'
#' @param device  the desired device of returned tensor. Default: if `NULL`, uses
#' the current device for the default tensor type (see `tch_set_default_tensor_type()`).
#' device will be the CPU for CPU tensor types and the current CUDA device for
#' CUDA tensor types.
#' @param requires_grad If autograd should record operations on the
#' returned tensor. Default: `FALSE`.
#'
#' @examples
#' tch_linspace(3, 10, steps = 5)
#' tch_linspace(-10, 10, steps = 5)
#' tch_linspace(start = -10, end = 10, steps = 5)
#' tch_linspace(0, 1)
#'
#' @export
tch_linspace <- function(start, end, steps = 100, dtype = NULL, layout = NULL, device = NULL, requires_grad = FALSE) {
  `torch::Tensor`$dispatch(torch_linspace_(start, end, steps, dtype, layout, device, requires_grad))
}


#' Loglinear spaced tensor
#'
#' Returns a one-dimensional tensor of steps points logarithmically spaced between 10^start and 10^end.
#'
#' The output tensor is 1-D of size steps.
#'
#' @param start the starting value for the set of points.
#' @param end the ending value for the set of points.
#' @param steps number of points to sample between start and end. Default: 100.
#' @param fill_value the number to fill the output tensor with.
#' @param dtype the desired data type of returned tensor. Default: if `NULL`, infers
#' data type from `x`.
#' @param the desired layout of returned Tensor. Default: 'strided'
#' @param device  the desired device of returned tensor. Default: if `NULL`, uses
#' the current device for the default tensor type (see `tch_set_default_tensor_type()`).
#' device will be the CPU for CPU tensor types and the current CUDA device for
#' CUDA tensor types.
#' @param requires_grad If autograd should record operations on the
#' returned tensor. Default: `FALSE`.
#'
#' @examples
#' tch_linspace(3, 10, steps = 5)
#' tch_linspace(-10, 10, steps = 5)
#' tch_linspace(start = -10, end = 10, steps = 5)
#' tch_linspace(0, 1)
#'
#' @export
tch_logspace <- function(start, end, steps = 100, dtype = NULL, layout = NULL, device = NULL, requires_grad = FALSE) {
  `torch::Tensor`$dispatch(torch_logspace_(start, end, steps, dtype, layout, device, requires_grad))
}


#' One filled tensor
#'
#' Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument sizes.
#'
#' @param sizes a sequence of integers defining the shape of the output tensor.
#' @param dtype the desired data type of returned tensor. Default: if `NULL`, infers
#' data type from `x`.
#' @param the desired layout of returned Tensor. Default: 'strided'
#' @param device  the desired device of returned tensor. Default: if `NULL`, uses
#' the current device for the default tensor type (see `tch_set_default_tensor_type()`).
#' device will be the CPU for CPU tensor types and the current CUDA device for
#' CUDA tensor types.
#' @param requires_grad If autograd should record operations on the
#' returned tensor. Default: `FALSE`.
#'
#' @examples
#' tch_ones(c(2, 4))
#' tch_ones(5)
#'
#' @export
tch_ones <- function(sizes, dtype = NULL, layout = NULL, device = NULL, requires_grad = FALSE) {
  `torch::Tensor`$dispatch(torch_ones_(sizes, dtype, layout, device, requires_grad))
}


#' Random uniform
#'
#' Returns a tensor filled with random numbers from a uniform distribution on the interval [0, 1).
#'
#' The shape of the tensor is defined by the variable argument sizes.
#'
#' @param sizes a sequence of integers defining the shape of the output tensor.
#' @param dtype the desired data type of returned tensor. Default: if `NULL`, infers
#' data type from `x`.
#' @param the desired layout of returned Tensor. Default: 'strided'
#' @param device  the desired device of returned tensor. Default: if `NULL`, uses
#' the current device for the default tensor type (see `tch_set_default_tensor_type()`).
#' device will be the CPU for CPU tensor types and the current CUDA device for
#' CUDA tensor types.
#' @param requires_grad If autograd should record operations on the
#' returned tensor. Default: `FALSE`.
#'
#' @examples
#' tch_rand(c(2, 2))
#' tch_rand(c(2, 2), dtype = "double")
#'
#' @export
tch_rand <- function(sizes, dtype = NULL, layout = NULL, device = NULL, requires_grad = FALSE) {
  `torch::Tensor`$dispatch(torch_rand_(sizes, dtype, layout, device, requires_grad))
}


#' Random discrete uniform
#'
#' Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
#'
#' The shape of the tensor is defined by the variable argument size.
#'
#' @param low (optional) Lowest integer to be drawn from the distribution. Default: 0.
#' @param hogh One above the highest integer to be drawn from the distribution.
#' @param sizes a sequence of integers defining the shape of the output tensor.
#' @param dtype the desired data type of returned tensor. Default: if `NULL`, infers
#' data type from `x`.
#' @param the desired layout of returned Tensor. Default: 'strided'
#' @param device  the desired device of returned tensor. Default: if `NULL`, uses
#' the current device for the default tensor type (see `tch_set_default_tensor_type()`).
#' device will be the CPU for CPU tensor types and the current CUDA device for
#' CUDA tensor types.
#' @param requires_grad If autograd should record operations on the
#' returned tensor. Default: `FALSE`.
#'
#' @examples
#' tch_randint(3, 5, 3)
#' tch_randint(10, c(2, 2))
#' tch_randint(3, 10, c(2, 2))
#'
#' @export
tch_randint <- function(low = 0, high = NULL, sizes = NULL, dtype = NULL, layout = NULL, device = NULL, requires_grad = FALSE) {
  # this is necessary to make the call tch_randint(10, c(2, 2))
  # works because the first argument is low instead of high.
  if(is.null(sizes)) {
    sizes <- high
    high <- low
    low <- 0
  }

  `torch::Tensor`$dispatch(torch_randint_(low, high, sizes, dtype, layout, device, requires_grad))
}

#' Random permutation
#'
#' Returns a random permutation of integers from 0 to n - 1.
#'
#' The shape of the tensor is defined by the variable argument sizes.
#'
#' @param n the upper bound (exclusive).
#' @param dtype the desired data type of returned tensor. Default: if `NULL`, infers
#' data type from `x`.
#' @param the desired layout of returned Tensor. Default: 'strided'
#' @param device  the desired device of returned tensor. Default: if `NULL`, uses
#' the current device for the default tensor type (see `tch_set_default_tensor_type()`).
#' device will be the CPU for CPU tensor types and the current CUDA device for
#' CUDA tensor types.
#' @param requires_grad If autograd should record operations on the
#' returned tensor. Default: `FALSE`.
#'
#' @examples
#' tch_randperm(4)
#'
#' @export
tch_randperm <- function(n, dtype = NULL, layout = NULL, device = NULL, requires_grad = FALSE) {
  `torch::Tensor`$dispatch(torch_randperm_(n, dtype, layout, device, requires_grad))
}


#' Zero filled tensor
#'
#' Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument sizes.
#'
#' @param sizes a sequence of integers defining the shape of the output tensor.
#' @param dtype the desired data type of returned tensor. Default: if `NULL`, infers
#' data type from `x`.
#' @param the desired layout of returned Tensor. Default: 'strided'
#' @param device  the desired device of returned tensor. Default: if `NULL`, uses
#' the current device for the default tensor type (see `tch_set_default_tensor_type()`).
#' device will be the CPU for CPU tensor types and the current CUDA device for
#' CUDA tensor types.
#' @param requires_grad If autograd should record operations on the
#' returned tensor. Default: `FALSE`.
#'
#' @examples
#' tch_zeros(c(2, 4))
#' tch_zeros(5)
#'
#' @export
tch_zeros <- function(sizes, dtype = NULL, layout = NULL, device = NULL, requires_grad = FALSE) {
  `torch::Tensor`$dispatch(torch_zeros_(sizes, dtype, layout, device, requires_grad))
}
