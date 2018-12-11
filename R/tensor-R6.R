`torch::Tensor` <- R6::R6Class(
  "tensor",

  private = list(
    xp = NULL
  ),

  active = list(
    pointer = function(xp) {
      if (missing(xp)) {
        private$xp
      } else {
        stop("Pointer is read-only!")
      }
    },

    data = function(x) {
      if (missing(x)) {
        `torch::Tensor`$dispatch(tensor_data_(self$pointer))
      } else {
        stop("Data is read-only!")
      }
    },

    grad = function(x) {
      if (missing(x)) {
        `torch::Tensor`$dispatch(tensor_grad_(self$pointer))
      } else {
        stop("Grad is read-only!")
      }
    }
  ),

  public = list(

    initialize = function (xp) {
     private$xp <- xp
    },

    print = function (...){
      cat(crayon::silver(glue::glue("{cl}", cl = class(self)[[1]])), "\n")
      tensor_print_(self$pointer)
      invisible(self)
    },

    as_vector = function () {

      a <- as_array_tensor_(self$pointer)

      if (length(a$dim) <= 1L) {
        out <- a$vec
      } else if (length(a$dim) == 2L) {
        out <- t(matrix(a$vec, ncol = a$dim[1], nrow = a$dim[2]))
      } else {
        out <- aperm(array(a$vec, dim = rev(a$dim)), seq(length(a$dim), 1))
      }

      out

    },

    abs = function () {
      `torch::Tensor`$dispatch(tensor_abs_(self$pointer))
    },

    acos = function () {
      `torch::Tensor`$dispatch(tensor_acos_(self$pointer))
    },

    add = function(y) {
      `torch::Tensor`$dispatch(tensor_add_(self$pointer, y$pointer))
    },

    addbmm = function(batch1, batch2, beta = 1, alpha = 1) {
      `torch::Tensor`$dispatch(
        tensor_addbmm_(self$pointer, batch1$pointer, batch2$pointer, beta, alpha)
      )
    },

    addcdiv = function(tensor1, tensor2, value = 1) {
      `torch::Tensor`$dispatch(
        tensor_addcdiv_(self$pointer, tensor1$pointer, tensor2$pointer, value)
      )
    },

    addcmul = function(tensor1, tensor2, value = 1) {
      `torch::Tensor`$dispatch(
        tensor_addcmul_(self$pointer, tensor1$pointer, tensor2$pointer, value)
      )
    },

    addmm = function(mat1, mat2, beta = 1, alpha = 1) {
      `torch::Tensor`$dispatch(
        tensor_addmm_(self$pointer, mat1$pointer, mat2$pointer, beta, alpha)
      )
    },

    addmv = function(mat, vec, beta = 1, alpha = 1) {
      `torch::Tensor`$dispatch(
        tensor_addmv_(self$pointer, mat$pointer, vec$pointer, beta, alpha)
      )
    },

    addr = function(vec1, vec2, beta = 1, alpha = 1) {
      `torch::Tensor`$dispatch(
        tensor_addr_(self$pointer, vec1$pointer, vec2$pointer, beta, alpha)
      )
    },

    all = function(dim = NULL, keepdim = FALSE) {
      `torch::Tensor`$dispatch(tensor_all_(self$pointer, dim, keepdim))
    },

    allclose = function(other, rtol = 1e-05, atol = 1e-08, equal_nan = FALSE) {
      tensor_allclose_(self$pointer, other$pointer, rtol, atol, equal_nan)
    },

    any = function(dim = NULL, keepdim = FALSE) {
      `torch::Tensor`$dispatch(tensor_any_(self$pointer, dim, keepdim))
    },

    argmax = function(dim = NULL, keepdim = FALSE) {
      `torch::Tensor`$dispatch(tensor_argmax_(self$pointer, dim, keepdim))
    },

    argmin = function(dim = NULL, keepdim = FALSE) {
      `torch::Tensor`$dispatch(tensor_argmin_(self$pointer, dim, keepdim))
    },

    as_strided = function(size, stride, storage_offset = NULL) {
      `torch::Tensor`$dispatch(tensor_as_strided_(
        self$pointer, size, stride, storage_offset)
      )
    },

    asin = function(){
      `torch::Tensor`$dispatch(tensor_asin_(self$pointer))
    },

    atan = function() {
      `torch::Tensor`$dispatch(tensor_atan_(self$pointer))
    },

    atan2 = function(other) {
      `torch::Tensor`$dispatch(tensor_atan2_(self$pointer, other$pointer))
    },

    backward = function(gradient = NULL, keep_graph = FALSE, create_graph = FALSE) {
      tensor_backward_(self$pointer, gradient$pointer, keep_graph, create_graph)
    },

    baddbmm = function(batch1, batch2, beta = 1, alpha = 1) {
      `torch::Tensor`$dispatch(
        tensor_baddbmm_(self$pointer, batch1$pointer, batch2$pointer, beta, alpha)
      )
    },

    bernoulli = function(p = NULL) {
      `torch::Tensor`$dispatch(tensor_bernoulli_(self$pointer, p))
    },

    mean = function(dim = NULL, keepdim = NULL, dtype = NULL) {
    `torch::Tensor`$dispatch(tensor_mean_(self$pointer, dim, keepdim, dtype))
    },

    mm = function(mat2) {
      `torch::Tensor`$dispatch(tensor_mm_(self$pointer, mat2$pointer))
    },

    mul = function(other) {
      `torch::Tensor`$dispatch(tensor_mul_(self$pointer, other$pointer))
    },

    pow = function(exponent) {
      `torch::Tensor`$dispatch(tensor_pow_(self$pointer, exponent$pointer))
    },

    sub = function(other, alpha = 1) {
      `torch::Tensor`$dispatch(tensor_sub_(self$pointer, other$pointer, alpha))
    },

    sub_ = function(other, alpha = 1) {
      tensor_sub__(self$pointer, other$pointer, alpha)
      invisible(NULL)
    },

    sum = function(dim = NULL, keepdim = NULL, dtype = NULL) {
      `torch::Tensor`$dispatch(tensor_sum_(self$pointer, dim, keepdim, dtype))
    },

    t = function() {
      `torch::Tensor`$dispatch(tensor_t_(self$pointer))
    },

    to_string = function () {
      tensor_to_string_(self$pointer)
    },

    zero_ = function() {
      tensor_zero__(self$pointer)
      invisible(NULL)
    }

  )

)

external_ptr <- function(class, xp) {
  class$new(xp)
}

`torch::Tensor`$dispatch <- function(xp){
  external_ptr(`torch::Tensor`, xp)
}
