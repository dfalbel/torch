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

    all = function(dim = -1L, keepdim = FALSE) {
      `torch::Tensor`$dispatch(tensor_all_(self$pointer, dim, keepdim))
    },

    allclose = function(other, rtol = 1e-05, atol = 1e-08, equal_nan = FALSE) {
      tensor_allclose_(self$pointer, other$pointer, rtol, atol, equal_nan)
    },

    any = function(dim = -1L, keepdim = FALSE) {
      `torch::Tensor`$dispatch(tensor_any_(self$pointer, dim, keepdim))
    },

    argmax = function(dim = -1L, keepdim = FALSE) {
      `torch::Tensor`$dispatch(tensor_argmax_(self$pointer, dim, keepdim))
    },

    to_string = function () {
      tensor_to_string_(self$pointer)
    }

  )

)

external_ptr <- function(class, xp) {
  class$new(xp)
}

`torch::Tensor`$dispatch <- function(xp){
  external_ptr(`torch::Tensor`, xp)
}
