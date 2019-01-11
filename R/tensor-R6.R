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

    bincount = function(weights = NULL, minlength = 0) {
      `torch::Tensor`$dispatch(tensor_bincount_(self$pointer, weights$pointer, minlength))
    },

    bmm = function(mat2) {
      `torch::Tensor`$dispatch(tensor_bmm_(self$pointer, mat2$pointer))
    },

    btrifact = function(pivot = TRUE) {
      x <- tensor_btrifact_(self$pointer, pivot)
      list(
        `torch::Tensor`$dispatch(x[[1]]),
        `torch::Tensor`$dispatch(x[[2]])
      )
    },

    btrisolve = function(LU_data, LU_pivots) {
      `torch::Tensor`$dispatch(
        tensor_btrisolve_(self$pointer, LU_data$pointer, LU_pivots$pointer)
      )
    },

    cauchy_ = function(median = 0, sigma = 1) {
      tensor_cauchy__(self$pointer, median, sigma)
      invisible(NULL)
    },

    ceil = function() {
      `torch::Tensor`$dispatch(tensor_ceil_(self$pointer))
    },

    ceil_ = function() {
      tensor_ceil__(self$pointer)
      invisible(NULL)
    },

    chunk = function(chunks, dim) {
      lapply(
        tensor_chunk_(self$pointer, chunks, dim),
        `torch::Tensor`$dispatch
      )
    },

    clamp = function(min, max) {
      `torch::Tensor`$dispatch(tensor_clamp_(self$pointer, min, max))
    },

    clamp_ = function(min, max) {
      tensor_clamp__(self$pointer, min, max)
      invisible(NULL)
    },

    clamp_max = function(max) {
      `torch::Tensor`$dispatch(tensor_clamp_max_(self$pointer, max))
    },

    clamp_max_ = function(max) {
      tensor_clamp_max__(self$pointer, max)
      invisible(NULL)
    },

    clamp_min = function(min) {
      `torch::Tensor`$dispatch(tensor_clamp_min_(self$pointer, min))
    },

    clamp_min_ = function(min) {
      tensor_clamp_min__(self$pointer, min)
      invisible(NULL)
    },

    clone_ = function() {
      # TODO decide if clone_ is the best name for this method.
      `torch::Tensor`$dispatch(tensor_clone_(self$pointer))
    },

    contiguous = function() {
      `torch::Tensor`$dispatch(tensor_contiguous_(self$pointer))
    },

    copy_ = function(src, non_blocking = FALSE) {
      tensor_copy__(self$pointer, src$pointer, non_blocking)
      invisible(NULL)
    },

    cos = function() {
      `torch::Tensor`$dispatch(tensor_cos_(self$pointer))
    },

    cos_ = function() {
      tensor_cos__(self$pointer)
      invisible(NULL)
    },

    cosh = function() {
      `torch::Tensor`$dispatch(tensor_cosh_(self$pointer))
    },

    cosh_ = function() {
      tensor_cosh__(self$pointer)
      invisible(NULL)
    },

    cpu = function() {
      `torch::Tensor`$dispatch(tensor_cpu_(self$pointer))
    },

    cross = function(other, dim = -1) {
      `torch::Tensor`$dispatch(tensor_cross_(self$pointer, other$pointer, dim))
    },

    cuda = function() {
      `torch::Tensor`$dispatch(tensor_cuda_(self$pointer))
    },

    cumprod = function(dim) {
      `torch::Tensor`$dispatch(tensor_cumprod_(self$pointer, dim))
    },

    cumsum = function(dim) {
      `torch::Tensor`$dispatch(tensor_cumsum_(self$pointer, dim))
    },

    det = function() {
      `torch::Tensor`$dispatch(tensor_det_(self$pointer))
    },

    detach = function() {
      `torch::Tensor`$dispatch(tensor_detach_(self$pointer))
    },

    detach_ = function() {
      tensor_detach__(self$pointer)
      invisible(NULL)
    },

    device = function() {
      tensor_device_(self$pointer)
    },

    diag = function(diagonal = 0) {
      `torch::Tensor`$dispatch(tensor_diag_(self$pointer, diagonal))
    },

    diagflat = function(offset = 0) {
      `torch::Tensor`$dispatch(tensor_diagflat_(self$pointer, offset))
    },

    diagonal = function(offset = 0, dim1 = 0, dim2 = 1) {
      `torch::Tensor`$dispatch(tensor_diagonal_(self$pointer, offset, dim1, dim2))
    },

    digamma = function(){
      `torch::Tensor`$dispatch(tensor_digamma_(self$pointer))
    },

    digamma_ = function(){
      tensor_digamma__(self$pointer)
      invisible(NULL)
    },

    dim = function() {
      tensor_dim_(self$pointer)
    },

    dist = function(other, p = 2) {
      `torch::Tensor`$dispatch(tensor_dist_(self$pointer, other$pointer, p))
    },

    div = function(other) {
      if (is(other, "tensor")) {
        `torch::Tensor`$dispatch(tensor_div_tensor_(self$pointer, other$pointer))
      } else {
        `torch::Tensor`$dispatch(tensor_div_scalar_(self$pointer, other))
      }
    },

    div_ = function(other) {
      if (is(other, "tensor")) {
        tensor_div_tensor__(self$pointer, other$pointer)
      } else {
        tensor_div_scalar__(self$pointer, other)
      }
      invisible(NULL)
    },

    dot = function(tensor) {
      `torch::Tensor`$dispatch(tensor_dot_(self$pointer, tensor$pointer))
    },

    dtype = function() {
      tensor_dtype_(self$pointer)
    },

    gels = function(A) {
      out <- tensor_gels_(self$pointer, A$pointer)
      lapply(out, `torch::Tensor`$dispatch)
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

    permute = function(dims) {
      `torch::Tensor`$dispatch(tensor_permute_(self$pointer, dims))
    },

    pow = function(exponent) {
      `torch::Tensor`$dispatch(tensor_pow_(self$pointer, exponent$pointer))
    },

    qr = function() {
      out <- tensor_qr_(self$pointer)
      lapply(out, `torch::Tensor`$dispatch)
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
