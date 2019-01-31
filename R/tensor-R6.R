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
      cat(glue::glue("{cl}", cl = class(self)[[1]]), "\n")
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
      if (is(y, "tensor")) {
        `torch::Tensor`$dispatch(tensor_add_tensor_(self$pointer, y$pointer))
      } else {
        `torch::Tensor`$dispatch(tensor_add_scalar_(self$pointer, y))
      }
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

    sin = function(){
      `torch::Tensor`$dispatch(tensor_sin_(self$pointer))
    },

    sinh = function(){
      `torch::Tensor`$dispatch(tensor_sinh_(self$pointer))
    },

    atan = function() {
      `torch::Tensor`$dispatch(tensor_atan_(self$pointer))
    },

    atan2 = function(other) {
      `torch::Tensor`$dispatch(tensor_atan2_(self$pointer, other$pointer))
    },

    tan = function() {
      `torch::Tensor`$dispatch(tensor_tan_(self$pointer))
    },

    tanh = function() {
      `torch::Tensor`$dispatch(tensor_tanh_(self$pointer))
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

    eig = function(eigenvectors = FALSE) {
      out <- tensor_eig_(self$pointer, eigenvectors)
      lapply(out, `torch::Tensor`$dispatch)
    },

    eq = function(other) {
      if (is(other, "tensor")) {
        `torch::Tensor`$dispatch(tensor_eq_tensor_(self$pointer, other$pointer))
      } else {
        `torch::Tensor`$dispatch(tensor_eq_scalar_(self$pointer, other))
      }
    },

    equal = function(other) {
      tensor_equal_(self$pointer, other$pointer)
    },

    erf = function() {
      `torch::Tensor`$dispatch(tensor_erf_(self$pointer))
    },

    erf_ = function() {
      tensor_erf__(self$pointer)
      invisible(self)
    },

    erfc = function() {
      `torch::Tensor`$dispatch(tensor_erfc_(self$pointer))
    },

    erfc_ = function() {
      tensor_erfc__(self$pointer)
      invisible(self)
    },

    erfinv = function() {
      `torch::Tensor`$dispatch(tensor_erfinv_(self$pointer))
    },

    erfinv_ = function() {
      tensor_erfinv__(self$pointer)
      invisible(self)
    },

    exp = function() {
      `torch::Tensor`$dispatch(tensor_exp_(self$pointer))
    },

    exp_ = function() {
      tensor_exp__(self$pointer)
      invisible(self)
    },

    expand = function(sizes) {
      `torch::Tensor`$dispatch(tensor_expand_(self$pointer, sizes))
    },

    expand_as = function(other) {
      `torch::Tensor`$dispatch(tensor_expand_as_(self$pointer, other$pointer))
    },

    expm1 = function() {
      `torch::Tensor`$dispatch(tensor_expm1_(self$pointer))
    },

    expm1_ = function() {
      tensor_expm1__(self$pointer)
      invisible(self)
    },

    fill_ = function(value) {
      if (is(value, "tensor")) {
        tensor_fill_tensor__(self$pointer, value$pointer)
      } else {
        tensor_fill_scalar__(self$pointer, value)
      }
      invisible(self)
    },

    flatten = function(start_dim = 0, end_dim = -1) {
      `torch::Tensor`$dispatch(tensor_flatten_(self$pointer, start_dim, end_dim))
    },

    flip = function(dims) {
      `torch::Tensor`$dispatch(tensor_flip_(self$pointer, dims))
    },

    float = function() {
      `torch::Tensor`$dispatch(tensor_float_(self$pointer))
    },

    floor = function() {
      `torch::Tensor`$dispatch(tensor_floor_(self$pointer))
    },

    floor_ = function() {
      tensor_floor__(self$pointer)
      invisible(self)
    },

    fmod = function(other) {
      if(is(other, "tensor")) {
        `torch::Tensor`$dispatch(tensor_fmod_tensor_(self$pointer, other$pointer))
      } else {
        `torch::Tensor`$dispatch(tensor_fmod_scalar_(self$pointer, other))
      }
    },

    fmod_ = function(other) {
      if(is(other, "tensor")) {
        tensor_fmod_tensor__(self$pointer, other$pointer)
      } else {
        tensor_fmod_scalar__(self$pointer, other)
      }
      invisible(self)
    },

    frac = function() {
      `torch::Tensor`$dispatch(tensor_frac_(self$pointer))
    },

    frac_ = function() {
      tensor_frac__(self$pointer)
      invisible(self)
    },

    gather = function(dim, index) {
      `torch::Tensor`$dispatch(tensor_gather_(self$pointer, dim, index$pointer))
    },

    ge = function(other) {
      if (is(other, "tensor")) {
        `torch::Tensor`$dispatch(tensor_ge_tensor_(self$pointer, other$pointer))
      } else {
        `torch::Tensor`$dispatch(tensor_ge_scalar_(self$pointer, other))
      }
    },

    ge_ = function(other) {
      if (is(other, "tensor")) {
        tensor_ge_tensor__(self$pointer, other$pointer)
      } else {
        tensor_ge_scalar__(self$pointer, other)
      }
      invisible(self)
    },

    gels = function(A) {
      out <- tensor_gels_(self$pointer, A$pointer)
      lapply(out, `torch::Tensor`$dispatch)
    },

    log = function() {
      `torch::Tensor`$dispatch(tensor_log_(self$pointer))
    },

    log_ = function() {
      tensor_log__(self$pointer)
      invisible(self)
    },

    log2 = function() {
      `torch::Tensor`$dispatch(tensor_log2_(self$pointer))
    },

    log2_ = function() {
      tensor_log2__(self$pointer)
      invisible(self)
    },

    log10 = function() {
      `torch::Tensor`$dispatch(tensor_log10_(self$pointer))
    },

    log10_ = function() {
      tensor_log10__(self$pointer)
      invisible(self)
    },

    log1p = function() {
      `torch::Tensor`$dispatch(tensor_log1p_(self$pointer))
    },

    log1p_ = function() {
      tensor_log1p__(self$pointer)
      invisible(self)
    },

    logsumexp = function(dim = NULL, keepdim = NULL, dtype = NULL) {
      `torch::Tensor`$dispatch(tensor_logsumexp_(self$pointer, dim, keepdim, dtype))
    },

    max = function(dim = NULL, keepdim = NULL, dtype = NULL) {
      `torch::Tensor`$dispatch(tensor_max_(self$pointer, dim, keepdim, dtype))
    },

    mean = function(dim = NULL, keepdim = NULL, dtype = NULL) {
      `torch::Tensor`$dispatch(tensor_mean_(self$pointer, dim, keepdim, dtype))
    },

    median = function(dim = -1, keepdim = FALSE) {
      out <- tensor_median_(self$pointer, dim, keepdim)
      lapply(out, `torch::Tensor`$dispatch)
    },

    min = function(dim = NULL, keepdim = NULL, dtype = NULL) {
      `torch::Tensor`$dispatch(tensor_min_(self$pointer, dim, keepdim, dtype))
    },

    mode = function(dim = -1, keepdim = FALSE) {
      out <- tensor_mode_(self$pointer, dim, keepdim)
      lapply(out, `torch::Tensor`$dispatch)
    },

    mm = function(mat2) {
      `torch::Tensor`$dispatch(tensor_mm_(self$pointer, mat2$pointer))
    },

    mul = function(other) {
      if (is(other, "tensor")) {
        `torch::Tensor`$dispatch(tensor_mul_tensor_(self$pointer, other$pointer))
      } else {
        `torch::Tensor`$dispatch(tensor_mul_scalar_(self$pointer, other))
      }
    },

    permute = function(dims) {
      `torch::Tensor`$dispatch(tensor_permute_(self$pointer, dims))
    },

    pow = function(exponent) {
      if (is(exponent, "tensor")) {
        `torch::Tensor`$dispatch(tensor_pow_tensor_(self$pointer, exponent$pointer))
      } else {
        `torch::Tensor`$dispatch(tensor_pow_scalar_(self$pointer, exponent))
      }
    },

    prod = function(dim = NULL, keepdim = NULL, dtype = NULL) {
      `torch::Tensor`$dispatch(tensor_prod_(self$pointer, dim, keepdim, dtype))
    },

    qr = function() {
      out <- tensor_qr_(self$pointer)
      lapply(out, `torch::Tensor`$dispatch)
    },

    round = function() {
      `torch::Tensor`$dispatch(tensor_round_(self$pointer))
    },

    rsqrt = function() {
      `torch::Tensor`$dispatch(tensor_rsqrt_(self$pointer))
    },

    rsqrt_ = function() {
      tensor_rsqrt__(self$pointer)
      invisible(NULL)
    },

    sigmoid = function() {
      `torch::Tensor`$dispatch(tensor_sigmoid_(self$pointer))
    },

    sigmoid_ = function() {
      tensor_sigmoid__(self$pointer)
      invisible(NULL)
    },

    std = function(unbiased = TRUE, dim = NULL, keepdim = NULL, dtype = NULL) {
      `torch::Tensor`$dispatch(tensor_std_(self$pointer, unbiased, dim, keepdim, dtype))
    },

    sub = function(other, alpha = 1) {
      if (is(other, "tensor")) {
        `torch::Tensor`$dispatch(tensor_sub_tensor_(self$pointer, other$pointer, alpha))
      } else {
        `torch::Tensor`$dispatch(tensor_sub_scalar_(self$pointer, other, alpha))
      }
    },

    sub_ = function(other, alpha = 1) {
      if (is(other, "tensor")) {
        tensor_sub_tensor__(self$pointer, other$pointer, alpha)
      } else {
        tensor_sub_scalar__(self$pointer, other, alpha)
      }
      invisible(NULL)
    },

    sum = function(dim = NULL, keepdim = NULL, dtype = NULL) {
      `torch::Tensor`$dispatch(tensor_sum_(self$pointer, dim, keepdim, dtype))
    },

    t = function() {
      `torch::Tensor`$dispatch(tensor_t_(self$pointer))
    },

    tril = function(diagonal = 0) {
      `torch::Tensor`$dispatch(tensor_tril_(self$pointer, diagonal))
    },

    triu = function(diagonal = 0) {
      `torch::Tensor`$dispatch(tensor_triu_(self$pointer, diagonal))
    },

    to = function(dtype = NULL, device = NULL, requires_grad = FALSE) {
      `torch::Tensor`$dispatch(tensor_to_(self$pointer, dtype, device, requires_grad))
    },

    to_string = function () {
      tensor_to_string_(self$pointer)
    },

    var = function(unbiased = TRUE, dim = NULL, keepdim = NULL, dtype = NULL) {
      `torch::Tensor`$dispatch(tensor_var_(self$pointer, unbiased, dim, keepdim, dtype))
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
