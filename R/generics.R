generic_default <- function(x) {
  stop("Not implemented for ", class(x), ". Did you forget to convert to a torch::Tensor?")
}

#' @export
addbmm <- function(x, ...) {
  UseMethod("addbmm", x)
}

#' @export
addbmm.default <- function(x) {
  generic_default(x)
}

#' @export
addcdiv <- function(x, ...) {
  UseMethod("addcdiv", x)
}

#' @export
addcdiv.default <- function(x) {
  generic_default(x)
}

#' @export
addcmul <- function(x, ...) {
  UseMethod("addcmul", x)
}

#' @export
addcmul.default <- function(x) {
  generic_default(x)
}

#' @export
addmm <- function(x, ...) {
  UseMethod("addmm", x)
}

#' @export
addmm.default <- function(x) {
  generic_default(x)
}
