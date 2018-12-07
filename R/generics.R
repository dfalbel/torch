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

#' @export
addmv <- function(x, ...) {
  UseMethod("addmv", x)
}

#' @export
addmv.default <- function(x) {
  generic_default(x)
}

#' @export
addr <- function(x, ...) {
  UseMethod("addr", x)
}

#' @export
addr.default <- function(x) {
  generic_default(x)
}

#' @export
allclose <- function(x, ...) {
  UseMethod("allclose", x)
}

#' @export
allclose.default <- function(x) {
  generic_default(x)
}

#' @export
argmax <- function(x, ...) {
  UseMethod("argmax", x)
}

#' @export
argmax.default <- function(x) {
  generic_default(x)
}

#' @export
argmin <- function(x, ...) {
  UseMethod("argmin", x)
}

#' @export
argmin.default <- function(x) {
  generic_default(x)
}

#' @export
as_strided <- function(x, ...) {
  UseMethod("as_strided", x)
}

#' @export
as_strided.default <- function(x) {
  generic_default(x)
}

#' @export
atan2 <- function(x, ...) {
  UseMethod("atan2", x)
}

#' @export
atan2.default <- function(y, x) {
  base::atan2(y, x)
}

#' @export
baddbmm <- function(x, ...) {
  UseMethod("baddbmm", x)
}

#' @export
baddbmm.default <- function(y, x) {
  base::baddbmm(y, x)
}

#' @export
bernoulli <- function(x, ...) {
  UseMethod("bernoulli", x)
}

#' @export
bernoulli.default <- function(y, x) {
  generic_default(x)
}

#' @export
mm <- function(x, ...) {
  UseMethod("mm", x)
}

#' @export
mm.default <- function(x) {
  generic_default(x)
}

