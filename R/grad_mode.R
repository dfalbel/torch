#' With no grad
#'
#' Temporarily modify gradient recording.
#'
#' @param code code to be executed with no gradient recording.
#'
#' @examples
#' x <- tensor(runif(5), requires_grad = TRUE)
#' with_no_grad({x$sub_(tensor(as.numeric(1:5)))})
#' x
#'
#' @export
with_no_grad <- withr::with_(
  set = function() {
    set_grad_mode(FALSE)
  },
  reset = function(old) {
    set_grad_mode(TRUE)
  }
)
