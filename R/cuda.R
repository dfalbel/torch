#' Checks if cuda is available.
#' 
#' @examples 
#' tch_cuda_is_available()
#'
#' @export
tch_cuda_is_available <- function() {
  cuda_is_available_()
}