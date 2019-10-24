args_to_pointers <- function(args) {
  lapply(args, function(x) {

    if (inherits(x, "torch::Tensor"))
      return(x$pointer)

    x
  })
}
