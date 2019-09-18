#' Reads and get's the declaration file.
#'
#' Uses the options `torchgen.version` or `torchgen.path` to get a
#' Declarations.yaml file. If the `torchgen.path` is specified it's
#' always used instead of the `version`.
#'
#' @export
declarations <- function() {

  version <- getOption("torchgen.version", default = "1.2.0")
  path <- getOption("torchgen.path")

  if (is.null(path)) {
    path <- system.file(
      glue::glue("declaration/Declarations-{version}.yaml"),
      package = "torchgen"
    )
  }

  yaml::read_yaml(
    file = path,
    eval.expr = FALSE,
    handlers = list(
      'bool#yes' = function(x) if (x == "y") x else TRUE,
      'bool#no' = function(x) if (x == "n") x else FALSE,
      int = identity
    )
  )

}

#' Get all tensor methods from Declarations.yaml
#'
#' @export
tensor_methods <- function() {
  declarations() %>%
    purrr::keep(~"Tensor" %in% .x$method_of)
}

#' Creates a single id for a function.
#'
#' Based on all the arguments.
#
hash_arguments <- function(arguments) {
  types <- paste0(map_chr(arguments, ~.x$type), collapse = "")
  names <- paste0(map_chr(arguments, ~.x$name), collapse = "")
  openssl::md5(glue::glue("{types}{names}"))
}
