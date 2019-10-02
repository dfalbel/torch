#' Generates the S4 generic definition
#'
#' @param methods the methods list. something like
#'  `tensor_methods() %>% declaration_with_name("abs")`
#'
generic_s4_code <- function(methods) {

  possible_names <- get_possible_argument_names(methods)

  glue::glue(
    'setGeneric(
      "{method_s4_generic_name(methods)}",
      function({paste(possible_names, collapse =", ")}) standardGeneric("torch_{nm}_")
     )'
  )
}

#' Generaates the definition of an S4 method
#'
#' @inheritParams method_cpp_code
#'
method_s4_code <- function(method) {
  glue::glue(
"
setMethod(
 f = '{method_s4_generic_name(method)}',
 signature = {method_s4_signature(method)},
 definition = function({arguments_string(method)}) {{
  {body_string(method)}
 }})
"
  )
}

#' Get's the name of the generic function corresponding to the method
#'
#' @inheritParams method a single method or a list of methods.
#'
method_s4_generic_name <- function(method) {

  nm <- method[["name"]]

  if (is.null(nm))
    nm <- method[[1]]$name

  glue::glue("torch_{nm}_")
}

#' Returns a tibble of argument names and types of methods.
#' Already discarding argument names with `NA` that wound't be
#' converted.
#'
#' @inheritParams method_cpp_code
#'
method_s4_argument_names_and_types <- function(method) {

  argument_types <- method$arguments %>%
    purrr::map_chr(argument_type_to_r)
  argument_names <- method$arguments %>%
    purrr::map_chr(~.x$name)

  # discard NA types
  nas <- which(is.na(argument_types))
  argument_types <- argument_types[-nas]
  argument_names <- argument_names[-nas]

  tibble::tibble(
   argument_names = argument_names,
   argument_types = argument_types
  )
}

#' Creates the signature as required by S4 methods in R
#' See [methods::setMethod()]'s `signature` argument.
#'
#' @inheritParams method_cpp_code
#'
method_s4_signature <- function(method) {

  arguments <- method_s4_argument_names_and_types(method)

  argument_names <- arguments$names
  argument_types <- arguments$types

  args <- glue::glue('{argument_names} = "{argument_types}"') %>%
    glue::glue_collapse(sep = ", ")

  glue::glue("list({args})")
}

method_s4_impl_signature <- function(method) {

  arguments <- method_s4_argument_names_and_types(method)

  argument_names <- arguments$names
  argument_types <- arguments$types

}

method_s4_impl_body <- function(method) {

}

