# Functions related to the cpp Tensor methods generation.

#' Creates the cpp code for a method.
#'
#' @param method a single element of the [declarations()] list.
#'
method_cpp_code <- function(method) {
  glue::glue("
     {method_cpp_return_type(method)} {method_cpp_name(method)} ({method_cpp_signature(method)}) {{
       {method_cpp_body(method)}
     }}
  ")
}

#' Gets the return type of the cpp function.
#'
#' @inheritParams method_cpp_code
method_cpp_return_type <- function(method) {

}

#' Makes the name of the cpp function.
#'
#' @inheritParams method_cpp_code
method_cpp_name <- function(method) {

}

#' Makes the signature of the cpp function.
#'
#' @inheritParams method_cpp_code
method_cpp_signature <- function(method) {

}

#' Makes the body of the cpp function.
#'
#' @inheritParams method_cpp_code
method_cpp_body <- function(method) {

}
