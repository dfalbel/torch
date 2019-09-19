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

  if (length(method$returns) == 1) {

    dynamic_type <- method$returns[[1]]$dynamic_type

    if (dynamic_type == "Tensor") {

      return("Rcpp::XPtr<torch::Tensor>")

    } else if (dynamic_type %in% c("void", "bool", "double", "int64_t")) {

      return(dynamic_type)

    } else if (dynamic_type == "TensorList") {

      return("Rcpp::List")

    }

  } else if (all(purrr::map_chr(method$returns, ~.x$dynamic_type) == "Tensor")) {

    return("Rcpp::List")

  }

  stop(
    "Don't know how to deal with the return type: ",
    dput(method$returns),
    call. = FALSE
  )

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
