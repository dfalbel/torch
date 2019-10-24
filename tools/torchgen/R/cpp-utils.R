#' Maps a torch type to an Rcpp type.
#'
#' It will return `NA` if we don't know how to map the torch type to an Rcpp type.
#' The user should must how to deal with the `NA`s.
#'
#' @param argument an argument element as in `declarations()[[1]]$arguments[[1]]`
cpp_argument_type <- function(argument) {

  type <- argument$dynamic_type
  is_nullable <- argument$is_nullable

  if (type == "Tensor")
    type <- "Rcpp::XPtr<torch::Tensor>"

  if (type == "Scalar")
    type <- "SEXP"

  if (type == "IntArrayRef")
    type <- "std::vector<std::int64_t>"

  if (type == "ScalarType")
    type <- "std::string"

  if (type == "Device")
    type <- "std::string"

  if (type == "Storage")
    type <- "Rcpp::XPtr<torch::Storage>"

  if (type == "MemoryFormat")
    type <- "Rcpp::XPtr<torch::MemoryFormat>"

  if (type == "TensorList")
    type <- "Rcpp::List"

  if (type == "TensorOptions")
    type <- "Rcpp::XPtr<torch::TensorOptions>"

  if (is_nullable && argument$dynamic_type != "Scalar")
    type <- glue::glue("Rcpp::Nullable<{type}>")

  if (type == "Generator *")
    return(NA_character_) # remove generators from the call

  if (type == "ConstQuantizerPtr")
    return(NA_character_) # remove generators from the call

  return(type)
}

#' Returns how the argument should be used in the cpp code.
#'
#' For example: to use `Rcpp::XPtr<torch::Tensor> x` one must
#' refer to the pointer.
#'
#' It will return `NA` if we don't know how to deal with this argument type.
#' The user must know how to deal with the `NA`s.
#'
#' @inheritParams cpp_argument_type
cpp_use_argument <- function(argument) {

  argument_name <- argument$name

  if (grepl("c10::optional", argument$type) && !argument$dynamic_type %in% c("ScalarType", "Scalar"))
    argument_name <- glue::glue("resolve_null_argument({argument_name})")

  if (argument$dynamic_type == "Scalar" && ! argument$is_nullable)
    argument_name <- glue::glue("scalar_from_r_({argument_name})")

  if (argument$dynamic_type == "Scalar" && argument$is_nullable)
    argument_name <- glue::glue("resolve_null_scalar({argument_name})")

  if (argument$dynamic_type == "ScalarType")
    argument_name <- glue::glue("scalar_type_from_string({argument_name})")

  if (argument$dynamic_type == "TensorList")
    argument_name <- glue::glue("tensor_list_from_r_({argument_name})")

  if (argument$dynamic_type == "Device")
    argument_name <- glue::glue("device_from_string({argument_name})")

  if (argument$dynamic_type == "Storage")
    argument_name <- glue::glue("*{argument_name}")

  if (argument$dynamic_type == "TensorOptions")
    argument_name <- glue::glue("*{argument_name}")

  if (argument$dynamic_type == "MemoryFormat")
    argument_name <- glue::glue("*{argument_name}")

  if (argument$dynamic_type == "Tensor") {

    if (argument$is_nullable)
      argument_name <- glue::glue("Rcpp::as<Rcpp::XPtr<torch::Tensor>>({argument_name})")

    argument_name <- glue::glue("*{argument_name}")
  }

  if (argument$dynamic_type == "Generator *")
    return(NA_character_)

  if (argument$dynamic_type == "MemoryFormat")
    return(NA_character_)

  if (argument$dynamic_type == "ConstQuantizerPtr")
    return(NA_character_)


  argument_name
}

#' Creates a return statement for a torch cpp function.
#'
#' @param returns the returns object as in `declarations()[[1]]$returns`
cpp_return_statement <- function(returns) {

  if (length(returns) == 1) {

    dynamic_type <- returns[[1]]$dynamic_type

    if (dynamic_type == "Tensor")
      return("return make_tensor_ptr(out);")

    if (dynamic_type == "QScheme")
      return("return make_qscheme_ptr(out);")

    if (dynamic_type == "Scalar")
      return("return scalar_to_r_(out);")

    if (dynamic_type == "void")
      return("")

    if (dynamic_type %in% c("double", "bool", "int64_t"))
      return("return out;")

    if (dynamic_type == "TensorList")
      return(
        glue::glue(
          "
           Rcpp::List v;

           for (int i = 0; i < out.size(); ++i) {{
            v.push_back(make_tensor_ptr(out[i]));
           }}

           return v;
          "
        )
      )


  } else { # lenght returns > 1

    if (all(purrr::map_chr(returns, ~.x$dynamic_type) %in% c("Tensor", "TensorList", "int64_t", "double"))) {

      elements <- returns %>%
        purrr::set_names(seq_along(returns)) %>%
        purrr::imap_chr(function(.x, .y) {

          .y <- as.integer(.y)

          if (.x$dynamic_type == "Tensor")
            return(glue::glue("make_tensor_ptr(std::get<{.y-1}>(out))"))
          else if (.x$dynamic_type == "TensorList")
            return(glue::glue("tensorlist_to_r(std::get<{.y-1}>(out))"))
          else if (.x$dynamic_type %in% c("int64_t", "double"))
            return(glue::glue("std::get<{.y-1}>(out)"))
          else if (.x$dynamic_type == "Scalar")
            return(glue::glue("scalar_to_r_(std::get<{.y-1}>(out))"))

        }) %>%
        paste(collapse = ", ")

      return(glue::glue("return Rcpp::List::create({elements});"))

    }


  }

  stop("Return type not implemented.")

}

#' Cpp file dependencies
#'
cpp_deps <- function() {
  c(
    '#include "torch_types.h"',
    '#include "utils.hpp"',
    '#include "scalar.hpp"',
    '#include "device.hpp"'
  )
}
