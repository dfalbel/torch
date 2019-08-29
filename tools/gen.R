# This script is used to Generate wrapers for pytorch functions.
# The generated code is created by  splitting methods in the follwing groups:
#
# - Methods that have a single signature
#   - NUllable arguments based in c10::optional can be solved with `resolve_null_argument`
# - Methods that don't have a single signature
#   - Methods that have different signatures for scalars and tensors
#   - Methods that have a single additional argument.
#   - Methods that have a second signature with dim and keepdim args
#   - Other methods -> code is not automatically generated.
#
#

library(purrr)
library(glue)

methods_fname <- "src/methods.cpp"

declarations <- yaml::read_yaml(
  file = "~/Documents/pytorch/build/aten/src/ATen/Declarations.yaml",
  eval.expr = FALSE,
  handlers = list(
    'bool#yes' = function(x) if (x == "y") x else TRUE,
    'bool#no' = function(x) if (x == "n") x else FALSE,
    int = identity
  )
)

tensor_methods <- declarations %>%
  keep(~"Tensor" %in% .x$method_of)

method_names <- unique(map_chr(tensor_methods, ~.x$name))

exceptions <- c("qscheme", "item")

method_names <- method_names[!method_names %in% exceptions]

# finds out if multiple signatures are just because we deal differently with
# tensors vs scalars. eg.: add
case_tensor_vs_scalar <- function(method) {

  arguments <- method %>%
    map(~.x$arguments) %>%
    map_df(
      ~map_df(
        .x,
        ~tibble::tibble(
          name = .x$name %||% NA,
          type = .x$dynamic_type
        )
      ),
      .id = "id"
    )

  summary <- arguments %>%
    dplyr::group_by(name) %>%
    dplyr::summarise(
      n = dplyr::n_distinct(id),
      ntypes = dplyr::n_distinct(type),
      scalar_and_tensor = all(c("Scalar", "Tensor") %in% type) | all(c("double", "Tensor") %in% type)
    )

  # not all arguments have the same name for all signatures
  if (!all(summary$n == summary$n[1]))
    return(FALSE)

  # all arguments expect one have 2 types
  if (!nrow(dplyr::filter(summary, ntypes > 1) == 1))
    return(FALSE)

  # we dont have any scalar and a tensor
  if (!any(summary$scalar_and_tensor))
    return(FALSE)

  TRUE
}

# single scalar argument is different
case_single_additional_argument <- function(method) {

  arguments <- method %>%
    map(~.x$arguments) %>%
    map_df(
      ~map_df(
        .x,
        ~tibble::tibble(
          name = .x$name,
          type = .x$dynamic_type
        )
      ),
      .id = "id"
    )

  summary <- arguments %>%
    dplyr::group_by(name) %>%
    dplyr::summarise(
      n = dplyr::n_distinct(id),
      ntypes = dplyr::n_distinct(type)
    )

  # more than 2 methods
  if (max(summary$n) > 2)
    return(FALSE)

  if (!nrow(dplyr::filter(summary, n == 1)) == 1)
    return(FALSE)

  additional_arg_name <- dplyr::filter(summary, n == 1)$name
  if (!arguments$type[arguments$name == additional_arg_name] == "double")
    return(FALSE)

  TRUE
}

# one of the signatures is have dim and keepdim args.
case_dim_and_keepdim <- function(method) {

  arguments <- method %>%
    map(~.x$arguments) %>%
    map_df(
      ~map_df(
        .x,
        ~tibble::tibble(
          name = .x$name,
          type = .x$dynamic_type
        )
      ),
      .id = "id"
    )

  if (! all(c("dim", "keepdim") %in% arguments$name))
    return(FALSE)

  summary <- arguments %>%
    dplyr::group_by(id) %>%
    dplyr::summarise(
      nargs = dplyr::n(),
      args = list(unique(name))
      )

  if (! nrow(summary) == 2)
    return(FALSE)

  if (! max(summary$nargs) == min(summary$nargs) + 2)
    return(FALSE)

  TRUE
}

case_dim_keepdim_and_other <- function(method) {

  arguments <- method %>%
    map(~.x$arguments) %>%
    map_df(
      ~map_df(
        .x,
        ~tibble::tibble(
          name = .x$name,
          type = .x$dynamic_type
        )
      ),
      .id = "id"
    )

  if (! all(c("dim", "keepdim", "other") %in% arguments$name))
    return(FALSE)

  summary <- arguments %>%
    dplyr::group_by(id) %>%
    dplyr::summarise(
      nargs = dplyr::n(),
      args = list(unique(name))
    )

  if (! nrow(summary) == 3)
    return(FALSE)

  if (! all(c(1,2,3) %in% summary$nargs))
    return(FALSE)

  TRUE
}

argument_string <- function(argument) {

  type <- argument$dynamic_type

  if (type == "Tensor")
    type <- "Rcpp::XPtr<torch::Tensor>"

  if (type == "Scalar")
    type <- "SEXP"

  if (type == "IntArrayRef")
    type <- "std::vector<std::int64_t>"

  if (type == "ScalarType")
    type <- "std::string"

  if (type == "TensorList")
    type <- "Rcpp::List"

  if (argument$is_nullable)
    type <- glue::glue("Rcpp::Nullable<{type}>")

  if (type == "Generator *")
    return(NA_character_) # remove generators from the call

  if (type == "MemoryFormat")
    return(NA_character_) # remove generators from the call

  if (type == "ConstQuantizerPtr")
    return(NA_character_) # remove generators from the call

  glue::glue("{type} {argument$name}")
}

arguments_string <- function(arguments) {
  arguments %>%
    purrr::map_chr(argument_string) %>%
    purrr::discard(is.na) %>%
    paste(collapse = ", ")
}

return_type_string <- function(method) {
  if (length(method$returns) == 1) {
    if (method$returns[[1]]$dynamic_type == "Tensor")
      "Rcpp::XPtr<torch::Tensor>"
    else if (method$returns[[1]]$dynamic_type %in% c("void", "bool", "double", "int64_t"))
      method$returns[[1]]$dynamic_type
    else if (method$returns[[1]]$dynamic_type == "TensorList")
      "Rcpp::List"


  } else if (all(purrr::map_chr(method$returns, ~.x$dynamic_type) == "Tensor")) {
    "Rcpp::List"
  }
}

signature_string <- function(method) {

  return_type <-

  glue::glue("
     {return_type_string(method)} torch_{method$name}_ (
     {arguments_string(method$arguments)}
     )
     ")
}

arguments_call_string <- function(argument) {

  argument_name <- argument$name

  if (argument$dynamic_type == "Scalar")
    argument_name <- glue::glue("scalar_from_r_({argument_name})")

  if (argument$dynamic_type == "ScalarType")
    argument_name <- glue::glue("scalar_type_from_string({argument_name})")

  if (argument$dynamic_type == "TensorList")
    argument_name <- glue::glue("tensor_list_from_r_({argument_name})")

  if (argument$dynamic_type == "Generator *")
    return(NA_character_)

  if (argument$dynamic_type == "MemoryFormat")
    return(NA_character_)

  if (argument$dynamic_type == "ConstQuantizerPtr")
    return(NA_character_)

  argument_name
}

arguments_preprocess <- function(arguments) {
  arguments %>%
    purrr::map_chr(arguments_call_string) %>%
    purrr::discard(is.na) %>%
    paste(collapse = ", ")
}

body_string <- function(method) {

  # create intermediary result for inplace and non inplace
  body <- glue::glue("auto out = self->{method$name}({arguments_preprocess(method$arguments[-1])});")

  # create output based on return types
  if (length(method$returns) == 1) {

    if (method$returns[[1]]$dynamic_type == "Tensor")
      body <- paste(body, "return make_tensor_ptr(out);", sep = "\n")
    else if (method$returns[[1]]$dynamic_type == "void") {
      # explcitily doing nothing
    } else if (method$returns[[1]]$dynamic_type %in% c("double", "bool", "int64_t")) {
      body <- paste(body, "return out;", sep = "\n")
    } else if (method$returns[[1]]$dynamic_type == "TensorList") {

      body <- paste(body,
                    glue::glue(
                      "
                Rcpp::List v;

                for (int i = 0; i < out.size(); ++i) {{
                  v.push_back(make_tensor_ptr(out[i]));
                }}

                return v;
              "
                    ),
                    sep = "\n"
      )

    } else {
      stop("not implemented")
    }

  } else {

    if (all(purrr::map_chr(method$returns, ~.x$dynamic_type) == "Tensor")) {
      body <- paste(body, "return Rcpp::List::create(", sep = "\n")
      for (i in seq_along(method$returns)) {
        body <- paste(body, glue::glue("make_tensor_ptr(std::get<{i-1}>(out))"), sep = "")
        if (i != length(method$returns)) body <- paste(body, ",", sep = "")
      }
      body <- paste(body, ");")
    } else {
      "not implemented"
    }

  }

  body
}

method_string <- function(method) {
  glue::glue("
  // [[Rcpp::export]]
  {signature_string(method)} {{
    {body_string(method)}
  }};")
}

make_method_code <- function(method) {

  if (length(method) == 1) {

    method_string(method[[1]])

  } else {

    if (case_tensor_vs_scalar(method)) {

    } else if (case_single_additional_argument(method)) {

    } else if (case_dim_and_keepdim(method)) {

    } else if (case_dim_keepdim_and_other(method)) {

    } else {
      message(name, " must be manually implemented")
    }

  }

}

methods_code <- c()

for (name in method_names) {
  method <- keep(tensor_methods, ~.x$name == name)
  methods_code <- c(methods_code, make_method_code(method))
}

if (file.exists(methods_fname))
  file.remove(methods_fname)

includes <- '// Dont modify!
// This file is auto generated by tools/gen.R.
#include "torch_types.h"
#include "utils.hpp"
#include "scalar.hpp"
#include "device.hpp"
'
paste(c(includes, methods_code), collapse = "\n\n") %>%
  writeLines(con = methods_fname)






