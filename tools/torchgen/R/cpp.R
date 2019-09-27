# # This script is used to Generate wrapers for pytorch functions.
# # The generated code is created by  splitting methods in the follwing groups:
# #
# # - Methods that have a single signature
# #   - NUllable arguments based in c10::optional can be solved with `resolve_null_argument`
# # - Methods that don't have a single signature
# #   - Methods that have different signatures for scalars and tensors
# #   - Methods that have a single additional argument.
# #   - Methods that have a second signature with dim and keepdim args
# #   - Other methods -> code is not automatically generated.
# #
# #
#
# library(purrr)
# library(glue)
#
# methods_fname <- "src/methods.cpp"
#
# declarations <- yaml::read_yaml(
#   file = "~/Documents/pytorch/build/aten/src/ATen/Declarations.yaml",
#   eval.expr = FALSE,
#   handlers = list(
#     'bool#yes' = function(x) if (x == "y") x else TRUE,
#     'bool#no' = function(x) if (x == "n") x else FALSE,
#     int = identity
#   )
# )
#
# tensor_methods <- declarations %>%
#   keep(~"Tensor" %in% .x$method_of)
#
# exceptions <- c("qscheme", "item", "polygamma", "set_quantizer_")
#
# method_names <- method_names[!method_names %in% exceptions]
#
# # creates a single id for the function based on the
# hash_arguments <- function(arguments) {
#   types <- paste0(map_chr(arguments, ~.x$type), collapse = "")
#   names <- paste0(map_chr(arguments, ~.x$name), collapse = "")
#   openssl::md5(glue::glue("{types}{names}"))
# }
#
# argument_string <- function(argument) {
#
#   type <- argument$dynamic_type
#
#   if (type == "Tensor")
#     type <- "Rcpp::XPtr<torch::Tensor>"
#
#   if (type == "Scalar")
#     type <- "SEXP"
#
#   if (type == "IntArrayRef")
#     type <- "std::vector<std::int64_t>"
#
#   if (type == "ScalarType")
#     type <- "std::string"
#
#   if (type == "Device")
#     type <- "std::string"
#
#   if (type == "Storage")
#     type <- "Rcpp::XPtr<torch::Storage>"
#
#   if (type == "TensorList")
#     type <- "Rcpp::List"
#
#   if (type == "TensorOptions")
#     type <- "Rcpp::XPtr<torch::TensorOptions>"
#
#   if (argument$is_nullable && argument$dynamic_type != "Scalar")
#     type <- glue::glue("Rcpp::Nullable<{type}>")
#
#   if (type == "Generator *")
#     return(NA_character_) # remove generators from the call
#
#   if (type == "MemoryFormat")
#     return(NA_character_) # remove generators from the call
#
#   if (type == "ConstQuantizerPtr")
#     return(NA_character_) # remove generators from the call
#
#   glue::glue("{type} {argument$name}")
# }
#
# arguments_string <- function(arguments) {
#   arguments %>%
#     purrr::map_chr(argument_string) %>%
#     purrr::discard(is.na) %>%
#     paste(collapse = ", ")
# }
#
# return_type_string <- function(method) {
#   if (length(method$returns) == 1) {
#     if (method$returns[[1]]$dynamic_type == "Tensor")
#       "Rcpp::XPtr<torch::Tensor>"
#     else if (method$returns[[1]]$dynamic_type %in% c("void", "bool", "double", "int64_t"))
#       method$returns[[1]]$dynamic_type
#     else if (method$returns[[1]]$dynamic_type == "TensorList")
#       "Rcpp::List"
#
#
#   } else if (all(purrr::map_chr(method$returns, ~.x$dynamic_type) == "Tensor")) {
#     "Rcpp::List"
#   }
# }
#
# signature_string <- function(method) {
#
#   return_type <-
#
#     glue::glue("
#      {return_type_string(method)} torch_{method$name}_{hash_arguments(method$arguments)} (
#      {arguments_string(method$arguments)}
#      )
#      ")
# }
#
# arguments_call_string <- function(argument) {
#
#   argument_name <- argument$name
#
#   if (grepl("c10::optional", argument$type) && !argument$dynamic_type %in% c("ScalarType", "Scalar"))
#     argument_name <- glue::glue("resolve_null_argument({argument_name})")
#
#   if (argument$dynamic_type == "Scalar" && ! argument$is_nullable)
#     argument_name <- glue::glue("scalar_from_r_({argument_name})")
#
#   if (argument$dynamic_type == "Scalar" && argument$is_nullable)
#     argument_name <- glue::glue("resolve_null_scalar({argument_name})")
#
#   if (argument$dynamic_type == "ScalarType")
#     argument_name <- glue::glue("scalar_type_from_string({argument_name})")
#
#   if (argument$dynamic_type == "TensorList")
#     argument_name <- glue::glue("tensor_list_from_r_({argument_name})")
#
#   if (argument$dynamic_type == "Device")
#     argument_name <- glue::glue("device_from_string({argument_name})")
#
#   if (argument$dynamic_type == "Storage")
#     argument_name <- glue::glue("*{argument_name}")
#
#   if (argument$dynamic_type == "TensorOptions")
#     argument_name <- glue::glue("*{argument_name}")
#
#   if (argument$dynamic_type == "Tensor") {
#
#     if (argument$is_nullable)
#       argument_name <- glue::glue("Rcpp::as<Rcpp::XPtr<torch::Tensor>>({argument_name})")
#
#     argument_name <- glue::glue("*{argument_name}")
#   }
#
#   if (argument$dynamic_type == "Generator *")
#     return(NA_character_)
#
#   if (argument$dynamic_type == "MemoryFormat")
#     return(NA_character_)
#
#   if (argument$dynamic_type == "ConstQuantizerPtr")
#     return(NA_character_)
#
#   argument_name
# }
#
# arguments_preprocess <- function(arguments) {
#   arguments %>%
#     purrr::map_chr(arguments_call_string) %>%
#     purrr::discard(is.na) %>%
#     paste(collapse = ", ")
# }
#
# body_string
# <- function(method) {
#
#   # create intermediary result for inplace and non inplace
#
#   body <- glue::glue("self->{method$name}({arguments_preprocess(method$arguments[-1])});")
#
#   if (method$returns[[1]]$dynamic_type != "void"){
#     body <- glue::glue("auto out = {body}")
#   }
#
#   # create output based on return types
#   if (length(method$returns) == 1) {
#
#     dynamic_type <- method$returns[[1]]$dynamic_type
#
#     if (dynamic_type == "Tensor") {
#       body <- paste(body, "return make_tensor_ptr(out);", sep = "\n")
#     } else if (dynamic_type == "QScheme") {
#       body <- paste(body, "return make_qscheme_ptr(out);", sep = "\n")
#     } else if (dynamic_type == "Scalar") {
#       body <- paste(body, "return scalar_to_r_(out);", sep = "\n")
#     } else if (dynamic_type == "void") {
#       # explcitily doing nothing
#     } else if (dynamic_type %in% c("double", "bool", "int64_t")) {
#       body <- paste(body, "return out;", sep = "\n")
#     } else if (dynamic_type == "TensorList") {
#
#       body <- paste(body,
#                     glue::glue(
#                       "
#                 Rcpp::List v;
#
#                 for (int i = 0; i < out.size(); ++i) {{
#                   v.push_back(make_tensor_ptr(out[i]));
#                 }}
#
#                 return v;
#               "
#                     ),
#                     sep = "\n"
#       )
#
#     } else {
#       stop("not implemented")
#     }
#
#   } else {
#
#     if (all(purrr::map_chr(method$returns, ~.x$dynamic_type) == "Tensor")) {
#       body <- paste(body, "return Rcpp::List::create(", sep = "\n")
#       for (i in seq_along(method$returns)) {
#         body <- paste(body, glue::glue("make_tensor_ptr(std::get<{i-1}>(out))"), sep = "")
#         if (i != length(method$returns)) body <- paste(body, ",", sep = "")
#       }
#       body <- paste(body, ");")
#     } else {
#       "not implemented"
#     }
#
#   }
#
#   body
# }
#
# method_string <- function(method) {
#   glue::glue("
#   // [[Rcpp::export]]
#   {signature_string(method)} {{
#     {body_string(method)}
#   }};")
# }
#
# methods_code <- c()
# for (method in tensor_methods) {
#
#   if (method$name %in% exceptions)
#     next
#
#
#   methods_code <- c(methods_code, method_string(method))
# }
#
# if (file.exists(methods_fname))
#   file.remove(methods_fname)
#
# includes <- '// Dont modify!
# // This file is auto generated by tools/gen.R.
# include "torch_types.h"
# include "utils.hpp"
# include "scalar.hpp"
# include "device.hpp"
# '
# paste(c(includes, methods_code), collapse = "\n\n") %>%
#   writeLines(con = methods_fname)
#
#
#
#
#
#
