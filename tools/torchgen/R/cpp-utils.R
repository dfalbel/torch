#' Maps a torch type to an Rcpp type.
#'
#' It will return `NA` if we don't know how to map the torch type to an Rcpp type.
#' The user should know how to deal with the `NA`s.
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

  if (type == "TensorList")
    type <- "Rcpp::List"

  if (type == "TensorOptions")
    type <- "Rcpp::XPtr<torch::TensorOptions>"

  if (is_nullable && dynamic_type != "Scalar")
    type <- glue::glue("Rcpp::Nullable<{type}>")

  if (type == "Generator *")
    return(NA_character_) # remove generators from the call

  if (type == "MemoryFormat")
    return(NA_character_) # remove generators from the call

  if (type == "ConstQuantizerPtr")
    return(NA_character_) # remove generators from the call

  return(type)
}
