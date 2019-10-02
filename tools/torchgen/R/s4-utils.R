#' Get possible arguments for all overloaded methods.
#' The returned order is **important**!
#'
#' @inheritParams generic_s4_code
#'
get_possible_argument_names <- function(methods) {

  # remove args that are not wraped
  arguments <- map(methods, ~.x$arguments)

  argument_names <- map(arguments, ~map_chr(.x, function(.x) {

    if (.x$dynamic_type == "Generator *")
      return(NA_character_)

    .x$name
  }) %>% discard(is.na))

  unq <- argument_names %>%
    flatten_chr() %>%
    unique()

  ind <-  unq %>%
    map(function(x) {
      map_int(argument_names, ~which(.x == x)[1])
    }) %>%
    map_int(max, na.rm = TRUE) %>%
    order()

  unq[ind]
}

#' Returns the R type of an argument in the declarations
#'
#' @param argument an argument like `declarations()[[1]]$arguments[[1]]`
#' @note Returns `NA` when the argument doesn't have an R equivalent
#'
argument_type_to_r <- function(argument) {

  argument_name <- argument$name
  argument_type <- argument$dynamic_type

  if (argument_type == "bool")
    argument_type <- "logical"

  if (argument_type == "Tensor")
    argument_type <- "externalptr"

  if (argument_type == "Scalar")
    argument_type <- "numeric"

  if (argument_type == "int64_t")
    argument_type <- "numeric"

  if (argument_type == "IntArrayRef")
    argument_type <- "numeric"

  if (argument_type == "Generator *")
    return(NA)

  if (argument_type == "ScalarType")
    argument_type <- "character"

  if (argument_type == "MemoryFormat")
    argument_type <- "character"

  if (argument_type == "TensorList")
    argument_type <- "list"

  if (argument_type == "TensorOptions")
    argument_type <- "list"

  argument_type
}
