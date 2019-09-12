library(purrr)
library(glue)

methods_fname <- "R/methods.R"

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


# create all generics
generics_code <- tensor_methods %>%
  map_chr(~.x$name) %>%
  unique() %>%
  map_chr(function(nm) {
    possible_names <- tensor_methods %>%
      keep(~.x$name == nm) %>%
      map(~.x$arguments %>% map_chr(function(.x) {
        if (.x$dynamic_type == "Generator *")
          return(NA_character_)

        .x$name
      }) %>% discard(is.na)) %>%
      flatten_chr() %>%
      unique()

    glue::glue('setGeneric("torch_{nm}_", function({paste(possible_names, collapse =", ")}) standardGeneric("torch_{nm}_"))')
  })

exceptions <- c("qscheme", "item", "polygamma", "set_quantizer_")

method_string <- function(method) {
  glue::glue("
           setMethod(
            f='{generic_name(method)}',
            signature={s4_signature_string(method)},
            definition=function({arguments_string(method)}) {{
           {body_string(method)}
            }}
           )")
}

generic_name <- function(method) {
  glue::glue("torch_{method$name}_")
}

s4_signature_string <- function(method) {
  argument_types <- sapply(method$arguments, arg_r_type)
  argument_types <- purrr::discard(argument_types, is.null)
  glue::glue("list({paste(argument_types, collapse = ', ')})")
}

arg_r_type <- function(argument) {
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
    return(NULL)

  if (argument_type == "ScalarType")
    argument_type <- "character"

  if (argument_type == "MemoryFormat")
    argument_type <- "character"

  if (argument_type == "TensorList")
    argument_type <- "list"

  glue::glue("{argument_name}='{argument_type}'")
}

arguments_string <- function(method) {
  args <- sapply(method$arguments, function(x) {
    if (x$dynamic_type == "Generator *")
      return(NULL)
    x$name
  })
  args <- purrr::discard(args, is.null)
  paste0(args, collapse = ", ")
}

hash_arguments <- function(arguments) {
  types <- paste0(map_chr(arguments, ~.x$type), collapse = "")
  names <- paste0(map_chr(arguments, ~.x$name), collapse = "")
  openssl::md5(glue::glue("{types}{names}"))
}

cpp_fun_name <- function(method) {
  glue::glue("torch_{method$name}_{hash_arguments(method$arguments)}")
}

body_string <- function(method) {
  cpp_fun_name <- cpp_fun_name(method)
  glue::glue("{cpp_fun_name}({arguments_string(method)})")
}


c(
  generics_code %>% paste(collapse = "\n"),
  "\n",
  map_chr(tensor_methods, method_string) %>% paste(collapse = "\n")
) %>%
  writeLines(con = methods_fname)
