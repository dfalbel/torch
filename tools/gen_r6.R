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
tensor_methods %>%
  map_chr(~.x$name) %>%
  unique()

exceptions <- c("qscheme", "item", "polygamma", "set_quantizer_")

generic_names <- tensor_methods %>%
  map_chr(~.x$name) %>%
  unique()


make_method_code <- function(method_name, arguments_string, function_call) {
  glue::glue(
  '
  `torch::Tensor`$set("public", "{method_name}", function({arguments_string}) {{
    {function_call}
  })
  ')
}

make_method <- function(method) {

  method_name <- method[[1]]$name
  arguments_string <- arguments_string(method)
  function_call <- function_call_string(method)

  make_method_code(method_name, arguments_string, function_call)
}


arguments_string <- function(method) {
  if (length(method) == 1) {
    if (length(method[[1]]$arguments) > 1) {

    } else {
      return("")
    }
  }
}

function_call_string <- function(method) {
  if (length(method) == 1) {
    method <- method[[1]]
    if (length(method$arguments)  > 1) {

    } else {
      glue::glue("`torch::Tensor`$dispatch(torch_{method$name}_(self$pointer))")
    }
  }
}

code <- NULL

for (name in generic_names) {
  method <- tensor_methods %>% keep(~.x$name == name)
  if (length(method) == 1 && length(method[[1]]$arguments) == 1) {
    code <- c(code, make_method(method))
  }
}

paste(code, collapse = "\n\n") %>%
  writeLines("R/methods-r6.R")
