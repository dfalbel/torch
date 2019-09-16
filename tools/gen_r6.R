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
