declarations_file <- readr::read_lines("https://github.com/pytorch/pytorch/raw/master/aten/src/ATen/native/native_functions.yaml")
# declarations_file <- declarations_file[!stringr::str_detect(declarations_file, "variants:.*")]
# declarations <- yaml::yaml.load(paste(declarations_file, collapse = "\n"))

funcs <- which(stringr::str_detect(declarations_file, "func:"))
variants <- which(stringr::str_detect(declarations_file, "variants:"))

func <- purrr::map_int(
  variants,
  ~max(funcs[funcs < .x])
)

table(func)


text <- "
- func: _cast_Byte(Tensor self, bool non_blocking=False) -> Tensor
  matches_jit_signature: True
  variants: function
---
- func: _cast_Char(Tensor self, bool non_blocking=False) -> Tensor
  matches_jit_signature: True
  variants: function, akakaka
  variants: ajhsjahs
"

yaml::yaml.load(text)
