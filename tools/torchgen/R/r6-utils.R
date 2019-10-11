#' Returns a named list with default values for all arguments.
#'
#'
get_default_values <- function(methods) {

  arguments <- purrr::map(methods, ~.x$arguments) %>%
    flatten()

  argument_names <- arguments %>% purrr::map_chr(~.x$name)

  arguments %>% split(argument_names) %>%
    map_depth(2, ~.x$default %||% NA_character_) %>%
    map(as.character) %>%
    map(function(x) {
      dplyr::case_when(
        x == "c10::nullopt" ~ "NULL",
        TRUE ~ x
      )
    }) %>%
    map(unique)
}

