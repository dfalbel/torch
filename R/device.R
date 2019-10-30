Device <- R6::R6Class(
  classname = "Device",
  public = list(

    pointer = NULL,

    initialize = function(type = NULL, index = NULL, pointer = NULL) {
      if (!is.null(pointer))
        self$pointer <- pointer
      else if (!is.null(type))
        self$pointer <- device_from_r(type, index)
      else
        stop("You must specify a type and a index (or a Device pointer)")
    },

    has_index = function() {
      device_has_index(self$pointer)
    },

    is_cuda = function() {
      device_is_cuda(self$pointer)
    },

    is_cpu = function() {
      device_is_cpu(self$pointer)
    },

    set_index = function(index) {
      device_set_index(self$pointer, index)
      invisible(self)
    },

    print = function() {
      out <- self$type
      index <- self$index

      if (index >= 0)
        out <- paste0(out, ":", index)

      cat(out)
    }

  ),

  active = list(

    index = function(x) {
      if (missing(x))
        get_device_index(self$pointer)
      else
        set_device_index(self$pointer, x)
    },

    type = function(x) {
      if (missing(x))
        get_device_type(self$pointer)
      else
        stop("Can't change device type.", call. = FALSE)
    }

  )
)

#' Create a Device
#'
#' @param type a device type, 'cuda' or 'cpu'.
#' @param index an index for the device (starting from 0). only used for 'cuda'
#'  devices.
#'
#' @export
tch_device <- function(type, index = NULL) {
  if (grepl(type, ":") && is.null(index)) {
    type_index <- strsplit(type, ":")[[1]]
    Device$new(type_index[1], as.integer(type_index[2]))
  } else  {
    Device$new(type, index)
  }
}

`==.Device` <- function(e1, e2) {
  device_equals(e1$pointer, e2$pointer)
}
