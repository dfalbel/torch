Device <- R6::R6Class(
  classname = "Device",
  public = list(

    pointer = NULL,

    initialize = function(type, index) {
      self$pointer <- device_from_r(type, index)
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

tch_device <- function(type, index = NULL) {
  if (grepl(type, ":") && is.null(index)) {
    type_index <- strsplit(type, ":")[[1]]
    Device$new(type_index[1], as.integer(type_index[2]))
  } else  {
    Device$new(type, index)
  }
}
