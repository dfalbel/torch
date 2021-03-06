% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cpp-utils.R
\name{cpp_argument_type}
\alias{cpp_argument_type}
\title{Maps a torch type to an Rcpp type.}
\usage{
cpp_argument_type(argument)
}
\arguments{
\item{argument}{an argument element as in `declarations()[[1]]$arguments[[1]]`}
}
\description{
It will return `NA` if we don't know how to map the torch type to an Rcpp type.
The user should must how to deal with the `NA`s.
}
