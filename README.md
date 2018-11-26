
<!-- README.md is generated from README.Rmd. Please edit that file -->

# torch

torch from R\!

## Installation

Before installing you should [install libtorch](https://pytorch.org/) in
`usr/local/lib/`.

``` r
devtools::install_github("dfalbel/torch")
```

## Example

Currently this package is only a prrof of concept and you can only
create a Torch Tensor from an R object. And then convert back from a
torch Tensor to an R object.

``` r
library(torch)
x <- array(runif(8), dim = c(2, 2, 2))
y <- tensor(x)
y
#> torch::Tensor 
#> (1,.,.) = 
#>   0.8388  0.4398
#>   0.2740  0.8836
#> 
#> (2,.,.) = 
#>   0.9201  0.2277
#>   0.6008  0.2253
#> [ Variable[CPUDoubleType]{2,2,2} ]
identical(x, as.array(y))
#> [1] TRUE
```
