
<!-- README.md is generated from README.Rmd. Please edit that file -->

# torch

[![lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
[![Travis build
status](https://travis-ci.org/dfalbel/torch.svg?branch=master)](https://travis-ci.org/dfalbel/torch)
[![Coverage
status](https://codecov.io/gh/dfalbel/torch/branch/master/graph/badge.svg)](https://codecov.io/github/dfalbel/torch?branch=master)

torch from R\!

> A proof of concept for calling libtorch functions from R. API will
> change\! Use at your own risk. Most libtorch’s functionality is not
> implemented here too.

## Installation

Before installing you should [install libtorch](https://pytorch.org/) in
`usr/local/lib/`.

``` r
devtools::install_github("dfalbel/torch")
```

### Linux

On Linux the fastest way to get started is to run on
    `bash`:

    wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
    sudo unzip libtorch-shared-with-deps-latest.zip -d /usr/local/lib/

You can then install the package with

``` r
devtools::install_github("dfalbel/torch")
```

### MacOs

On MacOS the following should just work too. First install libtorch
with:

    wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-latest.zip
    sudo unzip libtorch-macos-latest.zip -d /usr/local/lib/

Finally run:

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
#> tensor 
#> (1,.,.) = 
#>   0.5770  0.2904
#>   0.8738  0.0852
#> 
#> (2,.,.) = 
#>   0.7112  0.0138
#>   0.7922  0.4174
#> [ Variable[CPUDoubleType]{2,2,2} ]
identical(x, as.array(y))
#> [1] TRUE
```

## Simple Autograd Example

In the following snippet we let torch, using the autograd feature,
calculate the derivatives:

``` r
x <- tensor(1, requires_grad = TRUE)
w <- tensor(2, requires_grad = TRUE)
b <- tensor(3, requires_grad = TRUE)

y <- w * x + b
y$backward()

x$grad
#> tensor 
#>  2
#> [ Variable[CPUDoubleType]{1} ]
w$grad
#> tensor 
#>  1
#> [ Variable[CPUDoubleType]{1} ]
b$grad
#> tensor 
#>  1
#> [ Variable[CPUDoubleType]{1} ]
```

## Linear Regression

In the following example we are going to fit a linear regression from
scratch using torch’s Autograd.

**Note** all methods that end with `_` (eg. `sub_`), will modify the
tensors in place.

``` r
x <- matrix(runif(100), ncol = 2)
y <- matrix(0.1 + 0.5 * x[,1] - 0.7 * x[,2], ncol = 1)

x_t <- tensor(x)
y_t <- tensor(y)

w <- tensor(matrix(rnorm(2), nrow = 2), requires_grad = TRUE)
b <- tensor(0, requires_grad = TRUE)

lr <- 0.5

for (i in 1:100) {
  y_hat <- tch_mm(x_t, w) + b
  loss <- tch_mean((y_t - y_hat)^2)
  
  loss$backward()
  
  with_no_grad({
    w$sub_(w$grad*lr)
    b$sub_(b$grad*lr)   
  })
  
  w$grad$zero_()
  b$grad$zero_()
}

print(w)
#> tensor 
#>  0.4996
#> -0.7002
#> [ Variable[CPUDoubleType]{2,1} ]
print(b)
#> tensor 
#>  0.1003
#> [ Variable[CPUDoubleType]{1} ]
```
