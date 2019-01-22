
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

Installation is very simple:

### CPU

```r
devtools::install_github(
  "dfalbel/torch", 
  configure.vars = c("TORCH_HOME=~/libtorch")
)
```
Code above will check whether `libtorch` is installed to `TORCH_HOME` dir. If not it will automatically download `libtorch` binaries from `pytorch.org` and unpack them to `TORCH_HOME`. After that it will install `torch` R package. 

Alternatively you can provide URL for binaries download by adding `TORCH_BINARIES` argument to `configure.vars`.

### GPU

On Linux you can also install `torch` with **CUDA 9.0** support (still very initial stage)

**Install CUDA 9.0**

  - [follow these
    instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
    and add necessary repositories
  - install **cuda-9-0** - `sudo apt-get install cuda-9-0`

**Install libtorch and torch R package**

``` r
devtools::install_github(
  "dfalbel/torch", 
  configure.vars = c(
    "TORCH_HOME=~/libtorch",
    "TORCH_BACKEND=CUDA", 
    "CUDA_HOME=/usr/local/cuda"
  )
)
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
#>   0.9947  0.8106
#>   0.1551  0.0735
#> 
#> (2,.,.) = 
#>   0.7466  0.7760
#>   0.0058  0.8846
#> [ Variable[CPUFloatType]{2,2,2} ]
identical(x, as.array(y))
#> [1] FALSE
```

### Simple Autograd Example

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
#> [ Variable[CPUFloatType]{1} ]
w$grad
#> tensor 
#>  1
#> [ Variable[CPUFloatType]{1} ]
b$grad
#> tensor 
#>  1
#> [ Variable[CPUFloatType]{1} ]
```

### Linear Regression

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
#>  0.5217
#> -0.6676
#> [ Variable[CPUFloatType]{2,1} ]
print(b) 
#> tensor 
#> 0.01 *
#>  6.9575
#> [ Variable[CPUFloatType]{1} ]
```
