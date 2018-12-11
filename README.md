
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
#> 
#> Attaching package: 'torch'
#> The following object is masked from 'package:base':
#> 
#>     atan2
x <- array(runif(8), dim = c(2, 2, 2))
y <- tensor(x)
y
#> tensor 
#> (1,.,.) = 
#>   0.6950  0.3185
#>   0.2154  0.2710
#> 
#> (2,.,.) = 
#>   0.1312  0.2095
#>   0.6386  0.0826
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
y <- 0.1 + 0.5 * x[,1] - 0.7 * x[,2]

x_t <- tensor(x)
y_t <- tensor(y)

w <- tensor(matrix(rnorm(2), nrow = 2), requires_grad = TRUE)
b <- tensor(0, requires_grad = TRUE)

lr <- tensor(0.5)

for (i in 1:100) {
  y_hat <- mm(x_t, w) + b
  loss <- mean((y_hat$t()$sub(y_t)$pow(tensor(2))))
  
  loss$backward()
  
  w$data$sub_(w$grad*lr)
  b$data$sub_(b$grad*lr)
  
  w$grad$zero_()
  b$grad$zero_()
}

print(as.array(w))
#>            [,1]
#> [1,]  0.5012888
#> [2,] -0.6986030
print(as.array(b))
#> [1] 0.09858889
```
