
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
#>   0.6014  0.3300
#>   0.3268  0.2070
#> 
#> (2,.,.) = 
#>   0.5046  0.3067
#>   0.3022  0.1415
#> [ Variable[CPUDoubleType]{2,2,2} ]
identical(x, as.array(y))
#> [1] TRUE
```

## Simple Autograd Example

Now let’s look at the most important feature of torch.

``` r
x <- matrix(runif(100), nrow = 50)
y <- 8 + 2*x[,1] + 5*x[,2]

x_t <- tensor(x)
y_t <- tensor(y)

w <- tensor(matrix(rnorm(2), nrow = 2), requires_grad = TRUE)
b <- tensor(0, requires_grad = TRUE)

y_hat <- mm(x_t, w) + b

loss <- sum(abs(y_hat + y_t))
loss$backward()
```
