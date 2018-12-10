
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
#>   0.3516  0.3185
#>   0.4159  0.4747
#> 
#> (2,.,.) = 
#>   0.4630  0.9065
#>   0.9090  0.6667
#> [ Variable[CPUDoubleType]{2,2,2} ]
identical(x, as.array(y))
#> [1] TRUE
```

## Simple Autograd Example

Now let’s look at the most important feature of torch.

``` r
x <- tensor(1, requires_grad = TRUE)
w <- tensor(2, requires_grad = TRUE)
b <- tensor(3, requires_grad = TRUE)

y <- w * x + b
y$backward()

x$grad
w$grad
b$grad
```
