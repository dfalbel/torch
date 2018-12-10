
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
#>   0.4058  0.6824
#>   0.5909  0.9868
#> 
#> (2,.,.) = 
#>   0.8365  0.4097
#>   0.7636  0.6616
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

Not working yet.

``` r
x <- matrix(runif(100), ncol = 2)
y <- 0.4 + 0.1 * x[,1] + 0.7 * x[,2]

x_t <- tensor(x)
y_t <- tensor(y)

w <- tensor(matrix(rnorm(2), nrow = 2), requires_grad = TRUE)
b <- tensor(0, requires_grad = TRUE)

for (i in 1:100) {
  y_hat <- mm(x_t, w) + b
  loss <- sum(abs(y_t$sub(y_hat)))
  loss$backward()
  
  w <- tensor(w$sub(tensor(0.001)*w$grad), requires_grad = TRUE)
  b <- tensor(b$sub(tensor(0.01)*b$grad), requires_grad = TRUE)
  print(loss)
}
```
