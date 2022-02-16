Home Work 2 for Data Science II
================
Roxy Zhang (rz2570)
2/12/2022

### Setup and import data

``` r
library(tidyverse)
library(caret)
library(corrplot)
```

``` r
train_df = 
  read_csv("data/housing_training.csv") %>% 
  janitor::clean_names() %>% 
  na.omit()

test_df = 
  read_csv("data/housing_test.csv") %>% 
  janitor::clean_names() %>% 
  na.omit()

dim(train_df)
```

    ## [1] 1440   26

``` r
table(sapply(train_df, class)) %>% 
  knitr::kable()
```

| Var1      | Freq |
|:----------|-----:|
| character |    4 |
| numeric   |   22 |

``` r
dim(test_df)
```

    ## [1] 959  26

``` r
table(sapply(test_df, class)) %>% 
  knitr::kable()
```

| Var1      | Freq |
|:----------|-----:|
| character |    4 |
| numeric   |   22 |

-   We want to predict outcome variable `sale_price` by selecting
    predictors from 26 variables, among which there are 4 categorical
    variables and 22 continuous variables.  
-   There are 1440 data in training data and 959 in test data.

### Data preparation

For the convenience of fitting models, we want to create vectors and
matrices in advance:

``` r
train_x <- model.matrix(sale_price ~ ., train_df)[ ,-1]
train_y <- train_df$sale_price
test_x <- model.matrix(sale_price ~ ., test_df)[ , -1]
test_y <- test_df$sale_price
```

In linear regression, correlation among variables can cause large
variance and make interpretation harder.  
So we want to have a look and the potential correlation among
predictors:

``` r
corrplot(
  cor(train_x), 
  type = "full",
  tl.cex = .5, 
  tl.col = "darkblue")
```

![](ds2_hw1_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

## Least Squares

Fit a linear model using least squares on the training data. Is there
any potential disadvantage of this model?

``` r
set.seed(2570)

fit.lm <- train(sale_price ~ .,
                data = train_df,
                method = "lm",
                trControl = trainControl(method = "cv", number = 10))
```

    ## Warning in predict.lm(modelFit, newdata): prediction from a rank-deficient fit
    ## may be misleading

    ## Warning in predict.lm(modelFit, newdata): prediction from a rank-deficient fit
    ## may be misleading

    ## Warning in predict.lm(modelFit, newdata): prediction from a rank-deficient fit
    ## may be misleading

    ## Warning in predict.lm(modelFit, newdata): prediction from a rank-deficient fit
    ## may be misleading

    ## Warning in predict.lm(modelFit, newdata): prediction from a rank-deficient fit
    ## may be misleading

    ## Warning in predict.lm(modelFit, newdata): prediction from a rank-deficient fit
    ## may be misleading

    ## Warning in predict.lm(modelFit, newdata): prediction from a rank-deficient fit
    ## may be misleading

    ## Warning in predict.lm(modelFit, newdata): prediction from a rank-deficient fit
    ## may be misleading

    ## Warning in predict.lm(modelFit, newdata): prediction from a rank-deficient fit
    ## may be misleading

    ## Warning in predict.lm(modelFit, newdata): prediction from a rank-deficient fit
    ## may be misleading
