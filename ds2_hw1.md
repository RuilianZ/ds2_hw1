ds2_hw1
================
Roxy Zhang (rz2570)
2/12/2022

## Setup and import data

``` r
library(tidyverse)
library(caret)
```

``` r
train_df = 
  read_csv("data/housing_training.csv") %>% 
  janitor::clean_names() %>% 
  na.omit()
```

    ## Rows: 1440 Columns: 26

    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr  (4): Overall_Qual, Kitchen_Qual, Fireplace_Qu, Exter_Qual
    ## dbl (22): Gr_Liv_Area, First_Flr_SF, Second_Flr_SF, Total_Bsmt_SF, Low_Qual_...

    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
test_df = 
  read_csv("data/housing_test.csv") %>% 
  janitor::clean_names() %>% 
  na.omit()
```

    ## Rows: 959 Columns: 26

    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr  (4): Overall_Qual, Kitchen_Qual, Fireplace_Qu, Exter_Qual
    ## dbl (22): Gr_Liv_Area, First_Flr_SF, Second_Flr_SF, Total_Bsmt_SF, Low_Qual_...

    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

``` r
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

## Linear model

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
