Home Work 2 for Data Science II
================
Roxy Zhang (rz2570)
2/12/2022

### Setup and import data

``` r
library(tidyverse)
library(caret)
library(corrplot)
library(leaps)
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
table(sapply(train_df[ , -1], class)) %>% 
  knitr::kable()
```

| Var1      | Freq |
|:----------|-----:|
| character |    4 |
| numeric   |   21 |

``` r
dim(test_df)
```

    ## [1] 959  26

``` r
table(sapply(test_df[ , -1], class)) %>% 
  knitr::kable()
```

| Var1      | Freq |
|:----------|-----:|
| character |    4 |
| numeric   |   21 |

-   We want to predict outcome variable `sale_price` by selecting
    predictors from 26 variables, among which there are 4 categorical
    variables and 21 continuous variables.  
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
  method = "circle",
  type = "full",
  tl.cex = .5, 
  tl.col = "darkblue")
```

![](ds2_hw1_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

-   From the plot above, we can see some positive correlations in the
    upright corner among some area-related predictors such as
    `gr_liv_area` and `second_flr_sf`, and also negative correlations
    among categorical variables such as `exter_qualTypical` and
    `kitchen_qualTypical`.  
-   To reduce the influence of correlation, we may want to reduce the
    number of predictors using best subset model selection.

``` r
lm_subsets <- regsubsets(sale_price ~ .,
                         data = train_df,
                         method = "exhaustive",
                         nbest = 6)
```

    ## Warning in leaps.setup(x, y, wt = wt, nbest = nbest, nvmax = nvmax, force.in =
    ## force.in, : 1 linear dependencies found

    ## Reordering variables and trying again:

``` r
plot(lm_subsets, scale = "bic")
```

![](ds2_hw1_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
# Look at actual numbers of predictors
# summary(lm_subsets)
```

-   The plot above gives us a sense of model selsction. Hoever, we will
    stick to using all the predictors in this assignment.

## Least Squares

Fit a linear model using least squares on the training data. Is there
any potential disadvantage of this model?

### Model fitting

``` r
set.seed(2570)

# Specify resampling method
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

# Fit a linear model using caret
lm_fit <- train(sale_price ~ .,
                data = train_df,
                method = "lm",
                preProcess = c("center", "scale"),
                trControl = ctrl)
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

``` r
# Extract coefficiencts
round(lm_fit$finalModel$coefficients, 3) %>% 
  knitr::kable()
```

|                            |          x |
|:---------------------------|-----------:|
| (Intercept)                | 177568.502 |
| gr_liv_area                |  11918.089 |
| first_flr_sf               |  15645.580 |
| second_flr_sf              |  17658.087 |
| total_bsmt_sf              |  14564.164 |
| low_qual_fin_sf            |         NA |
| wood_deck_sf               |   1609.235 |
| open_porch_sf              |   1027.166 |
| bsmt_unf_sf                |  -8661.325 |
| mas_vnr_area               |   1756.926 |
| garage_cars                |   3056.314 |
| garage_area                |   1566.065 |
| year_built                 |   9546.428 |
| tot_rms_abv_grd            |  -5883.709 |
| full_bath                  |  -2344.038 |
| overall_qualAverage        |  -2287.250 |
| overall_qualBelow_Average  |  -3314.296 |
| overall_qualExcellent      |  12221.926 |
| overall_qualFair           |  -1367.698 |
| overall_qualGood           |   4994.161 |
| overall_qualVery_Excellent |  12335.966 |
| overall_qualVery_Good      |  11604.560 |
| kitchen_qualFair           |  -3410.143 |
| kitchen_qualGood           |  -9158.701 |
| kitchen_qualTypical        | -13332.542 |
| fireplaces                 |   7400.044 |
| fireplace_quFair           |  -1198.986 |
| fireplace_quGood           |    258.914 |
| fireplace_quNo_Fireplace   |   1697.355 |
| fireplace_quPoor           |   -677.410 |
| fireplace_quTypical        |  -2624.348 |
| exter_qualFair             |  -3914.380 |
| exter_qualGood             |  -9346.023 |
| exter_qualTypical          | -11719.787 |
| lot_frontage               |   3327.997 |
| lot_area                   |   5015.883 |
| longitude                  |   -923.258 |
| latitude                   |   1071.879 |
| misc_val                   |    541.446 |
| year_sold                  |   -831.994 |

``` r
# Calculate mean training RMSE
mean(lm_fit$resample$RMSE)
```

    ## [1] 23004.66

### Make prediction

``` r
# Make prediction on test data
lm_predict <- predict(lm_fit, newdata = test_df)
```

    ## Warning in predict.lm(modelFit, newdata): prediction from a rank-deficient fit
    ## may be misleading

``` r
# Calculate test RMSE
RMSE(lm_predict, test_df$sale_price)
```

    ## [1] 21149.18

**Potential Disadvantage:**  
\* The model contains too many predictors, which is hard for
interpretation.  
\* As seen above, there are correlations among predictors, which may
lead to: 1. higher variance and RMSE 2. less prediction accuracy 3.
difficulty for interpretation  
\* Due to the nature of its modeling method, Least Squares is sensitive
to outliers.  
\* Large data set is necessary in order to obtain reliable results. Our
sample in this case might not be large enough.

## Lasso

Fit a lasso model on the training data and report the test error. When
the 1SE rule is applied, how many predictors are included in the model?

### Model fitting

``` r
set.seed(2570)

# Fit a lasso model using caret
```
