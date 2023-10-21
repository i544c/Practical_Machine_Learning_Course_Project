---
title : "Practical Machine Learning Course Project"
author: "Isaac G Veras"
date  : "05/10/2023"
output: html_document
---

## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement -- a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants.

They were asked to perform barbell lifts correctly and incorrectly in 5 different ways, exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.

More details can be found in the paper "Qualitative Activity Recognition of Weight Lifting Exercises" written by Eduardo Velloso et. al., which can be found at this site: <http://web.archive.org/web/20170519033209/http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf>

Data to conduct this analysis is found in these sites. The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

## Methodology

The following steps were followed in order to determine what is the best way to predict "classe" on function of the data acquired through devices located on different parts of the participant bodies.

1.  Download, and read the data
2.  Inspect the data
3.  Clean the data by removing variables (columns) that don't have values/NA, or great majority are zeros)
4.  Analyze the train set by creating models using different techniques such as "Predicting with Trees", "Random Forest", "Boosting", "Linear Discriminant Analysis", and finally "Combining Predictors"
5.  A summary of the results will be provided and a recommendation on what is the most applicable technique to predict "classe"

**Descriptive Summary of the Variables:**

1.  `pml_model_nb:` Trained model using the Naive Bayes algorithm to predict the "classe" variable based on the training dataset. The model was fitted using cross-validation, and the overall accuracy is stored in "accuracy_nb."

2.  `pml_training_data and pml_test_data:` DataFrames that store the raw data from the training and test datasets, respectively. The data was read from specific URLs.

3.  `pml_training_data_clean and pml_test_data_clean:` DataFrames containing data after cleaning. Variables with many blank or NA values were removed. The cleaned DataFrames will be used for model training and testing.

4.  `pml_training, pml_validation, and pml_testing:` Subsets of the training and test data that were split for training, validation, and testing, respectively. This allows for evaluating the model's performance.

5.  `cluster:` A parallel computing cluster used to improve performance in operations that can be parallelized. It's configured to use a smaller number of cores than the total available cores (one reserved for the operating system).

6.  `fit_control:` A control setting for model training. This includes details about the cross-validation method and whether parallel execution is allowed.

7.  `pml_model_rp, pml_model_rf, pml_model_gbm, pml_model_lda:` Models trained with Decision Trees, Random Forests, Gradient Boosting Machine, and Linear Discriminant Analysis algorithms, respectively. Each model is trained and tuned according to specific settings.

8.  `pml_val_rp, pml_val_rf, pml_val_gbm, pml_val_lda:` Predictions made by the corresponding models on the validation data.

9.  `pml_levels:` Levels of the "classe" variable after conversion to a factor. This is used in accuracy assessment.

10. `accuracy_rp, accuracy_rf, accuracy_gbm, accuracy_lda:` Overall accuracy of the corresponding models when making predictions on the validation data. These measures reflect how well the models performed in the classification task.

11. `test_results:` Predictions made by the selected model (Boosting with Trees) on the test data.

12. `problem_id:` A DataFrame containing the "problem_id" variable from the test data. It's used to create a results table.

13. `test_results_table:` A table that combines "problem_id" and "classe" predictions for the test data. This provides predictions for each "problem ID" in the test dataset.

**Package installation:**

```{r}
if (!require("pacman")) install.packages("pacman")
pacman::p_load(pacman,        # Package Manager
               GGally,        # Extends 'ggplot2' by adding several functions to reduce the complexity
               knitr,         # Provides tools for transforming R Markdown documents into various output formats
               data.table,    # Offers tools for manipulating, processing, and analyzing large data sets
               tidyverse,     # A collection of R packages for data organization and manipulation
               gridExtra,     # Extends the functionality of 'grid' graphics
               caret,         # Provides functions for training and evaluating machine learning models
               gbm,           # Implements Gradient Boosting Machines for predictive modeling
               dplyr,         # A grammar of data manipulation
               rpart,         # Recursive Partitioning and Regression Trees
               rpart.plot,    # Enhances the visualization of decision trees
               rattle,        # A graphical user interface for data mining using R
               randomForest,  # Implements random forest algorithms for classification and regression
               corrplot,      # Visualizes correlation matrices and allows customization
               elasticnet,    # Provides functions for fitting elastic net models
               pgmm,          # A package for various statistical and graphical tools
               doParallel,    # Provides a parallel backend for the %dopar% function in foreach
               klaR           # # Provides diverse classification and visualization functions
)
```

## Data Preprocessing:

**Downloading, reading and saving data sets**

Use the above links to download, read and save the data sets.

```{r download and read data}
training_data_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(training_data_url, destfile = "./pml-training.csv")
pml_training_data <- read.csv("./pml-training.csv")

test_data_url     <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(test_data_url, destfile = "./pml-testing.csv")
pml_test_data     <- read.csv("./pml-testing.csv")
```

**Data inspection and cleaning**

The data set will be inspected and all the variables without values, zeros, and/or NA will be remove. Them the training set will be split in training set and validation set. The test set will be used to confirm the accuracy of the model.

```{r clean data sets}
dim(pml_training_data)
str(pml_training_data)

# Remove all the variable with zeros, NA out of the train set
# Here we get the indexes of the columns having at least 90% of NA or blank values
ind_col_to_remove_train   <- which(colSums(is.na(pml_training_data) | pml_training_data == "") > 0.9 * dim(pml_training_data)[1])
pml_training_data_clean   <- pml_training_data[, -ind_col_to_remove_train]
pml_training_data_clean   <- pml_training_data_clean[, -c(1:7)]

dim(pml_training_data_clean)

# The same protocol will be done in test data set
index_col_to_remove_test <- which(colSums(is.na(pml_test_data) | pml_test_data == "") > 0.9 * dim(pml_test_data)[1])
pml_test_data_clean      <- pml_test_data[, -index_col_to_remove_test]
pml_test_data_clean      <- pml_test_data_clean[, -c(1:7)]

dim(pml_test_data_clean)
```

**Setting up Training, Validation and Test sets**

The train data set will be split in training set (`75%`) and Validation set (`25%`). The test set will not be changed.

```{r setting data sets}
set.seed(20210129)
in_train       <- createDataPartition(pml_training_data_clean$classe, p = 3 / 4)[[1]]
pml_training   <- pml_training_data_clean[in_train,]
pml_validation <- pml_training_data_clean[-in_train,]
pml_testing    <- pml_test_data_clean
dim(pml_training)
dim(pml_validation)
dim(pml_testing)
```

## Modeling

### Set up processing

```{r processing}
cluster     <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fit_control <- trainControl(
        method        = "cv",
        number        = 5,
        allowParallel = TRUE
)
```

**Predicting with Recursive Partitioning (`Trees`)**

```{r rpart model}
pml_model_rp <- train(
        classe ~ .,
        data       = pml_training,
        method     = "rpart",
        preProcess = "pca",
        na.action  = na.omit,
        trControl  = fit_control
)
pml_val_rp  <- predict(pml_model_rp, newdata =pml_validation)
pml_levels  <- levels(factor(pml_validation$classe))
accuracy_rp <- confusionMatrix(factor(pml_val_rp, levels = pml_levels), factor(pml_validation$classe, levels = pml_levels))$overall['Accuracy']
fancyRpartPlot(pml_model_rp$finalModel)
```
![Rplot18](https://github.com/i544c/Practical_Machine_Learning_Course_Project/assets/104391905/1a90c268-c33c-4157-99ec-a202757082c4)


**Predicting with Random Forests**

```{r rf model}
pml_model_rf <- train(
        classe ~ .,
        data       = pml_training,
        method     = "rf",
        preProcess = "pca",
        na.action  = na.omit,
        trControl  = fit_control
)
pml_val_rf  <- predict(pml_model_rf, newdata = pml_validation)
accuracy_rf <- confusionMatrix(factor(pml_val_rf, levels = pml_levels), factor(pml_validation$classe, levels = pml_levels))$overall['Accuracy']
```

**Prediction with GBM (`Gradient Boosting Machine`) (`Boosting with Trees`)**

```{r gbm model}
pml_model_gbm <- train(
        classe ~ .,
        data       = pml_training,
        method     = "gbm",
        preProcess = "pca",
        na.action  = na.omit,
        trControl  = fit_control
)
pml_val_gbm  <- predict(pml_model_gbm, newdata = pml_validation)
accuracy_gbm <- confusionMatrix(factor(pml_val_gbm, levels = pml_levels), factor(pml_validation$classe, levels = pml_levels))$overall['Accuracy']


```

### Prediction with Linear Discriminate Analysis

```{r lda model}
pml_model_lda <- train(
        classe ~ .,
        data       = pml_training,
        method     = "lda",
        preProcess = "pca",
        na.action  = na.omit,
        trControl  = fit_control
)
pml_val_lda  <- predict(pml_model_lda, newdata = pml_validation)
accuracy_lda <- confusionMatrix(factor(pml_val_lda, levels = pml_levels), factor(pml_validation$classe, levels = pml_levels))$overall['Accuracy']

```

### Prediction with Naive Bayes

```{r nb model,echo=FALSE}
pml_model_nb <- train(
        classe ~ .,
        data       = pml_training,
        method     = "nb",
        preProcess = "pca",
        na.action  = na.omit,
        trControl  = fit_control
)
pml_val_nb  <- predict(pml_model_nb, newdata =pml_validation)
accuracy_nb <- confusionMatrix(factor(pml_val_nb, levels = pml_levels), factor(pml_validation$classe, levels = pml_levels))$overall['Accuracy']
```

# Results

The below table summarizes the results of running 5 different techniques to predict "classe". Random Forest and Boosting with Trees offered the highest accuracy; where **`r accuracy_rf`**, **`r accuracy_gbm`** are their accuracies respectably. Trees have the highest accuracy, but it could be over fitting, therefore, it was selected Boosting with Trees to be used with test data set.

```{r results}
results                 <- matrix(c(accuracy_rp, accuracy_rf, accuracy_gbm, accuracy_lda, accuracy_nb), nrow = 5, ncol = 1)
row.names(results)      <- c("Trees", "Random Forests", "Boosting with Trees", "Linear Discriminate Analysis", "Naive Bayes")
results_table           <- as.table(results)
colnames(results_table) <- c("Accuracy")
results_table

```

# Using Model with Test Dataset

The below table summarizes the predictions for each "problem ID" in the test data set

```{r test}
test_results                  <- predict(pml_model_gbm, newdata = pml_testing)
problem_id                    <- as.data.frame(t(pml_testing["problem_id"]))
test_results_table            <- rbind(problem_id, as.character(test_results))
row.names(test_results_table) <- c("problem_ID", "Class")
unname(test_results_table)
```
