## Practical Machine Learning Course Project:

### Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants.

They were asked to perform barbell lifts correctly and incorrectly in 5 different ways, exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. More details can be found in the paper “Qualitative Activity Recognition of Weight Lifting Exercises” written by Eduardo Velloso et. al., which can be found at this site:

<http://web.archive.org/web/20170519033209/http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf>

Data to conduct this analysis is found in these sites. The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The goal of your project is to predict the manner in which they did the exercise. This is the “classe” variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

### Methodology

The following steps were followed in order to determine what is the best way to predict “classe” on function of the data acquired through devices located on different parts of the participant bodies.

1. Download, and read the data
2. Inspect the data
3. Clean the data by removing variables (columns) that don’t have values/NA, or great majority are zeros)
4. Analyze the train set by creating models using different techniques such as “Predicting with Trees”, “Random Forest”, “Boosting”, “Linear Discriminant Analysis”, and finally “Combining Predictors”
5. A summary of the results will be provided and a recommendation on what is the most applicable technique to predict “classe”

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
