# Recipe Rating Model

by George Chong (gchong@ucsd.edu) and Antara Sengupta (asengupt@ucsd.edu) 


## Framing the Problem 
In this project, we will be answering the predictive problem: 

> Given a recipe, what should we predict the average rating to be? 

Since average rating is a continuous variable, we want to create a regression model to predict the average rating. 

We will be evaluating our model with the R squared value. The reason for evaluating the model using the rmse is that we want to penalize large errors by a bigger magnitude than smaller errors. 

At the time of making our prediction, we will only have the information about the recipe itself. We do not know how the public will interact with the recipe, so information such as the comments or ratings a recipe will get will not be available.

In this project, we are exploring the prediction of the average rating of a recipe given the recipe name, based on both pre-existing and engineered features. This problem is a regression one as the target of our discovery is a continuous numeric value, the average rating of a recipe, and we are attempting to analyze and find a relationship between existing & engineered features and the average recipe rating using our model.

We chose average rating to be our response variable because we found it to be the most important and one of the most descriptive features in the dataset, which is arguably also the most crucial aspect in the perspective of a user of the website where our data is scraped from. Rating can tell the audience a lot about the recipe and whether or not it is worth their time and resources to recreate themselves. Ratings are crucial as it helps website users select which specific recipes are the most optimal. Additionally, we thought average rating was an interesting feature to explore as well because we wanted to observe what features contribute to lower/higher ratings, such as specific tags, nutrition content, and number of steps/length of the recipe. This information can be highly utilizable as well, as it can provide useful metrics to those who want to create recipes that warrant popularity by adjusting certain parameters such as number of steps, ingredients, etc.

The metrics that we are using to evaluate our model are RMSE, MAE and R-squared values. We chose these metrics because our model is a regression model, and these are some of the most commonly used and powerful metrics from regression models and analysis. We chose the R-squared value as a metric as it would help us evaluate how well the model would fit, which plays a crucial part in telling us how accurately and precisely our model will perform, and also how well it fits our data. However, R-squared alone does not tell us enough about the error of our model, which is why we use RMSE as our second metric. We opted for RMSE over metrics such as MSE, as MSE is very sensitive to outliers and is not as interpretable as RMSE. We opted to use MAE as well instead of MSE because MAE does not penalize large errors caused by outliers.

We made sure to use only information given to us by the data scraped from www.food.com to train our model, which was the information that was existent at the time of the ratings, which is our response variable. We did not use any additional new information from www.food.com to create new features or train our model, we only stuck to information and data existing at the time of the ratings, in order to get the most fair and precise predictions.


## Baseline Model

For our baseline model, we will create a linear regression model that uses the 
`minutes`, `n_ingredients`, `n_steps`, and `calories (#)` features of a recipe to predict the average rating of the recipe. 
All four features used in this model is quantitative, thus no categorical encoding is needed.  

The pipeline of our baseline model will be the following: 
1. Standard Scale all the features
2. Fit all the features into a linear regression model.

### Performance of our Baseline model
We will be evaluating the performance of our linear regression model with the $R^2$ score. A $R^2$ value of 1 indicates that our linear regression model perfectly predicts the average recipe ratings. A $R^2$ value of 0 indicates that our model doesn't explain any of the variability between the recipe's feature and the recipe's average rating. 

We will first conduct a train-test split with our recipe dataset, keeping 20% of our recipe as the test set, and the rest as out training set. Next, we will fit the training set into the linear regression model and evaluate the $R^2$ value of our mdoel on the test set.

Our linear regression model obtained a $R^2$ value of 9.88e-05, which is almost 0. This indicates that our model is quite poor and is unable to explain the relationship between the features of the recipe we selected and the average rating of each of these recipe.

Table of the metrics between using the mean as a the predictor vs our linear regression model 

| model | RMSE | $R^2$ |
|-----|-----|-----|
| mean average rating of training set |  0.6294382049132434 | 0
| Linear Regression Model | 0.6294239020014049 | $7.715*10^{-5}$
 
The metrics we used to determine the performance of our baseline model is RMSE and R-squared due to the reasoning explained in the above section. The RMSE of our baseline model is quite low, meaning that the model is not much better than if one was to simply just use the mean average as the prediction for all cases. The RMSE of predicting only mean average rating is the same as the RMSE of our model, meaning that the model is not learning too many beneficial patterns from the data, and is not much better at prediction than the overall average.

The R-squared value of the linear regression model is negative, meaning that there is no relationship between the variables as predicted by our model. The R-squared of predicting only the mean average is a large negative value, meaning that it is not a reliable predictor of our variable of interest, and even though our R-squared value of the linear regression model is extremely close to 0, along with being negative, it is still better than the R-squared value of predicting using only the mean.

The reason we compare the metrics with using our models as predictors as compared to the mean as predictors is because we want to observe whether our model is doing better than just simply using the overall average of the dataset as the predictor value. Seeing the difference between the 2 RMSE values and 2 R-Squared values (difference between the performance of the model vs. the simple overall mean) can help us gauge how accurate and precise our model is, and if it is truly assisting us in making predictions or behaving in a manner that tells us very little about the correlations in the data and patterns existing within it.

From what we can see, our baseline model is not good, as it barely performs better than simply using the mean as a predictor. Therefore, we can gauge that the existing features may not be sufficient in order to help our model make accurate predictions regarding the average rating, and we may have to engineer some features in order to help our model learn better. After getting an idea of how to move forward by creating this baseline, we will explore improvements in our model through feature engineering and utilization of various regressors to help us choose optimal hyperparameters to improve our model in the next section.
## Final Model 

### Final Model: Feature Engineer
In our final model, we featured engineered several new features. The features we engineered are the following: 

### Uniqueness Score 
The uniqueness score is a number that denotes how unique a 
recipe's ingredients are. The uniqueness of an ingredient is the number of times that ingredient appeared in a recipe. The uniqueness score 
is the average uniqueness of all the ingredients. The reason we added this feature is that we think there might be a correlation between how unique a recipe is the how someone would rate it. For instances, we expect foods such as oyster to be more unique than chicken and thus arouse a higher rating from people that cook an oyster recipe. 

### Frequency Encoding of Contributor ids
Frequency encoding encodes each category with the number of time 
that category appeared in our training data set. We will be encoding 
the `contributor_id` column such that each contributor will be encoded 
to the number of recipes that contributor made. The reason why we added 
this feature is that we expect contributors that have contributed many 
recipes before to be more experienced than contributors that contributed 
only a few recipe. We expect a more experience contributor to create a 
better tasting recipe, and hence arousing a higher rating. 

### One Hot Encoding of multiple features from the tags column
We created a set of binary columns from the following tags: 'dietary','easy', 'occasion','meat', 'main-dish', 'vegetables', 'healthy', 'inexpensive', 'north-american', 'beginner-cook'. The column is 1 if the recipe has the tag and 0 if it doesn't. For example, a recipe will have a 1 in the "meat" tag column if that recipe feature a meat in its ingredient. The reason why we added these tags is because we expect that 
the average rating is somewhat correlated to these tags. For instance, 
I expect a recipe with the "healthy" tag to be rated higher on average than those who don't.

### Square Root Transformation of Numerical Columns
Lastly, we noticed that all of the numerical columns in our dataset is right skewed. There are many outliers for each of the numerical columns. We decided to do a square root transformation to reduce the skewness of our data because the model we will be using, Gradient Boosting Regressor, is sensitive to outliers and skewness. 

### Final Model: Regression Model 
For our project, we used two regression model, Gradient Boosting Regressor, and Random Forest Regressor. Both models are ensemble methods, which are models created from multiple weaker models. They are also both tree based forest models. In both of these models, we conducted a grid search to find the optimal model. The optimal hyperparameters discovered by grid search is indicated below, alongsides the RMSE and MAE of test set for each model. 

| Models | Optimal Parameters | RMSE | MAE | $R^2$
|-----|-----|-----|-----|-----| 
|Gradient Boosting Regression| 'rfr__max_depth': 7, 'rfr__n_estimators': 140| 0.3877592466867749 | 0.45389555694978706 | 0.01332815505128604
|Random Forest Regression | 'rfr__max_depth': 7, 'rfr__n_estimators': 140 | 0.38688636535066573 | 0.4548144130254471 | 0.009034398351096762

Both model seems to be about the same in performance, while being significantly better than our baseline model. 

# Fairness Analysis

In our fairness analysis, we will be conducting a fairness analysis between recipes that have greater than 10 steps and recipes that have less than or equal to 10 steps. 
We will be binaring the recipes into two groups: `hard` recipes and `easy` recipes. We will categorize all recipes that have more than 10 steps to be considered as `hard` and recipes that have 10 or less steps to be `easy`. Then, we will conduct the fairness analysis. 

Our hypothesis are the following: 
- Null Hypothesis: Our model is fair. The RMSE between `hard` and `easy` recipes are the same, and any differences are due to random chance. 
- Alternative Hypothesis: Our model is unfair. The model's RMSE for `hard` recipes are greater the RMSE for `easy` recipes.  

The test statistics we chose for this analysis is the mean squared error, and the significance level we chose is 0.05. After running the permutation test, we got a p-value of 0.1448 > 0.05. Hence, we fail to reject the null hypothesis. We conclude that our model seems to be fair when predicting the average rating of recipes that are hard to make and recipes that are easy to make.

Visualization of our permutation test: 

<iframe src="fairnessplot.html" width=800 height=600 frameBorder=0></iframe>
