# reciperatingmodel
This is a project that works with data from www.food.com and creates a model to predict certain attributes in the dataset such as ratings. This project has been done in the DSC 80 course at UCSD.

# Recipe Rating Model


## Framing the Problem 
In this project, we will be answering the predictive problem: 

> Given a recipe, what should we predict the average rating to be? 

Since average rating is a continuous variable, we want to create a regression model to predict the average rating. 

We will be evaluating our model with the R squared value. The reason for evaluating the model using the rmse is that we want to penalize large errors by a bigger magnitude than smaller errors. 

At the time of making our prediction, we will only have the information about the recipe itself. We do not know how the public will interact with the recipe, so information such as the comments or ratings a recipe will get will not be available.



***START ANTARA*** 

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




   ***START ANTARA***



   Our baseline model is a simple linear regression model, which uses the features “minutes”, “n_ingredients”, “n_steps”, and “calories (#)” . All four of these columns are numerical, meaning that all of our features are numeric. We chose to do this because when dealing with numerical data (that has inherent value within itself), there is not as much immediate encoding necessary, such as with categorical features and variables, and we wanted to observe our baseline with mostly raw data (part of our cleaning process extracting the caloric values from another column and making it its own column), before we created our final model which involves quite a bit of feature engineering and encoding. 

The performance of our model is as follows:
The RMSE of the test set:  0.6327189588354093
RMSE of predicting only mean average rating 0.6327613848568943
R-squared of linear regression model:  7.714844734196813e-05
R-squared of predicting only the mean average:  -5.0755078305451066e+29

The metrics we used to determine the performance of our baseline model is RMSE and R-squared due to the reasoning explained in the above section. The RMSE of our baseline model is quite low, meaning that the model is not much better than if one was to simply just use the mean average as the prediction for all cases. The RMSE of predicting only mean average rating is the same as the RMSE of our model,  meaning that the model is not learning too many beneficial patterns from the data, and is not much better at prediction than the overall average. 

The R-squared value of the linear regression model is negative, meaning that there is no relationship between the variables as predicted by our model. The R-squared of predicting only the mean average is a large negative value, meaning that it is not a reliable predictor of our variable of interest, and even though our R-squared value of the linear regression model is extremely close to 0, along with being negative, it is still better than the R-squared value of predicting using only the mean. 

The reason we compare the metrics with using our models as predictors as compared to the mean as predictors is because we want to observe whether our model is doing better than just simply using the overall average of the dataset as the predictor value. Seeing the difference between the 2 RMSE values and 2 R-Squared values (difference between the performance of the model vs. the simple overall mean) can help us gauge how accurate and precise our model is, and if it is truly assisting us in making predictions or behaving in a manner that tells us very little about the correlations in the data and patterns existing within it. 

From what we can see, our baseline model is not good, as it barely performs better than simply using the mean as a predictor. Therefore, we can gauge that the existing features may not be sufficient in order to help our model make accurate predictions regarding the average rating, and we may have to engineer some features in order to help our model learn better. After getting an idea of how to move forward by creating this baseline, we will explore improvements in our model through feature engineering and utilization of various regressors to help us choose optimal hyperparameters to improve our model in the next section. 


### Performance of our Baseline model
We will be evaluating the performance of our linear regression model with the $R^2$ score. A $R^2$ value of 1 indicates that our linear regression model perfectly predicts the average recipe ratings. A $R^2$ value of 0 indicates that our model is not better than just predicting the recipe using the mean average rating from the training set. 


We will first conduct a train-test split with our recipe dataset, keeping 20% of our recipe as the test set, and the rest as out training set. Next, we will fit the training set into the linear regression model and evaluate the $R^2$ value of our mdoel on the test set.

Our linear regression model obtained a $R^2$ value of 9.88e-05, which is almost 0. This indicates that our model is quite poor and is unable to explain the relationship between the features of the recipe we selected and the average rating of each of these recipe.


## Final Model








