# reciperatingmodel
This is a project that works with data from www.food.com and creates a model to predict certain attributes in the dataset such as ratings. This project has been done in the DSC 80 course at UCSD.

# Recipe Rating Model


## Framing the Problem 
In this project, we will be answering the predictive problem: 

> Given a recipe, what should we predict the average rating to be? 

Since average rating is a continuous variable, we want to create a regression model to predict the average rating. 

We will be evaluating our model with the R squared value. The reason for evaluating the model using the rmse is that we want to penalize large errors by a bigger magnitude than smaller errors. 

At the time of making our prediction, we will only have the information about the recipe itself. We do not know how the public will interact with the recipe, so information such as the comments or ratings a recipe will get will not be available.

## Baseline Model

For our baseline model, we will create a linear regression model that uses the 
`minutes`, `n_ingredients`, `n_steps`, and `calories (#)` features of a recipe to predict the average rating of the recipe. 
All four features used in this model is quantitative, thus no categorical encoding is needed.  

The pipeline of our baseline model will be the following: 
1. Standard Scale all the features
2. Fit all the features into a linear regression model.

### Performance of our Baseline model
We will be evaluating the performance of our linear regression model with the $R^2$ score. A $R^2$ value of 1 indicates that our linear regression model perfectly predicts the average recipe ratings. A $R^2$ value of 0 indicates that our model is not better than just predicting the recipe using the mean average rating from the training set. 


We will first conduct a train-test split with our recipe dataset, keeping 20% of our recipe as the test set, and the rest as out training set. Next, we will fit the training set into the linear regression model and evaluate the $R^2$ value of our mdoel on the test set.

Our linear regression model obtained a $R^2$ value of 9.88e-05, which is almost 0. This indicates that our model is quite poor and is unable to explain the relationship between the features of the recipe we selected and the average rating of each of these recipe.
