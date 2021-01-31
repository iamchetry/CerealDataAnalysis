#setwd('/Users/iamchetry/Documents/UB_files/506/hw_2/')

#install.packages('caret')
#install.packages("Metrics")
#install.packages("leaps")
#install.packages("comprehenr")
#install.packages("e1071")
#install.packages("ISLR")
#install.packages("glmnet")
#install.packages("pls")

#-------- Loading Required Libraries ---------
library(Metrics)
library(caret)
library(leaps)
library(glue)
library(comprehenr)
library(e1071)
library(plyr)
library(ISLR)
library(glmnet)
library(pls)

#------------------------------ 1st Question -------------------------------------

data_ = read.csv('cereal.csv')

densityplot(~rating | mfr, data = data_, auto.key = list(space = "right"), 
            groups = mfr, main='Rating across MFR') # only one data point for A
densityplot(~rating | type, data = data_, auto.key = list(space = "right"), 
            groups = type, main='Rating across Type') # only 3 data points for H

print(dim(data_))
#---------- Pre-processing ------------
data_ = subset(subset(data_, mfr != 'A'), type != 'H')
data_ = subset(data_, select = -c(1, 3))
print(dim(data_))
attach(data_)

#--------- Scatter Plots with Rating ----------
par(mfrow = c(2, 2))
plot(calories, rating)
plot(sugars, rating)
plot(protein, rating)
plot(fiber, rating)

#--------- Stratified Samping into Train and Test based on MFR variable ---------
t = createDataPartition(mfr, p=0.65, list = FALSE)
train_ = data_[t, ]
test_ = data_[-t, ]

par(mfrow = c(2,1))
hist(train_$rating, main='Rating in Train data')
hist(test_$rating, main='Rating in Test data')

#--------- Linear Model ----------
model_ = lm(rating~., data = train_)
summary(model_)

training_predictions = predict(model_)
testing_predictions = predict(model_, newdata = test_)

print(mse(train_$rating, training_predictions)) # Training MSE
print(mse(test_$rating, testing_predictions)) # Testing MSE

#-------------- Subset Selection ---------------

best_sub = regsubsets(rating~., data = train_, nbest = 1, nvmax = 13,
                      method = "exhaustive") # Exhaustive SS
best_summary = summary(best_sub)


par(mfrow = c(2,1))
plot(best_summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l",
     main='For Exhaustive Subset Selection')
plot(best_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2",
     type = "l")

forward_sub = regsubsets(rating~., data = train_, nbest = 1, nvmax = 13,
                         method = "forward") #Forward SS
forward_summary = summary(forward_sub)

par(mfrow = c(2,1))
plot(forward_summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l",
     main='For Forward Subset Selection')
plot(forward_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted R^2",
     type = "l")


ratings_ = test_[c('rating')] # Actual Ratings
temp_test = cbind(rep(1, length(test_[, 1])), test_) # Creating Intercept Column
names(temp_test) = c('(Intercept)', names(test_)) 
print(head(temp_test))

test_dummy = model.matrix(~temp_test$mfr) # Creating Dummy Variables for MFR
test_dummy = subset(test_dummy, select = -c(1))
colnames(test_dummy) = to_vec(for(i in colnames(test_dummy)) 
  strsplit(i, '$', fixed = TRUE)[[1]][2])

temp_test = cbind(temp_test, test_dummy)

best_mse_list = list() # List created to store MSE for each Subset of Exhaustive SS

for (i in 1:13)
{
  coeff_ = coef(best_sub, id=i)
  print(coeff_)
  d = temp_test[c(names(coeff_))]
  print(head(d))
  predicted_rating = t(coeff_%*%t(d))
  
  e = mse(as.numeric(unlist(ratings_)), as.numeric(predicted_rating))
  best_mse_list[[glue('{i}')]] = e
}

forward_mse_list = list() # List created to store MSE for each Subset of Forward SS

for (i in 1:13)
{
  coeff_ = coef(forward_sub, id=i)
  print(coeff_)
  d = temp_test[c(names(coeff_))]
  print(head(d))
  predicted_rating = t(coeff_%*%t(d))
  
  e = mse(as.numeric(unlist(ratings_)), as.numeric(predicted_rating))
  forward_mse_list[[glue('{i}')]] = e
}

par(mfrow = c(2,1))
plot(unlist(best_mse_list), main = 'Test Performance : Exhaustive Subset Selection',
     xlab = 'No. of Variables', ylab = 'Test MSE', col='blue')
plot(unlist(forward_mse_list), main = 'Test Performance : Forward Subset Selection',
     xlab = 'No. of Variables', ylab = 'Test MSE', col='red')

print(coef(best_sub, id=9)) # Returning the Intercept and Coefficients for the Best Model


#----------------------------- 2nd Question ---------------------------------

d1 = load('zip.train.RData')
d2 = load('zip.test.RData')

zip_train = data.frame(zip.train)
zip_train = subset(zip_train, X1 == 2 | X1 == 3) # Slicing for Binary Classification

zip_test = data.frame(zip.test)
zip_test = subset(zip_test, X1 == 2 | X1 == 3) # Slicing for Binary Classification

par(mfrow = c(2,1))
plot(zip_train$X1, main = "2's and 3's Scatter Plot for Training", ylab = 'X1')
plot(zip_test$X1, main = "2's and 3's Scatter Plot for Testing", ylab = 'X1')

#---------- Linear Model -------------
model_lm = lm(X1~., data = zip_train)
summary(model_lm)
train_preds = predict(model_lm, type = 'response')
train_preds = ifelse(train_preds >= 2.5, 3, 2) # Train Prediction

test_preds = predict(model_lm, newdata = zip_test, type = 'response')
test_preds = ifelse(test_preds >= 2.5, 3, 2) # Test Prediction

train_actual = as.factor(zip_train$X1)
test_actual = as.factor(zip_test$X1)

train_preds = as.factor(train_preds)
test_preds = as.factor(test_preds)

tab_train = table(train_preds, train_actual)
tab_test = table(test_preds, test_actual)

#--------- Confusion Matrix to determine Accuracy ---------
conf_train = confusionMatrix(tab_train) 
conf_test = confusionMatrix(tab_test)

train_error = 1 - round(conf_train$overall['Accuracy'], 4) # Training Error
test_error = 1 - round(conf_test$overall['Accuracy'], 4) # Testing Error

print(train_error)
print(test_error)


#---------------------- K-Nearest Neighbor ------------------------

error_list_train = list() # List created to store Training Error for each value of K
error_list_test = list() # List created to store Testing Error for each value of K

require(class)
for (k in c(1, 3, 5, 7, 9, 11, 13, 15))
{
  KNN_train = knn(zip_train[-c(1)], zip_train[-c(1)], zip_train$X1, k) # Train Prediction
  KNN_test = knn(zip_train[-c(1)], zip_test[-c(1)], zip_train$X1, k) # Test Prediction
  
  train_predicted = as.factor(KNN_train)
  test_predicted = as.factor(KNN_test)
  
  train_actual = as.factor(zip_train$X1)
  test_actual = as.factor(zip_test$X1)
  
  tab_train = table(train_predicted, train_actual)
  tab_test = table(test_predicted, test_actual)
  
  #-------- Confusion Matrix for Accuracy ---------
  conf_train = confusionMatrix(tab_train)
  conf_test = confusionMatrix(tab_test)
  
  error_list_train[[glue('{k}')]] = 1 - round(
    conf_train$overall['Accuracy'], 4)
  error_list_test[[glue('{k}')]] = 1 - round(
    conf_test$overall['Accuracy'], 4)
  
}

#------------ Plotting Errors for all values of K ---------------
par(mfrow = c(2,1))
v = unlist(error_list_train)
names(v) = to_vec(for(i in names(v)) 
  strsplit(i, '.', fixed = TRUE)[[1]][1])
plot(as.numeric(names(v)), v, xaxt="n", col='blue', main = 'Train Error',
     xlab = 'Values of K', ylab = 'Error')
axis(1, at = seq(1, 15, by = 2), las=2)

v = unlist(error_list_test)
names(v) = to_vec(for(i in names(v)) 
  strsplit(i, '.', fixed = TRUE)[[1]][1])
plot(as.numeric(names(v)), v, xaxt="n", col='red', main = 'Test Error',
     xlab = 'Values of K', ylab = 'Error')
axis(1, at = seq(1, 15, by = 2), las=2)



#----------------------------- 3rd Question ----------------------------------

data_ = College
data_ = na.omit(data_)

#---------- Transformations -------------
data_$Top10perc = data_$Top10perc/100
data_$Top25perc = data_$Top25perc/100
data_$PhD = data_$PhD/100
data_$Terminal = data_$Terminal/100
data_$perc.alumni = data_$perc.alumni/100
data_$Grad.Rate = data_$Grad.Rate/100
attach(data_)

#----------- Stratified Sampling based on Private variable --------------
t = createDataPartition(Private, p=0.65, list = FALSE)
train_ = data_[t, ]
test_ = data_[-t, ]

#--------- Linear Model ---------
lin_model = lm(Apps~., data = train_)
summary(lin_model)

testing_predictions = predict(lin_model, newdata = test_)

print(rmse(test_$Apps, testing_predictions)) # Test RMSE

#------------ Data Splitting -------------
X = model.matrix(~., subset(train_, select = -c(2)))
y = train_$Apps
X_test = model.matrix(~., subset(test_, select = -c(2)))

#------------ Ridge Regression --------------
ridge_model = glmnet(X, y, alpha = 0)
ridge_out = cv.glmnet(X, y, alpha = 0)

bestlam = ridge_out$lambda.min
print(bestlam)

ridge_predictions = predict(ridge_model, s = bestlam, newx = X_test,
                            type = "response")

print(rmse(test_$Apps, ridge_predictions)) # Test RMSE

#-------------- LASSO Regression ---------------
lasso_model = glmnet(X, y, alpha = 1)
lasso_out = cv.glmnet(X, y, alpha = 1)

bestlam = lasso_out$lambda.min
print(bestlam)

lasso_predictions = predict(lasso_model, s = bestlam, newx = X_test,
                            type = "response")

print(rmse(test_$Apps, lasso_predictions)) # Test RMSE

par(mfrow = c(2,1))
plot(ridge_out, main='Ridge')
plot(lasso_out, main='LASSO')

#------------- PCR --------------
pcr_model = pcr(Apps~., data=train_, scale=TRUE, validation='CV')
print(summary(pcr_model))
pcr_predictions = predict(pcr_model, test_,ncomp=9) #90% of the Variance Captured
print(rmse(test_$Apps, pcr_predictions)) # Test RMSE


#------------- PLS --------------
pls_model = plsr(Apps~., data=train_, scale=TRUE, validation='CV')
print(summary(pls_model))
pls_predictions = predict(pls_model, test_,ncomp=13) #90% of the Variance Captured
print(rmse(test_$Apps, pls_predictions)) # Test RMSE


#-------PCR and PLS Validation Plots ---------
par(mfrow = c(2,1))
validationplot(pcr_model,val.type = "MSEP", main='PCR Validation Plot')
validationplot(pls_model,val.type = "MSEP", main='PLS Validation Plot')
