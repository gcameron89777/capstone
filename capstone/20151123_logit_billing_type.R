## load libraries, import data and lite data preparation
library("klaR") # not sure, from lecture notes
library("caret") # Naive Bayes and confusion matrix
library("e1071") # goes with caret
library("ROCR") # for ROC curve and auc

# import data
data <- read.csv("churn_data.csv")
data <- data[complete.cases(data),] # 11 records incomplete, minimal in comparison to whole dataset so safe to remove while keeping enough data for analysis

## split data between existing 2 class features, multi class features and numeric features to be pre processed into binary

# identify dichotomous variables then turn into binary numeric
# Dichotomous vars
twolevel_vars <- data[c("Churn", "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling")]

# Binarify
twolevel_vars$NumChurn <- 0
twolevel_vars$NumChurn[twolevel_vars$Churn == "Yes"] <- 1

twolevel_vars$NumGender <- 0
twolevel_vars$NumGender[twolevel_vars$gender == "Female"] <- 1 

twolevel_vars$NumPartner <- 0
twolevel_vars$NumPartner[twolevel_vars$Partner == "Yes"] <- 1 

twolevel_vars$NumDependants <- 0
twolevel_vars$NumDependants[twolevel_vars$Dependents == "Yes"] <- 1 

twolevel_vars$NumPhoneService <- 0
twolevel_vars$NumPhoneService[twolevel_vars$PhoneService == "Yes"] <- 1

twolevel_vars$NumPaperlessBilling <- 0
twolevel_vars$NumPaperlessBilling[twolevel_vars$PaperlessBilling == "Yes"] <- 1

# multi level vars to be matrixed
multilevel_vars <- data[c("PaymentMethod", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                          "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract")]

# numeric vars to be transformed to integer since R Naive Bayes seems to want this and not discreet numbers
numeric_vars <- data[c("tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen")]

# Convert multi level vars to matrix' (n - 1 since final var from each new matrix can be inferred already)
binary <- cbind(
  # from multi level vars
  with(multilevel_vars, model.matrix(~ PaymentMethod + 0))[,-1],
  with(multilevel_vars, model.matrix(~ MultipleLines + 0))[,-1],
  with(multilevel_vars, model.matrix(~ InternetService + 0))[,-1],
  with(multilevel_vars, model.matrix(~ OnlineSecurity + 0))[,-1],
  with(multilevel_vars, model.matrix(~ OnlineBackup + 0))[,-1],
  with(multilevel_vars, model.matrix(~ DeviceProtection + 0))[,-1],
  with(multilevel_vars, model.matrix(~ TechSupport + 0))[,-1],
  with(multilevel_vars, model.matrix(~ StreamingTV + 0))[,-1],
  with(multilevel_vars, model.matrix(~ StreamingMovies + 0))[,-1],
  with(multilevel_vars, model.matrix(~ Contract + 0))[,-1]
)

binary <- as.data.frame(binary)

## preprocessed data

# combine component pieces into master data frame for analysis
pdata <- cbind(twolevel_vars, binary, numeric_vars)

# these vars have new, numeric versions added so remove the original factors versions
pdata <- subset(pdata,,-c(Churn, gender, Partner, Dependents, PhoneService, PaperlessBilling))

# by making into integer you can run different models through the pipeline
pdata <- lapply(pdata, as.integer)

# target variable needs to be a class (factor)
pdata$NumChurn <- as.factor(pdata$NumChurn)

pdata <- as.data.frame(pdata)

### this area for experimenting with variables
## try creating categories of variable

target <- subset(pdata,,c(NumChurn))

demographic_vars <- subset(pdata,,c(NumGender, 
                                    NumPartner, 
                                    NumDependants, 
                                    SeniorCitizen
))

usage_engagement_vars <- subset(pdata,,c(MultipleLinesYes,
                                         NumPhoneService, 
                                         MultipleLinesNo.phone.service, 
                                         InternetServiceFiber.optic, 
                                         InternetServiceNo,
                                         OnlineSecurityNo.internet.service, 
                                         OnlineSecurityYes, 
                                         OnlineBackupNo.internet.service, 
                                         OnlineBackupYes, 
                                         DeviceProtectionNo.internet.service,  
                                         DeviceProtectionYes, 
                                         TechSupportNo.internet.service, 
                                         TechSupportYes, 
                                         StreamingTVNo.internet.service, 
                                         StreamingTVYes,
                                         StreamingMoviesNo.internet.service, 
                                         StreamingMoviesYes,tenure, 
                                         MonthlyCharges, 
                                         TotalCharges
))

billing_contracttype_vars <- subset(pdata,,c(NumPaperlessBilling, 
                                             PaymentMethodCredit.card..automatic., 
                                             PaymentMethodElectronic.check, 
                                             PaymentMethodMailed.check
))

contract_type_vars <- subset(pdata,,c(ContractOne.year, 
                                      ContractTwo.year))

mylist <- list(demographic_vars, usage_engagement_vars, billing_contracttype_vars, contract_type_vars)

# redefine pdata based on chosen vars

# each item being cbinded here is a dataframe of features.
pdata <- cbind(target, billing_contracttype_vars)

## create data frames for holding predictions and actuals from k folds cross validation
cv_prediction <- data.frame()
testsetCopy <- data.frame()

## cross validation (10 folds by default)
set.seed(123) # make results reproducible
folds <- createFolds(pdata$NumChurn)

# cross validation
for ( f in folds ) {
  train <- pdata[-f,]
  test <- pdata[f,]
  
  model <- glm(NumChurn ~. , data=train, family = "binomial")
  predictions <- predict(model, test, type='response')
  predictions <- ifelse(predictions > 0.5,1,0)
  
  temp <- as.data.frame(predictions)
  
  cv_prediction <- rbind(cv_prediction, temp)
  testsetCopy <- rbind(testsetCopy, test)
}

## testing
# confusion matrix
confusionMatrix(testsetCopy$NumChurn, cv_prediction$predictions, positive="1")

# ROC curve
pred <- prediction(cv_prediction$predictions, testsetCopy$NumChurn)
perf <- performance(pred, "tpr", "fpr")
plot(perf, lty=1)
abline(a=0, b=1)

# AUC
auc.tmp <- performance(pred,"auc"); auc <- as.numeric(auc.tmp@y.values)
auc