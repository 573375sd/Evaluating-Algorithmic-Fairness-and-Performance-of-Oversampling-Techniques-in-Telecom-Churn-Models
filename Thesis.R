setwd("~/BAM/Thesis/Data")
library(readxl)
library(dplyr)
library(fastDummies)
library(ROSE)
library(themis)
library(tidymodels)
library(tidyverse)
library(skimr)
library(ggridges)
library(vip)
library(xgboost)
set.seed(14)

# Define metrics
calculate_fair_utility <- function(data, group_column, target_column, prediction_column) {
  privileged_group <- data %>% filter(.data[[group_column]] == 1)
  unprivileged_group <- data %>% filter(.data[[group_column]] == 0)
  TPR_priv <- yardstick::sens(privileged_group, truth = all_of(target_column), estimate = all_of(prediction_column))$.estimate
  TNR_priv <- yardstick::spec(privileged_group, truth = all_of(target_column), estimate = all_of(prediction_column))$.estimate
  TPR_unpriv <- yardstick::sens(unprivileged_group, truth = all_of(target_column), estimate = all_of(prediction_column))$.estimate
  TNR_unpriv <- yardstick::spec(unprivileged_group, truth = all_of(target_column), estimate = all_of(prediction_column))$.estimate
  TPRD <- TPR_priv - TPR_unpriv
  TNRD <- TNR_priv - TNR_unpriv
  BA <- 0.5 * (TPR_priv + TNR_priv)
  Fair_Utility <- BA * 1/2 * ((1 - abs(TPRD)) + (1 - abs(TNRD)))
  return(Fair_Utility)
}

calculate_spd <- function(data, group_column, target_column, prediction_column) {
  rate_group1 <- mean(data[[prediction_column]] == 1 & data[[group_column]] == 1) 
  rate_group2 <- mean(data[[prediction_column]] == 1 & data[[group_column]] == 0)
  SPD <- abs(rate_group1 - rate_group2)
  return(SPD)
}

calculate_eod <- function(data, group_column, target_column, prediction_column) {
  data_group1 <- data[data[[group_column]] == 1, ]
  data_group2 <- data[data[[group_column]] == 0, ]
  TPR_group1 <- yardstick::sens(data_group1, truth = all_of(target_column), estimate = all_of(prediction_column))$.estimate
  TPR_group2 <- yardstick::sens(data_group2, truth = all_of(target_column), estimate = all_of(prediction_column))$.estimate
  FPR_group1 <- 1 - yardstick::spec(data_group1, truth = all_of(target_column), estimate = all_of(prediction_column))$.estimate
  FPR_group2 <- 1 - yardstick::spec(data_group2, truth = all_of(target_column), estimate = all_of(prediction_column))$.estimate
  EOD_TPR <- abs(TPR_group1 - TPR_group2) 
  EOD_FPR <- abs(FPR_group1 - FPR_group2) 
  EOD <- EOD_TPR + EOD_FPR
  return(EOD)
}

# PART 1: TELCO DATA
telco <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", header = TRUE, sep = ",")

telco |>
  count(Churn) |>
  mutate(prop = n / sum(n))
telco |>
  count(gender) |>
  mutate(prop = n / sum(n))
telco |>
  count(SeniorCitizen) |>
  mutate(prop = n / sum(n))
telco |>
  count(Partner) |>
  mutate(prop = n / sum(n))
telco |>
  count(Dependents) |>
  mutate(prop = n / sum(n))

summary(telco)
sum(is.na(telco))

# Telco data processing
tell_bin <- c("Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn")
telco[tell_bin] <- lapply(telco[tell_bin], function(x) ifelse(x == "Yes", 1, ifelse(x == "No", 0, x)))
tell_mbin <- c("MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")
telco[tell_mbin] <- lapply(telco[tell_mbin], function(x) ifelse(x == "Yes", 1, 0))
telco <- fastDummies::dummy_cols(telco, 
                                 select_columns = c("InternetService", "Contract", "PaymentMethod"),
                                 remove_first_dummy = TRUE,  
                                 remove_selected_columns = TRUE)
telco <- fastDummies::dummy_cols(telco, 
                                 select_columns = c("gender"),
                                 remove_first_dummy = TRUE,  
                                 remove_selected_columns = TRUE)
tell_nrm <- c("tenure", "MonthlyCharges", "TotalCharges")
telco[tell_nrm] <- lapply(telco[tell_nrm], function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
})
telco$TotalCharges[is.na(telco$TotalCharges) & telco$tenure == 0] <- 0
telco <- telco %>% select(-customerID)
colnames(telco) <- gsub(" ", "_", colnames(telco))
colnames(telco) <- gsub("_\\(automatic\\)", "", colnames(telco))
sum(is.na(telco))

telco$Churn <- factor(telco$Churn, levels = c("0", "1"))
telco <- telco  
for (col in names(telco)) {
  if (is.character(telco[[col]])) {
    telco[[col]] <- as.numeric(as.character(telco[[col]]))
  }
}

# Unbalanced
telco_UB <- telco

# ROS
tellN <- sum(telco$Churn == 0) * 2
telco_ROS <- ovun.sample(Churn ~ ., data = telco, method = "over", N = tellN)$data

# SMOTE
telco_SMOTE <- smote(telco, "Churn", k=5, over_ratio = 1)

# ADASYN
telco_ADASYN <- adasyn(telco, "Churn", k=5, over_ratio = 1)

# Training and testing splits LR
telco_UB_split <- initial_validation_split(telco_UB, prop = c(.4, .4), strata = Churn)
telco_UB_train <- training(telco_UB_split)
telco_UB_validation <- validation(telco_UB_split)
telco_UB_test <- testing(telco_UB_split)

telco_ROS_split <- initial_validation_split(telco_ROS, prop = c(.4, .4), strata = Churn)
telco_ROS_train <- training(telco_ROS_split)
telco_ROS_validation <- validation(telco_ROS_split)
telco_ROS_test <- testing(telco_ROS_split)

telco_SMOTE_split <- initial_validation_split(telco_SMOTE, prop = c(.4, .4), strata = Churn)
telco_SMOTE_train <- training(telco_SMOTE_split)
telco_SMOTE_validation <- validation(telco_SMOTE_split)
telco_SMOTE_test <- testing(telco_SMOTE_split)

telco_ADASYN_split <- initial_validation_split(telco_ADASYN, prop = c(.4, .4), strata = Churn)
telco_ADASYN_train <- training(telco_ADASYN_split)
telco_ADASYN_validation <- validation(telco_ADASYN_split)
telco_ADASYN_test <- testing(telco_ADASYN_split)

# LR
telco_UB_lr <- logistic_reg() |> set_mode("classification") |> set_engine("glm")
telco_UB_lr_recipe <- recipe(Churn ~ ., data = telco_UB_train )
telco_UB_lr_workflow <- workflow() |> add_model(telco_UB_lr) |> add_recipe(telco_UB_lr_recipe)
telco_UB_lr_last_fit <- telco_UB_lr_workflow |> last_fit(telco_UB_split, metrics = metric_set(roc_auc, accuracy), add_validation_set = TRUE)

telco_ROS_lr <- logistic_reg() |> set_mode("classification") |> set_engine("glm")
telco_ROS_lr_recipe <- recipe(Churn ~ ., data = telco_ROS_train )
telco_ROS_lr_workflow <- workflow() |> add_model(telco_ROS_lr) |> add_recipe(telco_ROS_lr_recipe)
telco_ROS_lr_last_fit <- telco_ROS_lr_workflow |> last_fit(telco_ROS_split, metrics = metric_set(roc_auc, accuracy), add_validation_set = TRUE)

telco_SMOTE_lr <- logistic_reg() |> set_mode("classification") |> set_engine("glm")
telco_SMOTE_lr_recipe <- recipe(Churn ~ ., data = telco_SMOTE_train )
telco_SMOTE_lr_workflow <- workflow() |> add_model(telco_SMOTE_lr) |> add_recipe(telco_SMOTE_lr_recipe)
telco_SMOTE_lr_last_fit <- telco_SMOTE_lr_workflow |> last_fit(telco_SMOTE_split, metrics = metric_set(roc_auc, accuracy), add_validation_set = TRUE)

telco_ADASYN_lr <- logistic_reg() |> set_mode("classification") |> set_engine("glm")
telco_ADASYN_lr_recipe <- recipe(Churn ~ ., data = telco_ADASYN_train )
telco_ADASYN_lr_workflow <- workflow() |> add_model(telco_ADASYN_lr) |> add_recipe(telco_ADASYN_lr_recipe)
telco_ADASYN_lr_last_fit <- telco_ADASYN_lr_workflow |> last_fit(telco_ADASYN_split, metrics = metric_set(roc_auc, accuracy), add_validation_set = TRUE)

# LR Predictive Performance
telco_UB_test_pred <- telco_UB_lr_last_fit |> collect_predictions() |> select(-Churn) |> bind_cols(telco_UB_test)
telco_UB_lr_f1 <- f_meas(telco_UB_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_UB_lr_mcc <- mcc(telco_UB_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_UB_test_pred$.pred_class <- as.numeric(levels(telco_UB_test_pred$.pred_class))[telco_UB_test_pred$.pred_class]
telco_UB_lr_auc <- roc_auc(telco_UB_test_pred, truth = "Churn", ".pred_0")$.estimate

telco_ROS_test_pred <- telco_ROS_lr_last_fit |> collect_predictions() |> select(-Churn) |> bind_cols(telco_ROS_test)
telco_ROS_lr_f1 <- f_meas(telco_ROS_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_ROS_lr_mcc <- mcc(telco_ROS_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_ROS_test_pred$.pred_class <- as.numeric(levels(telco_ROS_test_pred$.pred_class))[telco_ROS_test_pred$.pred_class]
telco_ROS_lr_auc <- roc_auc(telco_ROS_test_pred, truth = "Churn", ".pred_0")$.estimate

telco_SMOTE_test_pred <- telco_SMOTE_lr_last_fit |> collect_predictions() |> select(-Churn) |> bind_cols(telco_SMOTE_test)
telco_SMOTE_lr_f1 <- f_meas(telco_SMOTE_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_SMOTE_lr_mcc <- mcc(telco_SMOTE_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_SMOTE_test_pred$.pred_class <- as.numeric(levels(telco_SMOTE_test_pred$.pred_class))[telco_SMOTE_test_pred$.pred_class]
telco_SMOTE_lr_auc <- roc_auc(telco_SMOTE_test_pred, truth = "Churn", ".pred_0")$.estimate

telco_ADASYN_test_pred <- telco_ADASYN_lr_last_fit |> collect_predictions() |> select(-Churn) |> bind_cols(telco_ADASYN_test)
telco_ADASYN_lr_f1 <- f_meas(telco_ADASYN_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_ADASYN_lr_mcc <- mcc(telco_ADASYN_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_ADASYN_test_pred$.pred_class <- as.numeric(levels(telco_ADASYN_test_pred$.pred_class))[telco_ADASYN_test_pred$.pred_class]
telco_ADASYN_lr_auc <- roc_auc(telco_ADASYN_test_pred, truth = "Churn", ".pred_0")$.estimate

# LR Fair utility
telco_UB_test_pred$.pred_class <- as.factor(telco_UB_test_pred$.pred_class)
FU_telco_UB_lr_gender <- calculate_fair_utility(
  data = telco_UB_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_UB_lr_age <- calculate_fair_utility(
  data = telco_UB_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_UB_lr_dep <- calculate_fair_utility(
  data = telco_UB_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

telco_ROS_test_pred$.pred_class <- as.factor(telco_ROS_test_pred$.pred_class)
FU_telco_ROS_lr_gender <- calculate_fair_utility(
  data = telco_ROS_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_ROS_lr_age <- calculate_fair_utility(
  data = telco_ROS_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_ROS_lr_dep <- calculate_fair_utility(
  data = telco_ROS_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

telco_SMOTE_test_pred$.pred_class <- as.factor(telco_SMOTE_test_pred$.pred_class)
FU_telco_SMOTE_lr_gender <- calculate_fair_utility(
  data = telco_SMOTE_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_SMOTE_lr_age <- calculate_fair_utility(
  data = telco_SMOTE_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_SMOTE_lr_dep <- calculate_fair_utility(
  data = telco_SMOTE_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

telco_ADASYN_test_pred$.pred_class <- as.factor(telco_ADASYN_test_pred$.pred_class)
FU_telco_ADASYN_lr_gender <- calculate_fair_utility(
  data = telco_ADASYN_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_ADASYN_lr_age <- calculate_fair_utility(
  data = telco_ADASYN_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_ADASYN_lr_dep <- calculate_fair_utility(
  data = telco_ADASYN_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# LR SPD
SPD_telco_UB_lr_gender <- calculate_spd(
  data = telco_UB_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_UB_lr_age <- calculate_spd(
  data = telco_UB_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_UB_lr_dep <- calculate_spd(
  data = telco_UB_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ROS_lr_gender <- calculate_spd(
  data = telco_ROS_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ROS_lr_age <- calculate_spd(
  data = telco_ROS_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ROS_lr_dep <- calculate_spd(
  data = telco_ROS_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_SMOTE_lr_gender <- calculate_spd(
  data = telco_SMOTE_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_SMOTE_lr_age <- calculate_spd(
  data = telco_SMOTE_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_SMOTE_lr_dep <- calculate_spd(
  data = telco_SMOTE_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ADASYN_lr_gender <- calculate_spd(
  data = telco_ADASYN_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ADASYN_lr_age <- calculate_spd(
  data = telco_ADASYN_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ADASYN_lr_dep <- calculate_spd(
  data = telco_ADASYN_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# LR EOD
EOD_telco_UB_lr_gender <- calculate_eod(
  data = telco_UB_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_UB_lr_age <- calculate_eod(
  data = telco_UB_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_UB_lr_dep <- calculate_eod(
  data = telco_UB_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ROS_lr_gender <- calculate_eod(
  data = telco_ROS_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ROS_lr_age <- calculate_eod(
  data = telco_ROS_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ROS_lr_dep <- calculate_eod(
  data = telco_ROS_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_SMOTE_lr_gender <- calculate_eod(
  data = telco_SMOTE_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_SMOTE_lr_age <- calculate_eod(
  data = telco_SMOTE_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_SMOTE_lr_dep <- calculate_eod(
  data = telco_SMOTE_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ADASYN_lr_gender <- calculate_eod(
  data = telco_ADASYN_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ADASYN_lr_age <- calculate_eod(
  data = telco_ADASYN_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ADASYN_lr_dep <- calculate_eod(
  data = telco_ADASYN_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# Overview LR 
evaluation_results_LR_telco <- data.frame(
  Model = c("UB", "ROS", "SMOTE", "ADASYN"),
  F1_Score = c(telco_UB_lr_f1, telco_ROS_lr_f1, telco_SMOTE_lr_f1, telco_ADASYN_lr_f1),
  MCC = c(telco_UB_lr_mcc, telco_ROS_lr_mcc, telco_SMOTE_lr_mcc, telco_ADASYN_lr_mcc),
  AUC = c(telco_UB_lr_auc, telco_ROS_lr_auc, telco_SMOTE_lr_auc, telco_ADASYN_lr_auc),
  SPD_gender = c(SPD_telco_UB_lr_gender, SPD_telco_ROS_lr_gender, SPD_telco_SMOTE_lr_gender, SPD_telco_ADASYN_lr_gender),
  EOD_gender = c(EOD_telco_UB_lr_gender, EOD_telco_ROS_lr_gender, EOD_telco_SMOTE_lr_gender, EOD_telco_ADASYN_lr_gender),
  Fair_Utility_gender = c(FU_telco_UB_lr_gender, FU_telco_ROS_lr_gender, FU_telco_SMOTE_lr_gender, FU_telco_ADASYN_lr_gender),
  SPD_age = c(SPD_telco_UB_lr_age, SPD_telco_ROS_lr_age, SPD_telco_SMOTE_lr_age, SPD_telco_ADASYN_lr_age),
  EOD_age = c(EOD_telco_UB_lr_age, EOD_telco_ROS_lr_age, EOD_telco_SMOTE_lr_age, EOD_telco_ADASYN_lr_age),
  Fair_Utility_age = c(FU_telco_UB_lr_age, FU_telco_ROS_lr_age, FU_telco_SMOTE_lr_age, FU_telco_ADASYN_lr_age),
  SPD_dep = c(SPD_telco_UB_lr_dep, SPD_telco_ROS_lr_dep, SPD_telco_SMOTE_lr_dep, SPD_telco_ADASYN_lr_dep),
  EOD_dep = c(EOD_telco_UB_lr_dep, EOD_telco_ROS_lr_dep, EOD_telco_SMOTE_lr_dep, EOD_telco_ADASYN_lr_dep),
  Fair_Utility_dep = c(FU_telco_UB_lr_dep, FU_telco_ROS_lr_dep, FU_telco_SMOTE_lr_dep, FU_telco_ADASYN_lr_dep)
)

# Training and testing splits Rf
telco_UB_split2 <- initial_split(telco_UB, prop = 0.8, strata = Churn)
telco_UB_train2 <- training(telco_UB_split2)
telco_UB_test2 <- testing(telco_UB_split2)
cv_folds_telco_UB_train <- telco_UB_train2 |> vfold_cv(v = 10, repeats = 5, strata = "Churn")

telco_ROS_split2 <- initial_split(telco_ROS, prop = 0.8, strata = Churn)
telco_ROS_train2 <- training(telco_ROS_split2)
telco_ROS_test2 <- testing(telco_ROS_split2)
cv_folds_telco_ROS_train <- telco_ROS_train2 |> vfold_cv(v = 10, repeats = 5, strata = "Churn")

telco_SMOTE_split2 <- initial_split(telco_SMOTE, prop = 0.8, strata = Churn)
telco_SMOTE_train2 <- training(telco_SMOTE_split2)
telco_SMOTE_test2 <- testing(telco_SMOTE_split2)
cv_folds_telco_SMOTE_train <- telco_SMOTE_train2 |> vfold_cv(v = 10, repeats = 5, strata = "Churn")

telco_ADASYN_split2 <- initial_split(telco_ADASYN, prop = 0.8, strata = Churn)
telco_ADASYN_train2 <- training(telco_ADASYN_split2)
telco_ADASYN_test2 <- testing(telco_ADASYN_split2)
cv_folds_telco_ADASYN_train <- telco_ADASYN_train2 |> vfold_cv(v = 10, repeats = 5, strata = "Churn")

# RF
class_metrics <- metric_set(accuracy, roc_auc)

telco_UB_rf <- rand_forest(mtry = tune(), trees = 100) |> set_mode("classification") 
telco_UB_rf_recipe <- recipe(Churn ~ ., data = telco_UB_train2)
telco_UB_rf_workflow <- workflow() |> add_model(telco_UB_rf) |> add_recipe(telco_UB_rf_recipe)
rf_tune_grid <- grid_regular(mtry(range = c(1, 20)), levels = 10)
telco_UB_rf_tune <- tune_grid(telco_UB_rf_workflow, resamples = cv_folds_telco_UB_train, grid = rf_tune_grid, metrics = class_metrics)
telco_UB_rf_final_model <- telco_UB_rf_tune |> select_by_one_std_err(mtry, metric = "roc_auc")
telco_UB_rf_final_wf <- finalize_workflow(telco_UB_rf_workflow, telco_UB_rf_final_model)
telco_UB_rf_final_fit <- telco_UB_rf_final_wf |> last_fit(telco_UB_split2, metrics = class_metrics)

telco_ROS_rf <- rand_forest(mtry = tune(), trees = 100) |> set_mode("classification") 
telco_ROS_rf_recipe <- recipe(Churn ~ ., data = telco_ROS_train2)
telco_ROS_rf_workflow <- workflow() |> add_model(telco_ROS_rf) |> add_recipe(telco_ROS_rf_recipe)
rf_tune_grid <- grid_regular(mtry(range = c(1, 20)), levels = 10)
telco_ROS_rf_tune <- tune_grid(telco_ROS_rf_workflow, resamples = cv_folds_telco_ROS_train, grid = rf_tune_grid, metrics = class_metrics)
telco_ROS_rf_final_model <- telco_ROS_rf_tune |> select_by_one_std_err(mtry, metric = "roc_auc")
telco_ROS_rf_final_wf <- finalize_workflow(telco_ROS_rf_workflow, telco_ROS_rf_final_model)
telco_ROS_rf_final_fit <- telco_ROS_rf_final_wf |> last_fit(telco_ROS_split2, metrics = class_metrics)
telco_SMOTE_rf <- rand_forest(mtry = tune(), trees = 100) |> set_mode("classification") 
telco_SMOTE_rf_recipe <- recipe(Churn ~ ., data = telco_SMOTE_train2)
telco_SMOTE_rf_workflow <- workflow() |> add_model(telco_SMOTE_rf) |> add_recipe(telco_SMOTE_rf_recipe)
rf_tune_grid <- grid_regular(mtry(range = c(1, 20)), levels = 10)
telco_SMOTE_rf_tune <- tune_grid(telco_SMOTE_rf_workflow, resamples = cv_folds_telco_SMOTE_train, grid = rf_tune_grid, metrics = class_metrics)
telco_SMOTE_rf_final_model <- telco_SMOTE_rf_tune |> select_by_one_std_err(mtry, metric = "roc_auc")
telco_SMOTE_rf_final_wf <- finalize_workflow(telco_SMOTE_rf_workflow, telco_SMOTE_rf_final_model)
telco_SMOTE_rf_final_fit <- telco_SMOTE_rf_final_wf |> last_fit(telco_SMOTE_split2, metrics = class_metrics)

telco_ADASYN_rf <- rand_forest(mtry = tune(), trees = 100) |> set_mode("classification") 
telco_ADASYN_rf_recipe <- recipe(Churn ~ ., data = telco_ADASYN_train2)
telco_ADASYN_rf_workflow <- workflow() |> add_model(telco_ADASYN_rf) |> add_recipe(telco_ADASYN_rf_recipe)
rf_tune_grid <- grid_regular(mtry(range = c(1, 20)), levels = 10)
telco_ADASYN_rf_tune <- tune_grid(telco_ADASYN_rf_workflow, resamples = cv_folds_telco_ADASYN_train, grid = rf_tune_grid, metrics = class_metrics)
telco_ADASYN_rf_final_model <- telco_ADASYN_rf_tune |> select_by_one_std_err(mtry, metric = "roc_auc")
telco_ADASYN_rf_final_wf <- finalize_workflow(telco_ADASYN_rf_workflow, telco_ADASYN_rf_final_model)
telco_ADASYN_rf_final_fit <- telco_ADASYN_rf_final_wf |> last_fit(telco_ADASYN_split2, metrics = class_metrics)

# RF predictive performance
telco_UB_rf_test_pred <- telco_UB_rf_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(telco_UB_test2)
telco_UB_rf_f1 <- f_meas(telco_UB_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_UB_rf_mcc <- mcc(telco_UB_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_UB_rf_test_pred$.pred_class <- as.numeric(levels(telco_UB_rf_test_pred$.pred_class))[telco_UB_rf_test_pred$.pred_class]
telco_UB_rf_auc <- roc_auc(telco_UB_rf_test_pred, truth = "Churn", ".pred_0")$.estimate

telco_ROS_rf_test_pred <- telco_ROS_rf_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(telco_ROS_test2)
telco_ROS_rf_f1 <- f_meas(telco_ROS_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_ROS_rf_mcc <- mcc(telco_ROS_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_ROS_rf_test_pred$.pred_class <- as.numeric(levels(telco_ROS_rf_test_pred$.pred_class))[telco_ROS_rf_test_pred$.pred_class]
telco_ROS_rf_auc <- roc_auc(telco_ROS_rf_test_pred, truth = "Churn", ".pred_0")$.estimate

telco_SMOTE_rf_test_pred <- telco_SMOTE_rf_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(telco_SMOTE_test2)
telco_SMOTE_rf_f1 <- f_meas(telco_SMOTE_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_SMOTE_rf_mcc <- mcc(telco_SMOTE_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_SMOTE_rf_test_pred$.pred_class <- as.numeric(levels(telco_SMOTE_rf_test_pred$.pred_class))[telco_SMOTE_rf_test_pred$.pred_class]
telco_SMOTE_rf_auc <- roc_auc(telco_SMOTE_rf_test_pred, truth = "Churn", ".pred_0")$.estimate

telco_ADASYN_rf_test_pred <- telco_ADASYN_rf_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(telco_ADASYN_test2)
telco_ADASYN_rf_f1 <- f_meas(telco_ADASYN_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_ADASYN_rf_mcc <- mcc(telco_ADASYN_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_ADASYN_rf_test_pred$.pred_class <- as.numeric(levels(telco_ADASYN_rf_test_pred$.pred_class))[telco_ADASYN_rf_test_pred$.pred_class]
telco_ADASYN_rf_auc <- roc_auc(telco_ADASYN_rf_test_pred, truth = "Churn", ".pred_0")$.estimate

# RF Fair Utility
telco_UB_rf_test_pred$.pred_class <- as.factor(telco_UB_rf_test_pred$.pred_class)
FU_telco_UB_rf_gender <- calculate_fair_utility(
  data = telco_UB_rf_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_UB_rf_age <- calculate_fair_utility(
  data = telco_UB_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_UB_rf_dep <- calculate_fair_utility(
  data = telco_UB_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

telco_ROS_rf_test_pred$.pred_class <- as.factor(telco_ROS_rf_test_pred$.pred_class)
FU_telco_ROS_rf_gender <- calculate_fair_utility(
  data = telco_ROS_rf_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_ROS_rf_age <- calculate_fair_utility(
  data = telco_ROS_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_ROS_rf_dep <- calculate_fair_utility(
  data = telco_ROS_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

telco_SMOTE_rf_test_pred$.pred_class <- as.factor(telco_SMOTE_rf_test_pred$.pred_class)
FU_telco_SMOTE_rf_gender <- calculate_fair_utility(
  data = telco_SMOTE_rf_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_SMOTE_rf_age <- calculate_fair_utility(
  data = telco_SMOTE_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_SMOTE_rf_dep <- calculate_fair_utility(
  data = telco_SMOTE_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

telco_ADASYN_rf_test_pred$.pred_class <- as.factor(telco_ADASYN_rf_test_pred$.pred_class)
FU_telco_ADASYN_rf_gender <- calculate_fair_utility(
  data = telco_ADASYN_rf_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_ADASYN_rf_age <- calculate_fair_utility(
  data = telco_ADASYN_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_ADASYN_rf_dep <- calculate_fair_utility(
  data = telco_ADASYN_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# RF SPD
SPD_telco_UB_rf_gender <- calculate_spd(
  data = telco_UB_rf_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_UB_rf_age <- calculate_spd(
  data = telco_UB_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_UB_rf_dep <- calculate_spd(
  data = telco_UB_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ROS_rf_gender <- calculate_spd(
  data = telco_ROS_rf_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ROS_rf_age <- calculate_spd(
  data = telco_ROS_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ROS_rf_dep <- calculate_spd(
  data = telco_ROS_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_SMOTE_rf_gender <- calculate_spd(
  data = telco_SMOTE_rf_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_SMOTE_rf_age <- calculate_spd(
  data = telco_SMOTE_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_SMOTE_rf_dep <- calculate_spd(
  data = telco_SMOTE_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ADASYN_rf_gender <- calculate_spd(
  data = telco_ADASYN_rf_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ADASYN_rf_age <- calculate_spd(
  data = telco_ADASYN_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ADASYN_rf_dep <- calculate_spd(
  data = telco_ADASYN_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# RF EOD
EOD_telco_UB_rf_gender <- calculate_eod(
  data = telco_UB_rf_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_UB_rf_age <- calculate_eod(
  data = telco_UB_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_UB_rf_dep <- calculate_eod(
  data = telco_UB_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ROS_rf_gender <- calculate_eod(
  data = telco_ROS_rf_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ROS_rf_age <- calculate_eod(
  data = telco_ROS_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ROS_rf_dep <- calculate_eod(
  data = telco_ROS_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_SMOTE_rf_gender <- calculate_eod(
  data = telco_SMOTE_rf_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_SMOTE_rf_age <- calculate_eod(
  data = telco_SMOTE_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_SMOTE_rf_dep <- calculate_eod(
  data = telco_SMOTE_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ADASYN_rf_gender <- calculate_eod(
  data = telco_ADASYN_rf_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ADASYN_rf_age <- calculate_eod(
  data = telco_ADASYN_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ADASYN_rf_dep <- calculate_eod(
  data = telco_ADASYN_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# Overview RF 
evaluation_results_RF_telco <- data.frame(
  Model = c("UB", "ROS", "SMOTE", "ADASYN"),
  F1_Score = c(telco_UB_rf_f1, telco_ROS_rf_f1, telco_SMOTE_rf_f1, telco_ADASYN_rf_f1),
  MCC = c(telco_UB_rf_mcc, telco_ROS_rf_mcc, telco_SMOTE_rf_mcc, telco_ADASYN_rf_mcc),
  AUC = c(telco_UB_rf_auc, telco_ROS_rf_auc, telco_SMOTE_rf_auc, telco_ADASYN_rf_auc),
  SPD_gender = c(SPD_telco_UB_rf_gender, SPD_telco_ROS_rf_gender, SPD_telco_SMOTE_rf_gender, SPD_telco_ADASYN_rf_gender),
  EOD_gender = c(EOD_telco_UB_rf_gender, EOD_telco_ROS_rf_gender, EOD_telco_SMOTE_rf_gender, EOD_telco_ADASYN_rf_gender),
  Fair_Utility_gender = c(FU_telco_UB_rf_gender, FU_telco_ROS_rf_gender, FU_telco_SMOTE_rf_gender, FU_telco_ADASYN_rf_gender),
  SPD_age = c(SPD_telco_UB_rf_age, SPD_telco_ROS_rf_age, SPD_telco_SMOTE_rf_age, SPD_telco_ADASYN_rf_age),
  EOD_age = c(EOD_telco_UB_rf_age, EOD_telco_ROS_rf_age, EOD_telco_SMOTE_rf_age, EOD_telco_ADASYN_rf_age),
  Fair_Utility_age = c(FU_telco_UB_rf_age, FU_telco_ROS_rf_age, FU_telco_SMOTE_rf_age, FU_telco_ADASYN_rf_age),
  SPD_dep = c(SPD_telco_UB_rf_dep, SPD_telco_ROS_rf_dep, SPD_telco_SMOTE_rf_dep, SPD_telco_ADASYN_rf_dep),
  EOD_dep = c(EOD_telco_UB_rf_dep, EOD_telco_ROS_rf_dep, EOD_telco_SMOTE_rf_dep, EOD_telco_ADASYN_rf_dep),
  Fair_Utility_dep = c(FU_telco_UB_rf_dep, FU_telco_ROS_rf_dep, FU_telco_SMOTE_rf_dep, FU_telco_ADASYN_rf_dep)
)

# GBM
tuning_params <- parameters(trees(range = c(0, 30)),learn_rate(range = c(0.01, 0.1)), tree_depth(range = c(1, 3)))

telco_UB_gbm_recipe <- recipe(Churn ~ ., data = telco_UB_train2)
telco_UB_gbm_model_tune <- boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune(), stop_iter = 100) |> set_mode("classification") |> set_engine("xgboost")
telco_UB_gbm_tune_wf <- workflow() |> add_recipe(telco_UB_gbm_recipe) |> add_model(telco_UB_gbm_model_tune)
telco_UB_gbm_tune <- tune_bayes(telco_UB_gbm_tune_wf, resamples = cv_folds_telco_UB_train, iter = 20, param_info = tuning_params, metrics = class_metrics)
telco_UB_gbm_tune_metrics <- telco_UB_gbm_tune |> collect_metrics()
telco_UB_gbm_selected <- telco_UB_gbm_tune_metrics |> filter(.metric == "roc_auc") |> filter(tree_depth == 3) |> arrange(desc(mean)) |> slice_head(n = 1)
telco_UB_gbm_final_wf <- telco_UB_gbm_tune_wf |> finalize_workflow(telco_UB_gbm_selected)
telco_UB_gbm_final_fit <- telco_UB_gbm_final_wf |> last_fit(telco_UB_split2, metrics = class_metrics)

telco_ROS_gbm_recipe <- recipe(Churn ~ ., data = telco_ROS_train2)
telco_ROS_gbm_model_tune <- boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune(), stop_iter = 100) |> set_mode("classification") |> set_engine("xgboost")
telco_ROS_gbm_tune_wf <- workflow() |> add_recipe(telco_ROS_gbm_recipe) |> add_model(telco_ROS_gbm_model_tune)
telco_ROS_gbm_tune <- tune_bayes(telco_ROS_gbm_tune_wf, resamples = cv_folds_telco_ROS_train, iter = 20, param_info = tuning_params, metrics = class_metrics)
telco_ROS_gbm_tune_metrics <- telco_ROS_gbm_tune |> collect_metrics()
telco_ROS_gbm_selected <- telco_ROS_gbm_tune_metrics |> filter(.metric == "roc_auc") |> filter(tree_depth == 3) |> arrange(desc(mean)) |> slice_head(n = 1)
telco_ROS_gbm_final_wf <- telco_ROS_gbm_tune_wf |> finalize_workflow(telco_ROS_gbm_selected)
telco_ROS_gbm_final_fit <- telco_ROS_gbm_final_wf |> last_fit(telco_ROS_split2, metrics = class_metrics)

telco_SMOTE_gbm_recipe <- recipe(Churn ~ ., data = telco_SMOTE_train2)
telco_SMOTE_gbm_model_tune <- boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune(), stop_iter = 100) |> set_mode("classification") |> set_engine("xgboost")
telco_SMOTE_gbm_tune_wf <- workflow() |> add_recipe(telco_SMOTE_gbm_recipe) |> add_model(telco_SMOTE_gbm_model_tune)
telco_SMOTE_gbm_tune <- tune_bayes(telco_SMOTE_gbm_tune_wf, resamples = cv_folds_telco_SMOTE_train, iter = 20, param_info = tuning_params, metrics = class_metrics)
telco_SMOTE_gbm_tune_metrics <- telco_SMOTE_gbm_tune |> collect_metrics()
telco_SMOTE_gbm_selected <- telco_SMOTE_gbm_tune_metrics |> filter(.metric == "roc_auc")|> filter(tree_depth == 3) |> arrange(desc(mean)) |> slice_head(n = 1)
telco_SMOTE_gbm_final_wf <- telco_SMOTE_gbm_tune_wf |> finalize_workflow(telco_SMOTE_gbm_selected)
telco_SMOTE_gbm_final_fit <- telco_SMOTE_gbm_final_wf |> last_fit(telco_SMOTE_split2, metrics = class_metrics)

telco_ADASYN_gbm_recipe <- recipe(Churn ~ ., data = telco_ADASYN_train2)
telco_ADASYN_gbm_model_tune <- boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune(), stop_iter = 100) |> set_mode("classification") |> set_engine("xgboost")
telco_ADASYN_gbm_tune_wf <- workflow() |> add_recipe(telco_ADASYN_gbm_recipe) |> add_model(telco_ADASYN_gbm_model_tune)
telco_ADASYN_gbm_tune <- tune_bayes(telco_ADASYN_gbm_tune_wf, resamples = cv_folds_telco_ADASYN_train, iter = 20, param_info = tuning_params, metrics = class_metrics)
telco_ADASYN_gbm_tune_metrics <- telco_ADASYN_gbm_tune |> collect_metrics()
telco_ADASYN_gbm_selected <- telco_ADASYN_gbm_tune_metrics |> filter(.metric == "roc_auc")|> filter(tree_depth == 3) |> arrange(desc(mean)) |> slice_head(n = 1)
telco_ADASYN_gbm_final_wf <- telco_ADASYN_gbm_tune_wf |> finalize_workflow(telco_ADASYN_gbm_selected)
telco_ADASYN_gbm_final_fit <- telco_ADASYN_gbm_final_wf |> last_fit(telco_ADASYN_split2, metrics = class_metrics)

# GBM Predictive performance
telco_UB_gbm_test_pred <- telco_UB_gbm_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(telco_UB_test2)
telco_UB_gbm_f1 <- f_meas(telco_UB_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_UB_gbm_mcc <- mcc(telco_UB_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_UB_gbm_test_pred$.pred_class <- as.numeric(levels(telco_UB_gbm_test_pred$.pred_class))[telco_UB_gbm_test_pred$.pred_class]
telco_UB_gbm_auc <- roc_auc(telco_UB_gbm_test_pred, truth = "Churn", ".pred_0")$.estimate

telco_ROS_gbm_test_pred <- telco_ROS_gbm_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(telco_ROS_test2)
telco_ROS_gbm_f1 <- f_meas(telco_ROS_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_ROS_gbm_mcc <- mcc(telco_ROS_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_ROS_gbm_test_pred$.pred_class <- as.numeric(levels(telco_ROS_gbm_test_pred$.pred_class))[telco_ROS_gbm_test_pred$.pred_class]
telco_ROS_gbm_auc <- roc_auc(telco_ROS_gbm_test_pred, truth = "Churn", ".pred_0")$.estimate

telco_SMOTE_gbm_test_pred <- telco_SMOTE_gbm_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(telco_SMOTE_test2)
telco_SMOTE_gbm_f1 <- f_meas(telco_SMOTE_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_SMOTE_gbm_mcc <- mcc(telco_SMOTE_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_SMOTE_gbm_test_pred$.pred_class <- as.numeric(levels(telco_SMOTE_gbm_test_pred$.pred_class))[telco_SMOTE_gbm_test_pred$.pred_class]
telco_SMOTE_gbm_auc <- roc_auc(telco_SMOTE_gbm_test_pred, truth = "Churn", ".pred_0")$.estimate

telco_ADASYN_gbm_test_pred <- telco_ADASYN_gbm_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(telco_ADASYN_test2)
telco_ADASYN_gbm_f1 <- f_meas(telco_ADASYN_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_ADASYN_gbm_mcc <- mcc(telco_ADASYN_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
telco_ADASYN_gbm_test_pred$.pred_class <- as.numeric(levels(telco_ADASYN_gbm_test_pred$.pred_class))[telco_ADASYN_gbm_test_pred$.pred_class]
telco_ADASYN_gbm_auc <- roc_auc(telco_ADASYN_gbm_test_pred, truth = "Churn", ".pred_0")$.estimate

# GBM Fair Utility
telco_UB_gbm_test_pred$.pred_class <- as.factor(telco_UB_gbm_test_pred$.pred_class)
FU_telco_UB_gbm_gender <- calculate_fair_utility(
  data = telco_UB_gbm_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_UB_gbm_age <- calculate_fair_utility(
  data = telco_UB_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_UB_gbm_dep <- calculate_fair_utility(
  data = telco_UB_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

telco_ROS_gbm_test_pred$.pred_class <- as.factor(telco_ROS_gbm_test_pred$.pred_class)
FU_telco_ROS_gbm_gender <- calculate_fair_utility(
  data = telco_ROS_gbm_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_ROS_gbm_age <- calculate_fair_utility(
  data = telco_ROS_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_ROS_gbm_dep <- calculate_fair_utility(
  data = telco_ROS_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

telco_SMOTE_gbm_test_pred$.pred_class <- as.factor(telco_SMOTE_gbm_test_pred$.pred_class)
FU_telco_SMOTE_gbm_gender <- calculate_fair_utility(
  data = telco_SMOTE_gbm_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_SMOTE_gbm_age <- calculate_fair_utility(
  data = telco_SMOTE_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_SMOTE_gbm_dep <- calculate_fair_utility(
  data = telco_SMOTE_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

telco_ADASYN_gbm_test_pred$.pred_class <- as.factor(telco_ADASYN_gbm_test_pred$.pred_class)
FU_telco_ADASYN_gbm_gender <- calculate_fair_utility(
  data = telco_ADASYN_gbm_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_ADASYN_gbm_age <- calculate_fair_utility(
  data = telco_ADASYN_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_telco_ADASYN_gbm_dep <- calculate_fair_utility(
  data = telco_ADASYN_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# GBM SPD
SPD_telco_UB_gbm_gender <- calculate_spd(
  data = telco_UB_gbm_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_UB_gbm_age <- calculate_spd(
  data = telco_UB_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_UB_gbm_dep <- calculate_spd(
  data = telco_UB_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ROS_gbm_gender <- calculate_spd(
  data = telco_ROS_gbm_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ROS_gbm_age <- calculate_spd(
  data = telco_ROS_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ROS_gbm_dep <- calculate_spd(
  data = telco_ROS_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_SMOTE_gbm_gender <- calculate_spd(
  data = telco_SMOTE_gbm_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_SMOTE_gbm_age <- calculate_spd(
  data = telco_SMOTE_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_SMOTE_gbm_dep <- calculate_spd(
  data = telco_SMOTE_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ADASYN_gbm_gender <- calculate_spd(
  data = telco_ADASYN_gbm_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ADASYN_gbm_age <- calculate_spd(
  data = telco_ADASYN_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_telco_ADASYN_gbm_dep <- calculate_spd(
  data = telco_ADASYN_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# GBM EOD
EOD_telco_UB_gbm_gender <- calculate_eod(
  data = telco_UB_gbm_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_UB_gbm_age <- calculate_eod(
  data = telco_UB_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_UB_gbm_dep <- calculate_eod(
  data = telco_UB_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ROS_gbm_gender <- calculate_eod(
  data = telco_ROS_gbm_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ROS_gbm_age <- calculate_eod(
  data = telco_ROS_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ROS_gbm_dep <- calculate_eod(
  data = telco_ROS_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_SMOTE_gbm_gender <- calculate_eod(
  data = telco_SMOTE_gbm_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_SMOTE_gbm_age <- calculate_eod(
  data = telco_SMOTE_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_SMOTE_gbm_dep <- calculate_eod(
  data = telco_SMOTE_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ADASYN_gbm_gender <- calculate_eod(
  data = telco_ADASYN_gbm_test_pred,
  group_column = "gender_Male",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ADASYN_gbm_age <- calculate_eod(
  data = telco_ADASYN_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_telco_ADASYN_gbm_dep <- calculate_eod(
  data = telco_ADASYN_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# Overview GBM 
evaluation_results_GBM_telco <- data.frame(
  Model = c("UB", "ROS", "SMOTE", "ADASYN"),
  F1_Score = c(telco_UB_gbm_f1, telco_ROS_gbm_f1, telco_SMOTE_gbm_f1, telco_ADASYN_gbm_f1),
  MCC = c(telco_UB_gbm_mcc, telco_ROS_gbm_mcc, telco_SMOTE_gbm_mcc, telco_ADASYN_gbm_mcc),
  AUC = c(telco_UB_gbm_auc, telco_ROS_gbm_auc, telco_SMOTE_gbm_auc, telco_ADASYN_gbm_auc),
  SPD_gender = c(SPD_telco_UB_gbm_gender, SPD_telco_ROS_gbm_gender, SPD_telco_SMOTE_gbm_gender, SPD_telco_ADASYN_gbm_gender),
  EOD_gender = c(EOD_telco_UB_gbm_gender, EOD_telco_ROS_gbm_gender, EOD_telco_SMOTE_gbm_gender, EOD_telco_ADASYN_gbm_gender),
  Fair_Utility_gender = c(FU_telco_UB_gbm_gender, FU_telco_ROS_gbm_gender, FU_telco_SMOTE_gbm_gender, FU_telco_ADASYN_gbm_gender),
  SPD_age = c(SPD_telco_UB_gbm_age, SPD_telco_ROS_gbm_age, SPD_telco_SMOTE_gbm_age, SPD_telco_ADASYN_gbm_age),
  EOD_age = c(EOD_telco_UB_gbm_age, EOD_telco_ROS_gbm_age, EOD_telco_SMOTE_gbm_age, EOD_telco_ADASYN_gbm_age),
  Fair_Utility_age = c(FU_telco_UB_gbm_age, FU_telco_ROS_gbm_age, FU_telco_SMOTE_gbm_age, FU_telco_ADASYN_gbm_age),
  SPD_dep = c(SPD_telco_UB_gbm_dep, SPD_telco_ROS_gbm_dep, SPD_telco_SMOTE_gbm_dep, SPD_telco_ADASYN_gbm_dep),
  EOD_dep = c(EOD_telco_UB_gbm_dep, EOD_telco_ROS_gbm_dep, EOD_telco_SMOTE_gbm_dep, EOD_telco_ADASYN_gbm_dep),
  Fair_Utility_dep = c(FU_telco_UB_gbm_dep, FU_telco_ROS_gbm_dep, FU_telco_SMOTE_gbm_dep, FU_telco_ADASYN_gbm_dep)
)

# PART 2: CELL DATA
cell <- read.csv("cell2celltrain.csv", header = TRUE, sep = ",")

colnames(cell)
summary(cell)

cell |>
  count(Churn) |>
  mutate(prop = n / sum(n))
cell |>
  count(ChildrenInHH) |>
  mutate(prop = n / sum(n))
cell |>
  count(AgeHH1) |>
  mutate(prop = n / sum(n))
cell |>
  count(AgeHH2) |>
  mutate(prop = n / sum(n))
cell |>
  count(MaritalStatus) |>
  mutate(prop = n / sum(n))
cell |>
  count(Homeownership) |>
  mutate(prop = n / sum(n))
cell |>
  count(HandsetPrice) |>
  mutate(prop = n / sum(n))

summary(cell)
sum(is.na(cell))

# Cell data preprocessing
cell <- cell[complete.cases(cell), ]
cell <- cell %>% select(-CustomerID, -ServiceArea, -MaritalStatus, -Homeownership, -HandsetPrice)
cell <- cell |> rename(Dependents = ChildrenInHH)
cell <- cell |> mutate(SeniorCitizen = if_else(AgeHH1 >= 65, 1, 0)) |> select(-AgeHH1, -AgeHH2)
cell_bin <- c( "Churn", "Dependents", "HandsetRefurbished", "HandsetWebCapable", "TruckOwner", "RVOwner",
               "BuysViaMailOrder", "RespondsToMailOffers", "OptOutMailings", "NonUSTravel", "OwnsComputer",
               "HasCreditCard", "NewCellphoneUser", "NotNewCellphoneUser", "OwnsMotorcycle", "MadeCallToRetentionTeam")
cell[cell_bin] <- lapply(cell[cell_bin], function(x) ifelse(x == "Yes", 1, ifelse(x == "No", 0, x)))
cell <- cell |>
  mutate(CreditRating = as.numeric(gsub("-.*", "", CreditRating)))
cell <- fastDummies::dummy_cols(cell, 
                                select_columns = c("PrizmCode", "Occupation"),
                                remove_first_dummy = TRUE,  
                                remove_selected_columns = TRUE)
cell <- cell |>
  mutate(across(where(is.numeric), ~ ( . - min(.) ) / ( max(.) - min(.) ) ))
cell$Churn <- factor(cell$Churn, levels = c("0", "1"))
cell <- cell  
for (col in names(cell)) {
  if (is.character(cell[[col]])) {
    cell[[col]] <- as.numeric(as.character(cell[[col]]))
  }
}

# Unbalanced
cell_UB <- cell

# ROS
cellN <- sum(cell$Churn == 0) * 2
cell_ROS <- ovun.sample(Churn ~ ., data = cell, method = "over", N = cellN)$data

# SMOTE
cell_SMOTE <- smote(cell, "Churn", k=5, over_ratio = 1)

# ADASYN
cell_ADASYN <- adasyn(cell, "Churn", k=5, over_ratio = 1)

# Training and testing splits LR
cell_UB_split <- initial_validation_split(cell_UB, prop = c(.4, .4), strata = Churn)
cell_UB_train <- training(cell_UB_split)
cell_UB_validation <- validation(cell_UB_split)
cell_UB_test <- testing(cell_UB_split)

cell_ROS_split <- initial_validation_split(cell_ROS, prop = c(.4, .4), strata = Churn)
cell_ROS_train <- training(cell_ROS_split)
cell_ROS_validation <- validation(cell_ROS_split)
cell_ROS_test <- testing(cell_ROS_split)

cell_SMOTE_split <- initial_validation_split(cell_SMOTE, prop = c(.4, .4), strata = Churn)
cell_SMOTE_train <- training(cell_SMOTE_split)
cell_SMOTE_validation <- validation(cell_SMOTE_split)
cell_SMOTE_test <- testing(cell_SMOTE_split)

cell_ADASYN_split <- initial_validation_split(cell_ADASYN, prop = c(.4, .4), strata = Churn)
cell_ADASYN_train <- training(cell_ADASYN_split)
cell_ADASYN_validation <- validation(cell_ADASYN_split)
cell_ADASYN_test <- testing(cell_ADASYN_split)

# LR
cell_UB_lr <- logistic_reg() |> set_mode("classification") |> set_engine("glm")
cell_UB_lr_recipe <- recipe(Churn ~ ., data = cell_UB_train )
cell_UB_lr_workflow <- workflow() |> add_model(cell_UB_lr) |> add_recipe(cell_UB_lr_recipe)
cell_UB_lr_last_fit <- cell_UB_lr_workflow |> last_fit(cell_UB_split, metrics = metric_set(roc_auc, accuracy), add_validation_set = TRUE)

cell_ROS_lr <- logistic_reg() |> set_mode("classification") |> set_engine("glm")
cell_ROS_lr_recipe <- recipe(Churn ~ ., data = cell_ROS_train )
cell_ROS_lr_workflow <- workflow() |> add_model(cell_ROS_lr) |> add_recipe(cell_ROS_lr_recipe)
cell_ROS_lr_last_fit <- cell_ROS_lr_workflow |> last_fit(cell_ROS_split, metrics = metric_set(roc_auc, accuracy), add_validation_set = TRUE)

variables_cell <- cell_ROS_lr_last_fit  |>
  extract_fit_parsnip() |>
  tidy() |>
  mutate(abs_estimate = abs(estimate)) |>
  arrange(desc(abs_estimate)) |>
  slice_head(n = 15) |>
  pull(term)

cell_SMOTE_lr <- logistic_reg() |> set_mode("classification") |> set_engine("glm")
cell_SMOTE_lr_recipe <- recipe(Churn ~ ., data = cell_SMOTE_train )
cell_SMOTE_lr_workflow <- workflow() |> add_model(cell_SMOTE_lr) |> add_recipe(cell_SMOTE_lr_recipe)
cell_SMOTE_lr_last_fit <- cell_SMOTE_lr_workflow |> last_fit(cell_SMOTE_split, metrics = metric_set(roc_auc, accuracy), add_validation_set = TRUE)

cell_ADASYN_lr <- logistic_reg() |> set_mode("classification") |> set_engine("glm")
cell_ADASYN_lr_recipe <- recipe(Churn ~ ., data = cell_ADASYN_train )
cell_ADASYN_lr_workflow <- workflow() |> add_model(cell_ADASYN_lr) |> add_recipe(cell_ADASYN_lr_recipe)
cell_ADASYN_lr_last_fit <- cell_ADASYN_lr_workflow |> last_fit(cell_ADASYN_split, metrics = metric_set(roc_auc, accuracy), add_validation_set = TRUE)

# LR Predictive Performance
cell_UB_test_pred <- cell_UB_lr_last_fit |> collect_predictions() |> select(-Churn) |> bind_cols(cell_UB_test)
cell_UB_lr_f1 <- f_meas(cell_UB_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_UB_lr_mcc <- mcc(cell_UB_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_UB_test_pred$.pred_class <- as.numeric(levels(cell_UB_test_pred$.pred_class))[cell_UB_test_pred$.pred_class]
cell_UB_lr_auc <- roc_auc(cell_UB_test_pred, truth = "Churn", ".pred_0")$.estimate

cell_ROS_test_pred <- cell_ROS_lr_last_fit |> collect_predictions() |> select(-Churn) |> bind_cols(cell_ROS_test)
cell_ROS_lr_f1 <- f_meas(cell_ROS_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_ROS_lr_mcc <- mcc(cell_ROS_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_ROS_test_pred$.pred_class <- as.numeric(levels(cell_ROS_test_pred$.pred_class))[cell_ROS_test_pred$.pred_class]
cell_ROS_lr_auc <- roc_auc(cell_ROS_test_pred, truth = "Churn", ".pred_0")$.estimate

cell_SMOTE_test_pred <- cell_SMOTE_lr_last_fit |> collect_predictions() |> select(-Churn) |> bind_cols(cell_SMOTE_test)
cell_SMOTE_lr_f1 <- f_meas(cell_SMOTE_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_SMOTE_lr_mcc <- mcc(cell_SMOTE_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_SMOTE_test_pred$.pred_class <- as.numeric(levels(cell_SMOTE_test_pred$.pred_class))[cell_SMOTE_test_pred$.pred_class]
cell_SMOTE_lr_auc <- roc_auc(cell_SMOTE_test_pred, truth = "Churn", ".pred_0")$.estimate

cell_ADASYN_test_pred <- cell_ADASYN_lr_last_fit |> collect_predictions() |> select(-Churn) |> bind_cols(cell_ADASYN_test)
cell_ADASYN_lr_f1 <- f_meas(cell_ADASYN_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_ADASYN_lr_mcc <- mcc(cell_ADASYN_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_ADASYN_test_pred$.pred_class <- as.numeric(levels(cell_ADASYN_test_pred$.pred_class))[cell_ADASYN_test_pred$.pred_class]
cell_ADASYN_lr_auc <- roc_auc(cell_ADASYN_test_pred, truth = "Churn", ".pred_0")$.estimate

# LR Fair utility
cell_UB_test_pred$.pred_class <- as.factor(cell_UB_test_pred$.pred_class)
cell_ROS_test_pred$.pred_class <- as.factor(cell_ROS_test_pred$.pred_class)
cell_SMOTE_test_pred$.pred_class <- as.factor(cell_SMOTE_test_pred$.pred_class)
cell_ADASYN_test_pred$.pred_class <- as.factor(cell_ADASYN_test_pred$.pred_class)

FU_cell_UB_lr_age <- calculate_fair_utility(
  data = cell_UB_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_UB_lr_dep <- calculate_fair_utility(
  data = cell_UB_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_ROS_lr_age <- calculate_fair_utility(
  data = cell_ROS_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_ROS_lr_dep <- calculate_fair_utility(
  data = cell_ROS_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_SMOTE_lr_age <- calculate_fair_utility(
  data = cell_SMOTE_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_SMOTE_lr_dep <- calculate_fair_utility(
  data = cell_SMOTE_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_ADASYN_lr_age <- calculate_fair_utility(
  data = cell_ADASYN_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_ADASYN_lr_dep <- calculate_fair_utility(
  data = cell_ADASYN_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# LR SPD
SPD_cell_UB_lr_age <- calculate_spd(
  data = cell_UB_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_UB_lr_dep <- calculate_spd(
  data = cell_UB_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_ROS_lr_age <- calculate_spd(
  data = cell_ROS_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_ROS_lr_dep <- calculate_spd(
  data = cell_ROS_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_SMOTE_lr_age <- calculate_spd(
  data = cell_SMOTE_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_SMOTE_lr_dep <- calculate_spd(
  data = cell_SMOTE_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)


SPD_cell_ADASYN_lr_age <- calculate_spd(
  data = cell_ADASYN_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_ADASYN_lr_dep <- calculate_spd(
  data = cell_ADASYN_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# LR EOD
EOD_cell_UB_lr_age <- calculate_eod(
  data = cell_UB_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_UB_lr_dep <- calculate_eod(
  data = cell_UB_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_ROS_lr_age <- calculate_eod(
  data = cell_ROS_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_ROS_lr_dep <- calculate_eod(
  data = cell_ROS_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_SMOTE_lr_age <- calculate_eod(
  data = cell_SMOTE_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_SMOTE_lr_dep <- calculate_eod(
  data = cell_SMOTE_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_ADASYN_lr_age <- calculate_eod(
  data = cell_ADASYN_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_ADASYN_lr_dep <- calculate_eod(
  data = cell_ADASYN_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# Overview LR 
evaluation_results_LR_cell <- data.frame(
  Model = c("UB", "ROS", "SMOTE", "ADASYN"),
  F1_Score = c(cell_UB_lr_f1, cell_ROS_lr_f1, cell_SMOTE_lr_f1, cell_ADASYN_lr_f1),
  MCC = c(cell_UB_lr_mcc, cell_ROS_lr_mcc, cell_SMOTE_lr_mcc, cell_ADASYN_lr_mcc),
  AUC = c(cell_UB_lr_auc, cell_ROS_lr_auc, cell_SMOTE_lr_auc, cell_ADASYN_lr_auc),
  SPD_age = c(SPD_cell_UB_lr_age, SPD_cell_ROS_lr_age, SPD_cell_SMOTE_lr_age, SPD_cell_ADASYN_lr_age),
  EOD_age = c(EOD_cell_UB_lr_age, EOD_cell_ROS_lr_age, EOD_cell_SMOTE_lr_age, EOD_cell_ADASYN_lr_age),
  Fair_Utility_age = c(FU_cell_UB_lr_age, FU_cell_ROS_lr_age, FU_cell_SMOTE_lr_age, FU_cell_ADASYN_lr_age),
  SPD_dep = c(SPD_cell_UB_lr_dep, SPD_cell_ROS_lr_dep, SPD_cell_SMOTE_lr_dep, SPD_cell_ADASYN_lr_dep),
  EOD_dep = c(EOD_cell_UB_lr_dep, EOD_cell_ROS_lr_dep, EOD_cell_SMOTE_lr_dep, EOD_cell_ADASYN_lr_dep),
  Fair_Utility_dep = c(FU_cell_UB_lr_dep, FU_cell_ROS_lr_dep, FU_cell_SMOTE_lr_dep, FU_cell_ADASYN_lr_dep)
)

# Training and testing splits Rf
cell_UB_split2 <- initial_split(cell_UB, prop = 0.8, strata = Churn)
cell_UB_train2 <- training(cell_UB_split2)
cell_UB_test2 <- testing(cell_UB_split2)
cv_folds_cell_UB_train <- cell_UB_train2 |> vfold_cv(v = 10, repeats = 5, strata = "Churn")

cell_ROS_split2 <- initial_split(cell_ROS, prop = 0.8, strata = Churn)
cell_ROS_train2 <- training(cell_ROS_split2)
cell_ROS_test2 <- testing(cell_ROS_split2)
cv_folds_cell_ROS_train <- cell_ROS_train2 |> vfold_cv(v = 10, repeats = 5, strata = "Churn")

cell_SMOTE_split2 <- initial_split(cell_SMOTE, prop = 0.8, strata = Churn)
cell_SMOTE_train2 <- training(cell_SMOTE_split2)
cell_SMOTE_test2 <- testing(cell_SMOTE_split2)
cv_folds_cell_SMOTE_train <- cell_SMOTE_train2 |> vfold_cv(v = 10, repeats = 5, strata = "Churn")

cell_ADASYN_split2 <- initial_split(cell_ADASYN, prop = 0.8, strata = Churn)
cell_ADASYN_train2 <- training(cell_ADASYN_split2)
cell_ADASYN_test2 <- testing(cell_ADASYN_split2)
cv_folds_cell_ADASYN_train <- cell_ADASYN_train2 |> vfold_cv(v = 10, repeats = 5, strata = "Churn")

# RF
class_metrics <- metric_set(accuracy, roc_auc)
formula_string <- paste("Churn ~", paste(variables_cell, collapse = " + "))
formula_cell <- as.formula(formula_string)

cell_UB_rf <- rand_forest(mtry = tune(), trees = 50) |> set_mode("classification") 
cell_UB_rf_recipe <- recipe(formula_cell, data = cell_UB_train2)
cell_UB_rf_workflow <- workflow() |> add_model(cell_UB_rf) |> add_recipe(cell_UB_rf_recipe)
rf_tune_grid <- grid_regular(mtry(range = c(1, 10)), levels = 10)
cell_UB_rf_tune <- tune_grid(cell_UB_rf_workflow, resamples = cv_folds_cell_UB_train, grid = rf_tune_grid, metrics = class_metrics)
cell_UB_rf_final_model <- cell_UB_rf_tune |> select_by_one_std_err(mtry, metric = "roc_auc")
cell_UB_rf_final_wf <- finalize_workflow(cell_UB_rf_workflow, cell_UB_rf_final_model)
cell_UB_rf_final_fit <- cell_UB_rf_final_wf |> last_fit(cell_UB_split2, metrics = class_metrics)

cell_ROS_rf <- rand_forest(mtry = tune(), trees = 50) |> set_mode("classification") 
cell_ROS_rf_recipe <- recipe(formula_cell, data = cell_ROS_train2)
cell_ROS_rf_workflow <- workflow() |> add_model(cell_ROS_rf) |> add_recipe(cell_ROS_rf_recipe)
rf_tune_grid <- grid_regular(mtry(range = c(1, 10)), levels = 10)
cell_ROS_rf_tune <- tune_grid(cell_ROS_rf_workflow, resamples = cv_folds_cell_ROS_train, grid = rf_tune_grid, metrics = class_metrics)
cell_ROS_rf_final_model <- cell_ROS_rf_tune |> select_by_one_std_err(mtry, metric = "roc_auc")
cell_ROS_rf_final_wf <- finalize_workflow(cell_ROS_rf_workflow, cell_ROS_rf_final_model)
cell_ROS_rf_final_fit <- cell_ROS_rf_final_wf |> last_fit(cell_ROS_split2, metrics = class_metrics)

cell_SMOTE_rf <- rand_forest(mtry = tune(), trees = 50) |> set_mode("classification") 
cell_SMOTE_rf_recipe <- recipe(formula_cell, data = cell_SMOTE_train2)
cell_SMOTE_rf_workflow <- workflow() |> add_model(cell_SMOTE_rf) |> add_recipe(cell_SMOTE_rf_recipe)
rf_tune_grid <- grid_regular(mtry(range = c(1, 10)), levels = 10)
cell_SMOTE_rf_tune <- tune_grid(cell_SMOTE_rf_workflow, resamples = cv_folds_cell_SMOTE_train, grid = rf_tune_grid, metrics = class_metrics)
cell_SMOTE_rf_final_model <- cell_SMOTE_rf_tune |> select_by_one_std_err(mtry, metric = "roc_auc")
cell_SMOTE_rf_final_wf <- finalize_workflow(cell_SMOTE_rf_workflow, cell_SMOTE_rf_final_model)
cell_SMOTE_rf_final_fit <- cell_SMOTE_rf_final_wf |> last_fit(cell_SMOTE_split2, metrics = class_metrics)

cell_ADASYN_rf <- rand_forest(mtry = tune(), trees = 50) |> set_mode("classification") 
cell_ADASYN_rf_recipe <- recipe(formula_cell, data = cell_ADASYN_train2)
cell_ADASYN_rf_workflow <- workflow() |> add_model(cell_ADASYN_rf) |> add_recipe(cell_ADASYN_rf_recipe)
rf_tune_grid <- grid_regular(mtry(range = c(1, 10)), levels = 10)
cell_ADASYN_rf_tune <- tune_grid(cell_ADASYN_rf_workflow, resamples = cv_folds_cell_ADASYN_train, grid = rf_tune_grid, metrics = class_metrics)
cell_ADASYN_rf_final_model <- cell_ADASYN_rf_tune |> select_by_one_std_err(mtry, metric = "roc_auc")
cell_ADASYN_rf_final_wf <- finalize_workflow(cell_ADASYN_rf_workflow, cell_ADASYN_rf_final_model)
cell_ADASYN_rf_final_fit <- cell_ADASYN_rf_final_wf |> last_fit(cell_ADASYN_split2, metrics = class_metrics)

# RF predictive performance
cell_UB_rf_test_pred <- cell_UB_rf_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(cell_UB_test2)
cell_UB_rf_f1 <- f_meas(cell_UB_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_UB_rf_mcc <- mcc(cell_UB_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_UB_rf_test_pred$.pred_class <- as.numeric(levels(cell_UB_rf_test_pred$.pred_class))[cell_UB_rf_test_pred$.pred_class]
cell_UB_rf_auc <- roc_auc(cell_UB_rf_test_pred, truth = "Churn", ".pred_0")$.estimate

cell_ROS_rf_test_pred <- cell_ROS_rf_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(cell_ROS_test2)
cell_ROS_rf_f1 <- f_meas(cell_ROS_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_ROS_rf_mcc <- mcc(cell_ROS_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_ROS_rf_test_pred$.pred_class <- as.numeric(levels(cell_ROS_rf_test_pred$.pred_class))[cell_ROS_rf_test_pred$.pred_class]
cell_ROS_rf_auc <- roc_auc(cell_ROS_rf_test_pred, truth = "Churn", ".pred_0")$.estimate

cell_SMOTE_rf_test_pred <- cell_SMOTE_rf_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(cell_SMOTE_test2)
cell_SMOTE_rf_f1 <- f_meas(cell_SMOTE_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_SMOTE_rf_mcc <- mcc(cell_SMOTE_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_SMOTE_rf_test_pred$.pred_class <- as.numeric(levels(cell_SMOTE_rf_test_pred$.pred_class))[cell_SMOTE_rf_test_pred$.pred_class]
cell_SMOTE_rf_auc <- roc_auc(cell_SMOTE_rf_test_pred, truth = "Churn", ".pred_0")$.estimate

cell_ADASYN_rf_test_pred <- cell_ADASYN_rf_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(cell_ADASYN_test2)
cell_ADASYN_rf_f1 <- f_meas(cell_ADASYN_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_ADASYN_rf_mcc <- mcc(cell_ADASYN_rf_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_ADASYN_rf_test_pred$.pred_class <- as.numeric(levels(cell_ADASYN_rf_test_pred$.pred_class))[cell_ADASYN_rf_test_pred$.pred_class]
cell_ADASYN_rf_auc <- roc_auc(cell_ADASYN_rf_test_pred, truth = "Churn", ".pred_0")$.estimate

# RF Fair utility
cell_UB_rf_test_pred$.pred_class <- as.factor(cell_UB_rf_test_pred$.pred_class)

FU_cell_UB_rf_age <- calculate_fair_utility(
  data = cell_UB_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_UB_rf_dep <- calculate_fair_utility(
  data = cell_UB_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

cell_ROS_rf_test_pred$.pred_class <- as.factor(cell_ROS_rf_test_pred$.pred_class)

FU_cell_ROS_rf_age <- calculate_fair_utility(
  data = cell_ROS_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_ROS_rf_dep <- calculate_fair_utility(
  data = cell_ROS_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

cell_SMOTE_rf_test_pred$.pred_class <- as.factor(cell_SMOTE_rf_test_pred$.pred_class)

FU_cell_SMOTE_rf_age <- calculate_fair_utility(
  data = cell_SMOTE_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_SMOTE_rf_dep <- calculate_fair_utility(
  data = cell_SMOTE_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

cell_ADASYN_rf_test_pred$.pred_class <- as.factor(cell_ADASYN_rf_test_pred$.pred_class)

FU_cell_ADASYN_rf_age <- calculate_fair_utility(
  data = cell_ADASYN_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_ADASYN_rf_dep <- calculate_fair_utility(
  data = cell_ADASYN_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# RF SPD
SPD_cell_UB_rf_age <- calculate_spd(
  data = cell_UB_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_UB_rf_dep <- calculate_spd(
  data = cell_UB_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_ROS_rf_age <- calculate_spd(
  data = cell_ROS_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_ROS_rf_dep <- calculate_spd(
  data = cell_ROS_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_SMOTE_rf_age <- calculate_spd(
  data = cell_SMOTE_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_SMOTE_rf_dep <- calculate_spd(
  data = cell_SMOTE_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_ADASYN_rf_age <- calculate_spd(
  data = cell_ADASYN_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_ADASYN_rf_dep <- calculate_spd(
  data = cell_ADASYN_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# RF EOD
EOD_cell_UB_rf_age <- calculate_eod(
  data = cell_UB_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_UB_rf_dep <- calculate_eod(
  data = cell_UB_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_ROS_rf_age <- calculate_eod(
  data = cell_ROS_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_ROS_rf_dep <- calculate_eod(
  data = cell_ROS_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_SMOTE_rf_age <- calculate_eod(
  data = cell_SMOTE_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_SMOTE_rf_dep <- calculate_eod(
  data = cell_SMOTE_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_ADASYN_rf_age <- calculate_eod(
  data = cell_ADASYN_rf_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_ADASYN_rf_dep <- calculate_eod(
  data = cell_ADASYN_rf_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# Overview RF 
evaluation_results_RF_cell <- data.frame(
  Model = c("UB", "ROS", "SMOTE", "ADASYN"),
  F1_Score = c(cell_UB_rf_f1, cell_ROS_rf_f1, cell_SMOTE_rf_f1, cell_ADASYN_rf_f1),
  MCC = c(cell_UB_rf_mcc, cell_ROS_rf_mcc, cell_SMOTE_rf_mcc, cell_ADASYN_rf_mcc),
  AUC = c(cell_UB_rf_auc, cell_ROS_rf_auc, cell_SMOTE_rf_auc, cell_ADASYN_rf_auc),
  SPD_age = c(SPD_cell_UB_rf_age, SPD_cell_ROS_rf_age, SPD_cell_SMOTE_rf_age, SPD_cell_ADASYN_rf_age),
  EOD_age = c(EOD_cell_UB_rf_age, EOD_cell_ROS_rf_age, EOD_cell_SMOTE_rf_age, EOD_cell_ADASYN_rf_age),
  Fair_Utility_age = c(FU_cell_UB_rf_age, FU_cell_ROS_rf_age, FU_cell_SMOTE_rf_age, FU_cell_ADASYN_rf_age),
  SPD_dep = c(SPD_cell_UB_rf_dep, SPD_cell_ROS_rf_dep, SPD_cell_SMOTE_rf_dep, SPD_cell_ADASYN_rf_dep),
  EOD_dep = c(EOD_cell_UB_rf_dep, EOD_cell_ROS_rf_dep, EOD_cell_SMOTE_rf_dep, EOD_cell_ADASYN_rf_dep),
  Fair_Utility_dep = c(FU_cell_UB_rf_dep, FU_cell_ROS_rf_dep, FU_cell_SMOTE_rf_dep, FU_cell_ADASYN_rf_dep)
)

# GBM
tuning_params <- parameters(trees(range = c(0, 30)),learn_rate(range = c(0.01, 0.05)), tree_depth(range = c(1, 3)))

cell_UB_gbm_recipe <- recipe(formula_cell, data = cell_UB_train2)
cell_UB_gbm_model_tune <- boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune(), stop_iter = 100) |> set_mode("classification") |> set_engine("xgboost")
cell_UB_gbm_tune_wf <- workflow() |> add_recipe(cell_UB_gbm_recipe) |> add_model(cell_UB_gbm_model_tune)
cell_UB_gbm_tune <- tune_bayes(cell_UB_gbm_tune_wf, resamples = cv_folds_cell_UB_train, iter = 20, param_info = tuning_params, metrics = class_metrics)
cell_UB_gbm_tune_metrics <- cell_UB_gbm_tune |> collect_metrics()
cell_UB_gbm_selected <- cell_UB_gbm_tune_metrics |> filter(.metric == "roc_auc") |> filter(tree_depth == 3) |> arrange(desc(mean)) |> slice_head(n = 1)
cell_UB_gbm_final_wf <- cell_UB_gbm_tune_wf |> finalize_workflow(cell_UB_gbm_selected)
cell_UB_gbm_final_fit <- cell_UB_gbm_final_wf |> last_fit(cell_UB_split2, metrics = class_metrics)

cell_ROS_gbm_recipe <- recipe(formula_cell, data = cell_ROS_train2)
cell_ROS_gbm_model_tune <- boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune(), stop_iter = 100) |> set_mode("classification") |> set_engine("xgboost")
cell_ROS_gbm_tune_wf <- workflow() |> add_recipe(cell_ROS_gbm_recipe) |> add_model(cell_ROS_gbm_model_tune)
cell_ROS_gbm_tune <- tune_bayes(cell_ROS_gbm_tune_wf, resamples = cv_folds_cell_ROS_train, iter = 20, param_info = tuning_params, metrics = class_metrics)
cell_ROS_gbm_tune_metrics <- cell_ROS_gbm_tune |> collect_metrics()
cell_ROS_gbm_selected <- cell_ROS_gbm_tune_metrics |> filter(.metric == "roc_auc") |> filter(tree_depth == 3) |> arrange(desc(mean)) |> slice_head(n = 1)
cell_ROS_gbm_final_wf <- cell_ROS_gbm_tune_wf |> finalize_workflow(cell_ROS_gbm_selected)
cell_ROS_gbm_final_fit <- cell_ROS_gbm_final_wf |> last_fit(cell_ROS_split2, metrics = class_metrics)

cell_SMOTE_gbm_recipe <- recipe(formula_cell, data = cell_SMOTE_train2)
cell_SMOTE_gbm_model_tune <- boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune(), stop_iter = 100) |> set_mode("classification") |> set_engine("xgboost")
cell_SMOTE_gbm_tune_wf <- workflow() |> add_recipe(cell_SMOTE_gbm_recipe) |> add_model(cell_SMOTE_gbm_model_tune)
cell_SMOTE_gbm_tune <- tune_bayes(cell_SMOTE_gbm_tune_wf, resamples = cv_folds_cell_SMOTE_train, iter = 20, param_info = tuning_params, metrics = class_metrics)
cell_SMOTE_gbm_tune_metrics <- cell_SMOTE_gbm_tune |> collect_metrics()
cell_SMOTE_gbm_selected <- cell_SMOTE_gbm_tune_metrics |> filter(.metric == "roc_auc")|> filter(tree_depth == 3) |> arrange(desc(mean)) |> slice_head(n = 1)
cell_SMOTE_gbm_final_wf <- cell_SMOTE_gbm_tune_wf |> finalize_workflow(cell_SMOTE_gbm_selected)
cell_SMOTE_gbm_final_fit <- cell_SMOTE_gbm_final_wf |> last_fit(cell_SMOTE_split2, metrics = class_metrics)

cell_ADASYN_gbm_recipe <- recipe(formula_cell, data = cell_ADASYN_train2)
cell_ADASYN_gbm_model_tune <- boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune(), stop_iter = 100) |> set_mode("classification") |> set_engine("xgboost")
cell_ADASYN_gbm_tune_wf <- workflow() |> add_recipe(cell_ADASYN_gbm_recipe) |> add_model(cell_ADASYN_gbm_model_tune)
cell_ADASYN_gbm_tune <- tune_bayes(cell_ADASYN_gbm_tune_wf, resamples = cv_folds_cell_ADASYN_train, iter = 20, param_info = tuning_params, metrics = class_metrics)
cell_ADASYN_gbm_tune_metrics <- cell_ADASYN_gbm_tune |> collect_metrics()
cell_ADASYN_gbm_selected <- cell_ADASYN_gbm_tune_metrics |> filter(.metric == "roc_auc")|> filter(tree_depth == 3) |> arrange(desc(mean)) |> slice_head(n = 1)
cell_ADASYN_gbm_final_wf <- cell_ADASYN_gbm_tune_wf |> finalize_workflow(cell_ADASYN_gbm_selected)
cell_ADASYN_gbm_final_fit <- cell_ADASYN_gbm_final_wf |> last_fit(cell_ADASYN_split2, metrics = class_metrics)

# GBM Predictive performance
cell_UB_gbm_test_pred <- cell_UB_gbm_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(cell_UB_test2)
cell_UB_gbm_f1 <- f_meas(cell_UB_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_UB_gbm_mcc <- mcc(cell_UB_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_UB_gbm_test_pred$.pred_class <- as.numeric(levels(cell_UB_gbm_test_pred$.pred_class))[cell_UB_gbm_test_pred$.pred_class]
cell_UB_gbm_auc <- roc_auc(cell_UB_gbm_test_pred, truth = "Churn", ".pred_0")$.estimate

cell_ROS_gbm_test_pred <- cell_ROS_gbm_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(cell_ROS_test2)
cell_ROS_gbm_f1 <- f_meas(cell_ROS_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_ROS_gbm_mcc <- mcc(cell_ROS_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_ROS_gbm_test_pred$.pred_class <- as.numeric(levels(cell_ROS_gbm_test_pred$.pred_class))[cell_ROS_gbm_test_pred$.pred_class]
cell_ROS_gbm_auc <- roc_auc(cell_ROS_gbm_test_pred, truth = "Churn", ".pred_0")$.estimate

cell_SMOTE_gbm_test_pred <- cell_SMOTE_gbm_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(cell_SMOTE_test2)
cell_SMOTE_gbm_f1 <- f_meas(cell_SMOTE_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_SMOTE_gbm_mcc <- mcc(cell_SMOTE_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_SMOTE_gbm_test_pred$.pred_class <- as.numeric(levels(cell_SMOTE_gbm_test_pred$.pred_class))[cell_SMOTE_gbm_test_pred$.pred_class]
cell_SMOTE_gbm_auc <- roc_auc(cell_SMOTE_gbm_test_pred, truth = "Churn", ".pred_0")$.estimate

cell_ADASYN_gbm_test_pred <- cell_ADASYN_gbm_final_fit |> collect_predictions() |> select(-Churn) |> bind_cols(cell_ADASYN_test2)
cell_ADASYN_gbm_f1 <- f_meas(cell_ADASYN_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_ADASYN_gbm_mcc <- mcc(cell_ADASYN_gbm_test_pred, truth = "Churn", estimate = ".pred_class")$.estimate
cell_ADASYN_gbm_test_pred$.pred_class <- as.numeric(levels(cell_ADASYN_gbm_test_pred$.pred_class))[cell_ADASYN_gbm_test_pred$.pred_class]
cell_ADASYN_gbm_auc <- roc_auc(cell_ADASYN_gbm_test_pred, truth = "Churn", ".pred_0")$.estimate

# GBM Fair Utility
cell_UB_gbm_test_pred$.pred_class <- as.factor(cell_UB_gbm_test_pred$.pred_class)

FU_cell_UB_gbm_age <- calculate_fair_utility(
  data = cell_UB_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_UB_gbm_dep <- calculate_fair_utility(
  data = cell_UB_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

cell_ROS_gbm_test_pred$.pred_class <- as.factor(cell_ROS_gbm_test_pred$.pred_class)

FU_cell_ROS_gbm_age <- calculate_fair_utility(
  data = cell_ROS_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_ROS_gbm_dep <- calculate_fair_utility(
  data = cell_ROS_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

cell_SMOTE_gbm_test_pred$.pred_class <- as.factor(cell_SMOTE_gbm_test_pred$.pred_class)

FU_cell_SMOTE_gbm_age <- calculate_fair_utility(
  data = cell_SMOTE_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_SMOTE_gbm_dep <- calculate_fair_utility(
  data = cell_SMOTE_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

cell_ADASYN_gbm_test_pred$.pred_class <- as.factor(cell_ADASYN_gbm_test_pred$.pred_class)

FU_cell_ADASYN_gbm_age <- calculate_fair_utility(
  data = cell_ADASYN_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

FU_cell_ADASYN_gbm_dep <- calculate_fair_utility(
  data = cell_ADASYN_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# GBM SPD
SPD_cell_UB_gbm_age <- calculate_spd(
  data = cell_UB_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_UB_gbm_dep <- calculate_spd(
  data = cell_UB_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_ROS_gbm_age <- calculate_spd(
  data = cell_ROS_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_ROS_gbm_dep <- calculate_spd(
  data = cell_ROS_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_SMOTE_gbm_age <- calculate_spd(
  data = cell_SMOTE_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_SMOTE_gbm_dep <- calculate_spd(
  data = cell_SMOTE_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_ADASYN_gbm_age <- calculate_spd(
  data = cell_ADASYN_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

SPD_cell_ADASYN_gbm_dep <- calculate_spd(
  data = cell_ADASYN_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# GBM EOD
EOD_cell_UB_gbm_age <- calculate_eod(
  data = cell_UB_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_UB_gbm_dep <- calculate_eod(
  data = cell_UB_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_ROS_gbm_age <- calculate_eod(
  data = cell_ROS_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_ROS_gbm_dep <- calculate_eod(
  data = cell_ROS_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_SMOTE_gbm_age <- calculate_eod(
  data = cell_SMOTE_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_SMOTE_gbm_dep <- calculate_eod(
  data = cell_SMOTE_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_ADASYN_gbm_age <- calculate_eod(
  data = cell_ADASYN_gbm_test_pred,
  group_column = "SeniorCitizen",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

EOD_cell_ADASYN_gbm_dep <- calculate_eod(
  data = cell_ADASYN_gbm_test_pred,
  group_column = "Dependents",
  target_column = "Churn",
  prediction_column = ".pred_class"
)

# Overview GBM 
evaluation_results_GBM_cell <- data.frame(
  Model = c("UB", "ROS", "SMOTE", "ADASYN"),
  F1_Score = c(cell_UB_gbm_f1, cell_ROS_gbm_f1, cell_SMOTE_gbm_f1, cell_ADASYN_gbm_f1),
  MCC = c(cell_UB_gbm_mcc, cell_ROS_gbm_mcc, cell_SMOTE_gbm_mcc, cell_ADASYN_gbm_mcc),
  AUC = c(cell_UB_gbm_auc, cell_ROS_gbm_auc, cell_SMOTE_gbm_auc, cell_ADASYN_gbm_auc),
  SPD_age = c(SPD_cell_UB_gbm_age, SPD_cell_ROS_gbm_age, SPD_cell_SMOTE_gbm_age, SPD_cell_ADASYN_gbm_age),
  EOD_age = c(EOD_cell_UB_gbm_age, EOD_cell_ROS_gbm_age, EOD_cell_SMOTE_gbm_age, EOD_cell_ADASYN_gbm_age),
  Fair_Utility_age = c(FU_cell_UB_gbm_age, FU_cell_ROS_gbm_age, FU_cell_SMOTE_gbm_age, FU_cell_ADASYN_gbm_age),
  SPD_dep = c(SPD_cell_UB_gbm_dep, SPD_cell_ROS_gbm_dep, SPD_cell_SMOTE_gbm_dep, SPD_cell_ADASYN_gbm_dep),
  EOD_dep = c(EOD_cell_UB_gbm_dep, EOD_cell_ROS_gbm_dep, EOD_cell_SMOTE_gbm_dep, EOD_cell_ADASYN_gbm_dep),
  Fair_Utility_dep = c(FU_cell_UB_gbm_dep, FU_cell_ROS_gbm_dep, FU_cell_SMOTE_gbm_dep, FU_cell_ADASYN_gbm_dep)
)