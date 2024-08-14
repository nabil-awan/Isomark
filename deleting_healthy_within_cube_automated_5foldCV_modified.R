### Reading data

SSDN_13_Final_Cohort_withoutILS_BeforeCubeRemoval <- read.csv("C:/Isomark_Nabil/Data/Saved_data/Incoming_Animals_Model/SSDN_13_Final_Cohort_withoutILS_BeforeCubeRemoval.csv", header = T)
dim(SSDN_13_Final_Cohort_withoutILS_BeforeCubeRemoval)
# [1] 15817    37

data <- SSDN_13_Final_Cohort_withoutILS_BeforeCubeRemoval

# sum(is.na(data$Sex))
# sum(data$Sex=="")
# # [1] 55
# check.mis.sex <- subset(SSDN_13_Final_Cohort_withoutILS_BeforeCubeRemoval, Sex=="")
# dim(check.mis.sex)
# table(check.mis.sex$Farm)
# > table(check.mis.sex$Farm)
# 
# DARR 
# 55
# > table(check.mis.sex$InDate)
# 
# 2022-06-07 
# 55 
# sum(is.na(data$Farm))
# sum(data$Farm=="")
# sum(is.na(data$Analytics_CH4))
# sum(is.na(data$Analytics_Delta))
# sum(is.na(data$InWeight))

data <- subset(data, !(Sex==""))
dim(data)
# [1] 15762    37



# Load necessary libraries
library(dplyr)
library(pROC)
library(caret)  # For createFolds function

# Set a seed for reproducibility
set.seed(123)

# Function to convert response variable to binary factor format with valid levels
convert_to_binary <- function(data) {
  data %>%
    mutate(HealthCode_14_days_modified_binary = factor(
      ifelse(HealthCode_14_days_modified == "Sick", "Sick", "Healthy"),
      levels = c("Healthy", "Sick")
    ))
}

# Function to filter healthy animals within the cube of sick animals
filter_healthy_animals <- function(data, delta_tol, ch4_tol, weight_tol) {
  sick_animals <- data %>% filter(HealthCode_14_days_modified == "Sick")
  
  keep_healthy <- rep(TRUE, nrow(data))
  
  for (i in 1:nrow(sick_animals)) {
    sick <- sick_animals[i, ]
    
    delta_min <- sick$Analytics_Delta - delta_tol
    delta_max <- sick$Analytics_Delta + delta_tol
    ch4_min <- sick$Analytics_CH4 - ch4_tol
    ch4_max <- sick$Analytics_CH4 + ch4_tol
    weight_min <- sick$InWeight - weight_tol
    weight_max <- sick$InWeight + weight_tol
    
    keep_healthy <- keep_healthy & !(data$HealthCode_14_days_modified == "Healthy" &
                                       data$Analytics_Delta >= delta_min &
                                       data$Analytics_Delta <= delta_max &
                                       data$Analytics_CH4 >= ch4_min &
                                       data$Analytics_CH4 <= ch4_max &
                                       data$InWeight >= weight_min &
                                       data$InWeight <= weight_max)
  }
  
  filtered_data <- data[keep_healthy,]
  
  message(sprintf("Filtered data size: %d rows", nrow(filtered_data)))
  
  return(filtered_data)
}

# Function to calculate Precision-Recall AUC manually
calculate_pr_auc <- function(pred_probs, true_labels) {
  true_labels <- as.numeric(true_labels == "Sick")
  
  # Check if there are any positive true labels
  if (sum(true_labels) == 0) {
    return(NA)  # No positive samples in the test set
  }
  
  # Sort predictions and true labels
  pr_data <- data.frame(score = pred_probs, label = true_labels)
  pr_data <- pr_data[order(pr_data$score, decreasing = TRUE), ]
  
  # Calculate precision and recall
  precision <- cumsum(pr_data$label) / (1:length(pr_data$label))
  recall <- cumsum(pr_data$label) / sum(true_labels)
  
  # Add an initial point for precision and recall
  precision <- c(1, precision)
  recall <- c(0, recall)
  
  # Interpolate precision-recall curve and calculate AUC
  auc_pr <- sum(diff(recall) * (precision[-length(precision)] + precision[-1]) / 2)
  
  return(auc_pr)
}

# Custom function to calculate Brier Score
calculate_brier_score <- function(true_labels, pred_probs) {
  true_labels <- as.numeric(true_labels == "Sick")
  brier_score <- mean((true_labels - pred_probs)^2)
  return(brier_score)
}

# Function to calculate metrics for a given model and data
calculate_metrics <- function(pred_probs, true_labels) {
  true_labels <- as.numeric(true_labels == "Sick")
  
  log_loss <- -mean(true_labels * log(pred_probs + 1e-15) + (1 - true_labels) * log(1 - pred_probs + 1e-15))
  roc_curve <- roc(true_labels, pred_probs)
  auc_roc <- roc_curve$auc
  auc_pr <- calculate_pr_auc(pred_probs, true_labels)
  brier_score <- calculate_brier_score(true_labels, pred_probs)
  
  return(c(Log_Loss = log_loss, AUC_ROC = auc_roc, AUC_PR = auc_pr, Brier_Score = brier_score))
}

# Custom function to check fold balance
check_fold_balance <- function(data, folds) {
  fold_summary <- lapply(folds, function(indices) {
    fold_data <- data[indices, ]
    table(fold_data$HealthCode_14_days_modified_binary)
  })
  
  return(fold_summary)
}

# Function to perform cross-validation and calculate metrics
cv_metrics <- function(data, delta_tol, ch4_tol, weight_tol) {
  filtered_data <- filter_healthy_animals(data, delta_tol, ch4_tol, weight_tol)
  
  if (nrow(filtered_data) < 5) {
    message(sprintf("Insufficient data for cross-validation after filtering for delta_tol = %f, ch4_tol = %f, weight_tol = %f", delta_tol, ch4_tol, weight_tol))
    return(c(delta_tol = delta_tol, ch4_tol = ch4_tol, weight_tol = weight_tol, Log_Loss = NA, AUC_ROC = NA, AUC_PR = NA, Brier_Score = NA))
  }
  
  filtered_data <- convert_to_binary(filtered_data)
  
  folds <- createFolds(filtered_data$HealthCode_14_days_modified_binary, k = 5, list = TRUE, returnTrain = TRUE)
  
  # Check fold balance
  fold_summary <- check_fold_balance(filtered_data, folds)
  print("Fold class distributions:")
  print(fold_summary)
  
  metrics_list <- list()
  
  for (i in 1:length(folds)) {
    train_indices <- folds[[i]]
    test_indices <- setdiff(seq_len(nrow(filtered_data)), train_indices)
    
    train_data <- filtered_data[train_indices, ]
    test_data <- filtered_data[test_indices, ]
    
    model <- glm(HealthCode_14_days_modified_binary ~ InWeight + Analytics_Delta + Analytics_CH4 + Sex + Farm,
                 data = train_data, family = binomial)
    
    pred_probs <- predict(model, newdata = test_data, type = "response")
    true_labels <- test_data$HealthCode_14_days_modified_binary
    
    metrics <- calculate_metrics(pred_probs, true_labels)
    metrics_list[[i]] <- metrics
  }
  
  avg_metrics <- colMeans(do.call(rbind, metrics_list), na.rm = TRUE)
  return(c(delta_tol = delta_tol, ch4_tol = ch4_tol, weight_tol = weight_tol, avg_metrics))
}

# Define parameters
delta_tols <- seq(1.5, 3.5, by = 0.1)
ch4_tols <- seq(2, 4.5, by = 0.1)
weight_tols <- seq(30, 50, by = 5)

# Create combinations of tolerance values
tolerance_combinations <- expand.grid(delta_tol = delta_tols, ch4_tol = ch4_tols, weight_tol = weight_tols)

# Initialize a list to store results
results <- list()

# Loop through each combination of tolerances
for (i in 1:nrow(tolerance_combinations)) {
  tol_values <- tolerance_combinations[i, ]
  
  result <- cv_metrics(data, tol_values$delta_tol, tol_values$ch4_tol, tol_values$weight_tol)
  
  results[[i]] <- result
}

# Combine results into a data frame
results_df <- do.call(rbind, results)
results_df <- as.data.frame(results_df)

# Print dimensions of results
print(dim(results_df))
# [1] 2730    7

# Print first few rows of results
head(results_df)

results_df <- subset(results_df, select = -c(AUC_PR))


# Load dplyr if not already loaded
library(dplyr)
names(results_df)
# > names(results_df)
# [1] "delta_tol"   "ch4_tol"     "weight_tol"  "Log_Loss"    "AUC_ROC"    
# [6] "Brier_Score"


# For Log Loss lower values are better
sorted_Log_Loss <- results_df %>%
  arrange(Log_Loss)
sorted_Log_Loss[1:10,]


# For Brier lower values are better
sorted_Brier <- results_df %>%
  arrange(Brier_Score)
sorted_Brier[1:10,]


# For AUC_ROC, higher values are better
sorted_AUC_ROC <- results_df %>%
  arrange(desc(AUC_ROC))
sorted_AUC_ROC[1:10,]


# Calculating the volume of the cube
results_df$cube_vol <- (2*results_df$delta_tol)*(2*results_df$ch4_tol)*(2*results_df$weight_tol)

# Saving the results for cube size selection
write.csv(results_df, file = "C:/Isomark_Nabil/Outputs/Sample selection diagram and plots for definition/cube_size_determination.csv", row.names = F)


### Graphs of cube colume vs. metrics

# Load the necessary library
library(ggplot2)

# Create the plot
ggplot(results_df, aes(x = cube_vol, y = Log_Loss)) +
  geom_point() +                     # Add points
  # geom_line() +                      # Connect the dots with a line
  labs(x = "Cube volume",            # X-axis label
       y = "Log Loss",               # Y-axis label
       title = "Log Loss vs. Cube Volume") + # Title of the plot
  theme_minimal()                    # Use a minimal theme

# Create the plot
ggplot(results_df, aes(x = cube_vol, y = Brier_Score)) +
  geom_point() +                     # Add points
  # geom_line() +                      # Connect the dots with a line
  labs(x = "Cube volume",            # X-axis label
       y = "Brier Score",               # Y-axis label
       title = "Brier Score vs. Cube Volume") + # Title of the plot
  theme_minimal()                    # Use a minimal theme


# Create the plot
ggplot(results_df, aes(x = cube_vol, y = AUC_ROC)) +
  geom_point() +                     # Add points
  # geom_line() +                      # Connect the dots with a line
  labs(x = "Cube volume",            # X-axis label
       y = "AUC ROC",               # Y-axis label
       title = "AUC ROC vs. Cube Volume") + # Title of the plot
  theme_minimal()                    # Use a minimal theme
