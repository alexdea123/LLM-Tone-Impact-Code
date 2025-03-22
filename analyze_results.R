# Set working directory (modify as needed)
setwd("/Users/alexdeaconu/Documents/School/Data Mining/LLM-Tone-Impact-Code")

# Load necessary libraries
library(lme4)      # For mixed-effects models
library(lmerTest)  # For p-values in linear mixed models
library(dplyr)     # For data manipulation
library(jsonlite)  # For reading JSON files

# Define all metrics to load from JSON (includes all available metrics)
all_metrics <- c("lines_of_code", "comment_line_count",
                 "total_comment_char_count",
                 "comment_to_code_char_ratio", "avg_identifier_length", 
                 "avg_cyclomatic_complexity", "max_cyclomatic_complexity", 
                 "pylint_score", "pylint_warnings", "pylint_conventions")

# Function to load solution data from JSON files, handling NaN/Inf
load_solution_data <- function(base_results_dir = "results") {
  tone_dirs <- list.dirs(base_results_dir, full.names = TRUE, recursive = FALSE)
  all_solutions_list <- list()
  
  for (tone_dir in tone_dirs) {
    tone <- basename(tone_dir)
    json_path <- file.path(tone_dir, "metrics_results.json")
    
    if (!file.exists(json_path)) {
      warning(paste("Missing file:", json_path))
      next
    }
    
    json_text <- readLines(json_path, warn = FALSE)
    json_text <- paste(json_text, collapse = "\n")
    json_text <- gsub("\\bNaN\\b", "null", json_text)
    json_text <- gsub("\\bInf\\b", "null", json_text)
    json_text <- gsub("\\b-Inf\\b", "null", json_text)
    
    tone_data <- fromJSON(json_text, simplifyDataFrame = FALSE)
    solutions <- unlist(lapply(tone_data, function(x) x[["solutions"]]), recursive = FALSE)
    
    tone_solutions_list <- lapply(solutions, function(sol) {
      row <- list(
        problem_id = sol[["problem_id"]],
        tone = sol[["tone_category"]],
        correct = if (sol[["pass@1"]] == 1.0) 1.0 else 0.0
      )
      for (metric in all_metrics) {
        row[[metric]] <- if (is.null(sol[[metric]])) NA else sol[[metric]]
      }
      row
    })
    
    all_solutions_list <- c(all_solutions_list, tone_solutions_list)
  }
  
  df <- bind_rows(all_solutions_list)
  df$problem_id <- as.factor(df$problem_id)
  df$tone <- as.factor(df$tone)
  
  return(df)
}

# Load data and set neutral as reference
df <- load_solution_data(base_results_dir = "results")
df$tone <- relevel(df$tone, ref = "neutral")

# RQ1: Logistic mixed-effects model for correctness
model_correct <- glmer(correct ~ tone + (1 | problem_id), 
                       data = df, family = binomial(link = "logit"))
summary(model_correct)  # Coefficients are log-odds vs. neutral, use these p-values
model_correct_null <- glmer(correct ~ 1 + (1 | problem_id), 
                            data = df, family = binomial(link = "logit"))
anova(model_correct_null, model_correct)  # Overall test of tone effect

# RQ2 and RQ3: Linear mixed-effects models for selected metrics
metrics <- c("lines_of_code", "total_comment_char_count", "comment_to_code_char_ratio", 
             "avg_identifier_length", "avg_cyclomatic_complexity", 
             "max_cyclomatic_complexity", "pylint_score", "pylint_warnings")

for (metric in metrics) {
  cat("\n=== Model for", metric, "===\n")
  formula <- as.formula(paste(metric, "~ tone + (1 | problem_id)"))
  model <- lmer(formula, data = df)
  print(summary(model))  # Coefficients are mean differences vs. neutral, use these p-values
}