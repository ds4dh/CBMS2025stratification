library(units)
library(ricu)
library(dplyr)
library(purrr)
library(tibble)
library(argparse)

OUTPUT_DIR <- "data/"
create_labels <- function(src_name) {
    target_dir <- paste0(OUTPUT_DIR, 'raw', .Platform$file.sep, "step_1_", src_name)
    # make dir if not existing
    if (!dir.exists(target_dir)) { dir.create(target_dir, recursive = TRUE) }
    src <- get(src_name)
    if (grepl("mimic", src_name, ignore.case = TRUE)) {
        diagnoses_icd_df <- as.data.frame(src$diagnoses_icd)
        procedures_icd_df <- as.data.frame(src$procedures_icd)
        diagnoses_icd_df <- mutate(diagnoses_icd_df, primary = ifelse(seq_num == 1, 1, 0))
        procedures_icd_df <- mutate(procedures_icd_df, primary = ifelse(seq_num == 1, 1, 0))
        concatenated_table <- bind_rows(diagnoses_icd_df, procedures_icd_df)
        isdf <- as.data.frame(src$icustays)
        hadm_icd9 <- left_join(concatenated_table, isdf, by = "hadm_id", relationship = "many-to-many")
        hadm_icd9 <- select(hadm_icd9, 'icustay_id', icd9_code)
        
    } else if (grepl("miiv", src_name, ignore.case = TRUE)) {
        diagnoses_icd_df <- as.data.frame(src$diagnoses_icd)
        procedures_icd_df <- as.data.frame(src$procedures_icd)
        diagnoses_icd_df <- mutate(diagnoses_icd_df, primary = ifelse(seq_num == 1, 1, 0))
        procedures_icd_df <- mutate(procedures_icd_df, primary = ifelse(seq_num == 1, 1, 0))
        concatenated_table <- bind_rows(diagnoses_icd_df, procedures_icd_df)
        isdf <- as.data.frame(src$icustays)
        hadm_icd9 <- left_join(concatenated_table, isdf, by = "hadm_id", relationship = "many-to-many")
        hadm_icd9 <- select(hadm_icd9, 'stay_id', icd_code)
        
    } else if (grepl("eicu", src_name, ignore.case = TRUE)) {
        diagnoses_icd_df <- as.data.frame(src$diagnosis)
        hadm_icd9 <- select(diagnoses_icd_df, patientunitstayid, icd9code)
    } else if (grepl("sic", src_name, ignore.case = TRUE)) {
        diagnoses_icd_df <- as.data.frame(src$cases)
        hadm_icd9 <- select(diagnoses_icd_df, CaseID, ICD10Main)
        los <- select(diagnoses_icd_df, CaseID, TimeOfStay)
        death <- select(diagnoses_icd_df, CaseID, OffsetOfDeath)
        write.csv(los, paste0(target_dir, .Platform$file.sep, src_name, "_static_los.csv"), row.names = FALSE)
        write.csv(death, paste0(target_dir, .Platform$file.sep, src_name, "_ts_death.csv"), row.names = FALSE)
    }
    # print the path for logs 
    message("Saving hadm_icd9 to ", paste0(target_dir, .Platform$file.sep, src_name, "_icds.csv"))
    write.csv(hadm_icd9, paste0(target_dir, .Platform$file.sep, src_name, "_icds.csv"), row.names = FALSE)
}

process_columns <- function(features_df, feature_type, src_name) {
    target_dir <- paste0(OUTPUT_DIR, 'raw',.Platform$file.sep, "step_1_", src_name)
    # if target_dir not existing create it 
    if (!dir.exists(target_dir)) { dir.create(target_dir, recursive = TRUE) }
    
    features <- features_df %>% select(abbreviation) %>% pull()
    missing_concepts <- c()
    for (column_name in features) {
        message("Processing ", column_name)
        # if colum name is sep3 and dataset is hirid  ! calling `susp_inf()` with `si_mode = and` requires data from both `abx` and `samp` concepts

        # create empty column_data
        column_data <- data.frame()
        tryCatch({
              
            # x<-ricu::load_concepts('alt', 'miiv', verbose = FALSE)    
            # x<-ricu::load_concepts('age', 'sic', verbose = FALSE)    
            column_data <- ricu::load_concepts(column_name, src_name, verbose = FALSE)    
        }, error = function(e) {
            message("Error processing ", column_name, ": ", e$message)
            missing_concepts <- c(missing_concepts, column_name)
            # continue to the next column
            crashed<-TRUE
        })
    

        # Check if the data has rows before proceeding
        if (nrow(column_data) > 0) {
            # Convert special data types to character if they exist
            column_data <- column_data %>%
                mutate(across(everything(), as.character))
            
            # Diagnostic print to check structure of loaded data
            print(paste0("Data for ", column_name, ":"))
            # print(head(column_data))
            
            # Save the data to CSV
            write.csv(column_data, paste0(target_dir, .Platform$file.sep, src_name, "_", feature_type, "_", column_name, ".csv"), row.names = FALSE)
            message("Saved ", column_name, " to CSV: ", paste0(target_dir, .Platform$file.sep, src_name, "_", feature_type, "_", column_name, ".csv"))
        } else {
            message("No data for ", column_name)
        }
    }
}

# Create parser object
parser <- ArgumentParser(description = 'Process some features.')
parser$add_argument('--dataset', type = 'character', required = TRUE,
                    help = 'Name of the dataset (e.g., mimic, mimic_demo)')

# Parse arguments
args <- parser$parse_args()
all_features <- ricu::load_dictionary()

# Create a dataframe from the loaded features
all_features_df <- purrr::map_df(all_features, ~tibble(abbreviation = .x$name,description = .x$description,tag = .x$tag,target = .x$target))
# save all_features_df to csv in desktop 
# make dir data 
if (!dir.exists("data")) { dir.create("data") }
desktop_path <- paste0("data","/all_features.csv")

# Save the dataframe to CSV on desktop
write.csv(all_features_df, file = desktop_path, row.names = FALSE)

static_features_df <- filter(all_features_df, target == "id_tbl")
time_series_features_df <- filter(all_features_df, target == "ts_tbl")
win_series_features_df <- filter(all_features_df, target == "win_tbl")

# Get the dataset parameter from the command line argument
src <- args$dataset

# Create the folder path using the dataset variable
folder_path <- paste0(OUTPUT_DIR, 'raw',"step_1_", src)

# Check if the folder exists
if (!dir.exists(folder_path)) {
  # Create the folder if it doesn't exist
  dir.create(folder_path)
  message("Folder created: ", folder_path)
} else {
  message("Folder already exists: ", folder_path)
}

process_columns(static_features_df, "static", src)
process_columns(win_series_features_df, "ws", src)
process_columns(time_series_features_df, "ts", src)
# Run the function
if (src != "hirid") {
    create_labels(src)
}
# Rscript preprocessing/x_prep/step1.r --dataset mimic_demo
# Rscript preprocessing/x_prep/step1.r --dataset eicu_demo