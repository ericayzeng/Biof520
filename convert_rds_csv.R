# UROMOL paths
input_path_UROMOL <- "data/UROMOL_TaLG.teachingcohort.rds"
output_path_UROMOL <- "data/UROMOL_TaLG.csv"

# knowles paths
input_path_knowles <- "data/knowles_matched_TaLG_final.rds"
output_path_knowles <- "data/knowles_matched_TaLG_final.csv"

convert_rds_to_csv <- function(input_path, output_path) {
  obj <- readRDS(input_path)

  if (!is.data.frame(obj)) {
    obj <- as.data.frame(obj)
  }

  write.csv(obj, output_path, row.names = FALSE)
  cat("Converted", input_path, "to", output_path, "\n")
}

# convert both files
convert_rds_to_csv(input_path_UROMOL, output_path_UROMOL)
convert_rds_to_csv(input_path_knowles, output_path_knowles)