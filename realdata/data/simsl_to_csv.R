# Extract the IWPC warfarin data (warfarin.RData) from the simsl source package to CSV.
# Usage: Rscript simsl_to_csv.R <path to simsl/data/warfarin.RData> <output csv>
e <- new.env(); load(commandArgs(TRUE)[1], envir = e); w <- e$warfarin
write.csv(data.frame(Y = w$A, INR = w$INR, w$X), commandArgs(TRUE)[2], row.names = FALSE)
