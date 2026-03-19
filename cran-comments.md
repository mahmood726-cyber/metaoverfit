## Resubmission
This is a resubmission. In this version I have:

* Added references to statistical methods in DESCRIPTION with proper DOI formatting  
  (Harrell et al. 1996 and Higgins & Thompson 2002)
* Added comprehensive \value documentation for all exported functions including print.metaoverfit
* Replaced cat() with message() and redesigned sample_size_recommendation() to return 
  a proper S3 object with print method
* Added Dr. Laiba Khan as co-author
* Fixed .Rbuildignore file (removed comments that were causing regex errors)

## Test environments
* local OS X install, R 4.3.1 [or your R version]
* win-builder (devel and release)

## R CMD check results
0 errors | 0 warnings | 0 notes
