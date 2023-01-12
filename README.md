# Classification-of-Galaxies
Repository for classifying galaxy types from photos.

## Report
First step data structuring, creating the dataset from images.
Difficulties: 
 - missing images in the downloaded data
 --> Remove every row from Table 1 (gz2_hart16.csv) that contains an id not present in the image files
 - missing ids in the csv files
 --> Just use files with ids in the Table

first classifications:
 - smooth or features or star?