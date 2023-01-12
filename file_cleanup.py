import pandas as pd
import os
import sys

hart = pd.read_csv('gz2_hart16.csv')
maps = pd.read_csv('gz2_filename_mapping.csv')
files = os.listdir('images_gz2/images/')

not_in_files = []

# Look for files in gz2_hart16.csv that are not in the list of image files
# add their ids to the not_in_files list
for i in hart.index:
    if i % 25000 == 0: # Counter for feedback in the output
        print(i, len(not_in_files))
    
    # Matching row in the filename mapping file
    row = maps.loc[maps['objid'] == hart['dr7objid'][i]] 

    if str(row['asset_id'].item()) + '.jpg' not in files: 
        not_in_files.append(hart['dr7objid'][i])

print('Found ' + str(len(not_in_files)) + ' mismatched ids.')

while True:
    br = input('>> Remove and write to new file?\n(y|n) > ')

    if br == 'n':
        sys.exit(0)
    if br == 'y':
        break
    else:
        print('Input y for yes or n for no.')

# Remove lines with ids in not_in_files
for f in not_in_files:
    hart.drop(hart[hart['dr7objid'] == f].index, inplace=True)

# Write data to new file
hart.to_csv('gz2_hart16_cleaned.csv' , index=False)