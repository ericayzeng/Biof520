# install dependencies
pip install -r requirements.txt

# convert rds files to csv
Rscript convert_rds_csv.R

# preprocess data
python preprocess.py

# train/run model
python model.py