# This file is for checking out the csv file we downloaded and to clean the data where needed.

# We import pandas, a package that specializes on efficient and convenient implementations of data structures
import pandas as pd

def DataCleanup(csvname: str = "DFF") -> None:
    df = pd.read_csv(filepath_or_buffer=f"{csvname}.csv").dropna()

    date_col = df.columns[0]
    rates_col = df.columns[1]

    #convert from string to datetime object
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract min and max date
    min_date , max_date = df[date_col].min(), df[date_col].max()

    # Create business day DatetimeIndex object to reindex existing data:
    b_days = pd.bdate_range(start=min_date, end=max_date)

    # reindex
    df.set_index(keys=date_col,inplace=True)
    df = df.reindex(b_days, method="ffill")
    df.index.name = "date"

    # to convert them to their "proper" values:
    df[rates_col] /= 100.0

    #Round to 4 decimal places
    df[rates_col] = df[rates_col].round(4)

    # Rename DGS10 column for convenience
    df.rename(columns={"DGS10": "rate_pct"}, inplace=True)

    # Save as cleaned dataset csv format:
    df.to_csv(path_or_buf=f"{csvname.lower()}_clean.csv")
    return None

    
df = pd.read_csv(filepath_or_buffer="DGS10.csv").dropna()

# Take a look at the columns and if the data has been read in properly:
print(df.dtypes)
print(df.head())

date_col = df.columns[0]
rates_col = df.columns[1]

#convert from string to datetime object
df[date_col] = pd.to_datetime(df[date_col])

print(df.dtypes)

# Extract min and max date
min_date , max_date = df[date_col].min(), df[date_col].max()

# Create business day DatetimeIndex object to reindex existing data:
b_days = pd.bdate_range(start=min_date, end=max_date)

# Delete date_col:
# df.drop(columns="observation_date", inplace=True)

# reindex
df.set_index(keys=date_col,inplace=True)
df = df.reindex(b_days, method="ffill")
df.index.name = "date"

print(df.head())
# As we saw that the DGS10 values are given in percentage values, we want 

# to convert them to their "proper" values:
df[rates_col] /= 100.0

#Round to 4 decimal places
df[rates_col] = df[rates_col].round(4)

# Rename DGS10 column for convenience
df.rename(columns={"DGS10": "rate_pct"}, inplace=True)

# Save as cleaned dataset csv format:
df.to_csv(path_or_buf="dgs10_clean.csv")
