# This file is for checking out the csv file we downloaded and to clean the data where needed.

# We import pandas, a package that specializes on efficient and convenient implementations of data structures
import pandas as pd


def DataCleanup(
    filename: str = "../data/DFF.csv", print_statements: bool = True
) -> None:

    # Read in raw data:
    df = pd.read_csv(filepath_or_buffer=filename).dropna()

    # Get the names for the date and rates-values columns (dates in first column, values in second):
    date_col = df.columns[0]
    rates_col = df.columns[1]

    # convert from string to datetime object
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract min and max date
    min_date, max_date = df[date_col].min(), df[date_col].max()

    # Create business day DatetimeIndex object to reindex existing data:
    b_days = pd.bdate_range(start=min_date, end=max_date)

    # reindex
    df.set_index(keys=date_col, inplace=True)
    df = df.reindex(b_days, method="ffill")
    df.index.name = "date"

    # to convert them to their "proper" values:
    df[rates_col] /= 100.0

    # Round to 4 decimal places
    df[rates_col] = df[rates_col].round(4)

    # some string mainpulation to get the correct short-rate name:
    short_rate_name = filename.split("/")[-1].split(".csv")[0]
    print(short_rate_name)

    # Rename DGS10 column for convenience
    df.rename(columns={short_rate_name: "rate_pct"}, inplace=True)

    # final print statement:
    if print_statements:
        print(
            f"The datatypes for each column after datetime conversion are: \n {df.dtypes} \n"
        )
        print(f"The top 10 rows are : \n {df.head(n=10)} \n")

    # some string manipulation to store the file correctly:
    path = "/".join(filename.split("/")[0:-1])

    print(
        f"The cleaned dataset {short_rate_name}_clean.csv has been saved at: \n {path+"/"+short_rate_name}_clean.csv \n"
    )
    # Save as cleaned dataset csv format:
    df.to_csv(path_or_buf=f"{path+"/"+short_rate_name}_clean.csv")

    return None


if __name__ == "__main__":
    # This only runs if we run this file explicitly! If we import data_cleaning.py this will not be called!
    DataCleanup()
