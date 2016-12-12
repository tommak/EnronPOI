import pandas as pd
import numpy as np

total_payment_const = ["salary", "bonus", "long_term_incentive",
                    "deferred_income", "deferral_payments",
                     "other", "expenses" , "director_fees"]
total_stock_value_const = ["exercised_stock_options", "restricted_stock",
                    "restricted_stock_deferred"]

ordered_columns = total_payment_const + ["total_payments"] + \
            total_stock_value_const + ["total_stock_value"]


def clean(df):

    df.replace('NaN', np.nan, inplace=True)
    for c in (set(df.columns) - {"poi","email_address"}):
        df[c] = df[c].astype(np.float)

    df.drop("TOTAL", axis=0, inplace=True)
    df.drop("LOCKHART EUGENE E", axis=0, inplace=True)
    df.drop("THE TRAVEL AGENCY IN THE PARK", axis=0, inplace=True)

    df["other"] = df["loan_advances"].fillna(0) + df["other"]
    df.drop("loan_advances", axis=1, inplace=True)

    fixed_df = df.copy()
    fix_pers = {
        "BELFER ROBERT" : 1,
        "BHATNAGAR SANJAY" : -1
    }

    len_col = len(ordered_columns)
    for person, shift in fix_pers.items():
        for i in range(len_col):
            if i+shift<0 or i+shift>=len(ordered_columns):
                fixed_df.loc[person, ordered_columns[i]] = np.nan
            else:
                fixed_df.loc[person, ordered_columns[i]] = df.loc[person, ordered_columns[i+shift]]

    # fixed_df.loc["BANNANTINE JAMES M", set(total_payment_const) - {"expenses"} ] = np.nan
    # fixed_df.loc["BANNANTINE JAMES M", "total_payments"] = fixed_df.loc["BANNANTINE JAMES M", "expenses"]



    return fixed_df

def preprocess(dictionary):
    #Convert data frame to pandas data frame and perform cleaning
    df = pd.DataFrame.from_dict(dictionary, orient="index")
    fixed_df = clean(df)
    fixed_df = preprocess_df(df)

    #Convert data back to dictionary
    fixed_df.replace(np.nan, "NaN", inplace=True)
    return fixed_df.to_dict(orient="index")

def preprocess_df(fixed_df):
    fixed_df["from_poi_to_this_person_perc"] = fixed_df["from_poi_to_this_person"]/fixed_df["to_messages"]
    fixed_df["from_this_person_to_poi_perc"] = fixed_df["from_this_person_to_poi"]/fixed_df["from_messages"]

    fixed_df["shared_receipt_with_poi_perc"] = fixed_df["shared_receipt_with_poi"]/fixed_df["to_messages"]
    fixed_df["gross_payments"] = fixed_df["total_payments"] - 2*fixed_df["deferred_income"]
    fixed_df["gross_stock_value"] =  fixed_df["total_stock_value"] - 2*fixed_df["restricted_stock_deferred"]

    for c in total_payment_const:
        fixed_df[c + "_perc"] = abs(fixed_df[c]) / fixed_df["gross_payments"]

    for c in total_stock_value_const:
        fixed_df[c + "_perc"] = abs(fixed_df[c]) / fixed_df["gross_stock_value"]

    return fixed_df
