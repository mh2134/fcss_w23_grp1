"""
date: 2024-01-15 18:41
author: swekia
"""

import pandas as pd
import json5 as json
from utils import *

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

old = pd.read_csv("data/old.csv")
new = pd.read_csv("data/new.csv")

if not os.path.exists("plots"):
    os.makedirs("plots")

with open("code/config.json", 'r') as f:
    cfg = json.load(f)

# store merged data in one dataframe
allDF = pd.DataFrame()
keys = {}
# 2do: multiple years for random effect
for key, params in cfg.items():
    # load table
    tmpDF = pd.read_csv(f"raw_data/{params['fName']}.csv")
    # tmpDF = pd.read_csv(f"data/{params['fName']}.csv")
    # transform columns to make them more readable
    for column in params['columns'].split('|'):
        tmpDF = split_column(tmpDF, column)

    # filter data from 2019
    tmp2019DF = data_from_year(tmpDF, 2019)

    # sum up data for males and females
    if key == "alcDeath":
        tmp2019DF = tmp2019DF[tmp2019DF["sex"] == "T:Total"]
    elif key == "povertyRisk":
        tmp2019DF = tmp2019DF[tmp2019DF["age"] == "TOTAL:Total"]
    elif key == "unempRate":
        tmp2019DF = tmp2019DF[tmp2019DF["age_code"] == "Y25-74"]
    elif key == "healthYears":
        tmp2019DF = tmp2019DF[(tmp2019DF["sex"] == "T:Total") & (tmp2019DF["indic_he_code"] == "HLY_0")]

    # process data
    prms = params["processing"]
    if key in ["crime", "alcDeath"]:
        keys.update({key: {}})
        for var, rows in tmp2019DF.groupby(by=f'{prms["groupby"]}_code'):
            tmp = pd.pivot(data=rows, index=prms["index"], columns=prms["columns"], values=prms["values"])
            # tmp.rename(columns={"NR": ""})
            allDF = pd.concat([allDF, tmp.add_prefix(f"{var}_")], axis=1)
            keys[key].update({var: rows[f'{prms["groupby"]}_name'].unique()[0]})
    elif key in ["socNetPart", "Gini", "povertyRisk", "unempRate", "healthYears"]:
        if key == "povertyRisk":
            print("a")
        tmp = pd.pivot(data=tmp2019DF, index=prms["index"], columns=f"{prms['columns']}_code", values=prms["values"])
        if key in ["povertyRisk", "unempRate", "healthYears"]:
            allDF = pd.concat([allDF, tmp.add_prefix(f"{key}_")], axis=1)
        else:
            allDF = pd.merge(allDF, tmp, left_index=True, right_index=True)
        keys.update({key: tmp.columns.values[0]})
    else:
        print(key)

allDF = allDF.dropna(how='all')
# allDF.to_csv("data/new.csv")

# Linear regression
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# train = X_train.join(y_train)
# test = X_test.join(y_test)
log = open("data/regression.log", 'w');
factors = ["I_IUSNET", "GINI_HND", "povertyRisk_THS_PER", "unempRate_PC_ACT", "healthYears_YR"]
# first of all: check correlations
corrs = allDF[factors].corr()

for crime_code, crime_name in keys["crime"].items():
    try:
        # crime_code = "ICCS09051"
        df = allDF[allDF[f"{crime_code}_P_HTHAB"].notna()]
        # throw NaNs from the rows
        # countries = df.index.tolist()
        for factor in factors:
            df = df[df[factor].notna()]
        # df = new[new[f"{crime_code}_P_HTHAB"].notna()]
        # model = smf.mixedlm(f"{crime_code}_P_HTHAB ~ I_IUSNET + GINI_HND", data=df) #, groups=df.index) # (1|index)
        model = smf.mixedlm(f"{crime_code}_P_HTHAB ~ {' + '.join(factors)}", data=df, groups=df.index) #, groups=df.index) # (1|index)
        result = model.fit()
        log.write(f"{crime_code}\t{crime_name}:\t\n"
                  f"threw countries: {set(allDF.index) - set(df.index)}\n")

        for idx, table in enumerate(result.summary().tables):
            # log.writelines(table)
            table.to_csv(f"data/{crime_name}-table-{idx}.csv")
        result.params.to_csv(f"data/{crime_name}-params.csv")
        # log.write("\n---\n")

        sns.regplot(x="GINI_HND", y=f"{crime_code}_P_HTHAB", data=df)
        plt.suptitle(crime_name)
        plt.savefig(f"plots/{crime_name}.png")
        plt.show()
    except IndexError:
        print(f"no fit for {crime_code}: \t'{crime_name}'")
        log.write(f"no fit for {crime_code}:\t'{crime_name}'\n")
log.close()




