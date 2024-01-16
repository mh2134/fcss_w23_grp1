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

    # process data
    prms = params["processing"]
    if key in ["crime", "alcDeath"]:
        keys.update({key: {}})
        for var, rows in tmp2019DF.groupby(by=f'{prms["groupby"]}_code'):
            tmp = pd.pivot(data=rows, index=prms["index"], columns=prms["columns"], values=prms["values"])
            # tmp.rename(columns={"NR": ""})
            allDF = pd.concat([allDF, tmp.add_prefix(f"{var}_")], axis=1)
            keys[key].update({var: rows[f'{prms["groupby"]}_name'].unique()[0]})
    elif key in ["socNetPart", "Gini"]:
        tmp = pd.pivot(data=tmp2019DF, index=prms["index"], columns=f"{prms['columns']}_code", values=prms["values"])
        allDF = pd.merge(allDF, tmp, left_index=True, right_index=True)
        keys.update({key: tmp.columns.values[0]})

allDF = allDF.dropna(how='all')
allDF.to_csv("data/new.csv")
# # wet
# crimeDF = pd.read_csv("raw_data/Crimes_per_100tsd_estat_crim_off_cat_en.csv")
# for column in ["iccs", "freq", "unit", "geo"]:
#     crimeDF = split_column(crimeDF, column)
#
# # load income Gini
# giniDF = pd.read_csv("data/income_gini.csv")
# for column in ["indic_il", "freq", "geo"]:
#     giniDF = split_column(giniDF, column)
# # social network / internet use
# socNetPartDF = pd.read_csv("data/social_network_participation.csv")
# for column in ["indic_is", "ind_type", "freq", "unit", "geo"]:
#     socNetPartDF = split_column(socNetPartDF, column)
# # death by alcohol
# alcDeathDF = pd.read_csv("raw_data/Death_to_alcoholic_abuse_tps00140_linear.csv")
# for column in ["icd10", "age", "unit", "freq", "geo"]:
#     alcDeathDF = split_column(alcDeathDF, column)
#
#
# # get 2019 data for each factor
# crime2019DF = data_from_year(crimeDF, 2019)
# gini2019DF = data_from_year(giniDF, 2019)
# socNetPart2019DF = data_from_year(socNetPartDF, 2019)
# alcDeathDF2019DF = data_from_year(alcDeathDF, 2019)
#
# allDF = pd.DataFrame()
#
# # create columns for each crime
# crimes = {}
# for crime, crime_rows in crime2019DF.groupby(by="iccs_code"):
#     tmp = pd.pivot(data=crime_rows, index="geo_code", columns="unit_code", values="OBS_VALUE")
#     # tmp.rename(columns={"NR": ""})
#     allDF = pd.concat([allDF, tmp.add_prefix(f"{crime}_")], axis=1)
#     crimes.update({crime: crime_rows["iccs_name"].unique()[0]})
#
# # alcohol deaths
# deathtype = {}
# for death, death_rows in alcDeathDF2019DF.groupby(by="icd10_code"):
#     death_rows = death_rows[death_rows["sex"] == "T:Total"]
#     tmp = pd.pivot(data=death_rows, index="geo_code", columns="unit_code", values="OBS_VALUE")
#     # tmp.rename(columns={"NR": ""})
#     allDF = allDF.join(tmp.add_prefix(f"{death}_"))
#     deathtype.update({death: death_rows["icd10_name"].unique()[0]})
#
# # append columns of the factors
# tmp = pd.pivot(data=gini2019DF, index="geo_code", columns="indic_il_code", values="OBS_VALUE")
# allDF = pd.merge(allDF, tmp, left_index=True, right_index=True)
# tmp = pd.pivot(data=socNetPart2019DF, index="geo_code", columns="indic_is_name", values="OBS_VALUE")
# # rename column so that white space is gone to avoid crash in regression formula
# tmp["I_IUSNET"] = tmp["Internet use"]
# allDF = pd.merge(allDF, tmp, left_index=True, right_index=True)
# print(".")
# allDF.to_csv("data/old.csv")
# allDF = allDF.dropna(how='all')
# Linear regression
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# train = X_train.join(y_train)
# test = X_test.join(y_test)
log = open("data/regression.log", 'w');
# for crime_code, crime_name in crimes.items():
for crime_code, crime_name in keys["crime"].items():
    try:
        # crime_code = "ICCS09051"
        df = allDF[allDF[f"{crime_code}_P_HTHAB"].notna()]
        # throw NaNs from the rows
        # countries = df.index.tolist()
        for factor in ["I_IUSNET", "GINI_HND"]:
            df = df[df[factor].notna()]
        # df = new[new[f"{crime_code}_P_HTHAB"].notna()]
        model = smf.mixedlm(f"{crime_code}_P_HTHAB ~ I_IUSNET + GINI_HND", data=df, groups=df.index) # (1|index)
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




