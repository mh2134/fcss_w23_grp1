"""
date: 2024-01-15 18:41
author: swekia
"""

import pandas as pd
from utils import *

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

if not os.path.exists("plots"):
    os.makedirs("plots")

crimeDF = pd.read_csv("data/crime.csv")
for column in ["iccs", "freq", "unit", "geo"]:
    crimeDF = split_column(crimeDF, column)

# load income Gini
giniDF = pd.read_csv("data/income_gini.csv")
for column in ["indic_il", "freq", "geo"]:
    giniDF = split_column(giniDF, column)
socNetPartDF = pd.read_csv("data/social_network_participation.csv")
for column in ["indic_is", "ind_type", "freq", "unit", "geo"]:
    socNetPartDF = split_column(socNetPartDF, column)

crime2019DF = data_from_year(crimeDF, 2019)
gini2019DF = data_from_year(giniDF, 2019)
socNetPart2019DF = data_from_year(socNetPartDF, 2019)

allDF = pd.DataFrame()

# create columns for each crime
crimes = {}
for crime, crime_rows in crime2019DF.groupby(by="iccs_code"):
    tmp = pd.pivot(data=crime_rows, index="geo_code", columns="unit_code", values="OBS_VALUE")
    # tmp.rename(columns={"NR": ""})
    allDF = pd.concat([allDF, tmp.add_prefix(f"{crime}_")], axis=1)
    crimes.update({crime: crime_rows["iccs_name"].unique()[0]})

# append columns of the factors
tmp = pd.pivot(data=gini2019DF, index="geo_code", columns="indic_il_code", values="OBS_VALUE")
allDF = pd.merge(allDF, tmp, left_index=True, right_index=True)
tmp = pd.pivot(data=socNetPart2019DF, index="geo_code", columns="indic_is_name", values="OBS_VALUE")
tmp["internet_use"] = tmp["Internet use"]
allDF = pd.merge(allDF, tmp, left_index=True, right_index=True)
print(".")


# Linear regression
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# train = X_train.join(y_train)
# test = X_test.join(y_test)
log = open("data/regression.log", 'w');
for crime_code, crime_name in crimes.items():
    try:
        model = smf.mixedlm(f"{crime_code}_P_HTHAB ~ internet_use + GINI_HND", data=allDF, groups=allDF.index)
        result = model.fit()
        # log.write(result.summary())
        # log.write(result.params)

        sns.regplot(x="GINI_HND", y=f"{crime_code}_P_HTHAB", data=allDF)
        plt.suptitle(crime_name)
        plt.savefig(f"plots/{crime_name}.png")
        plt.show()
    except IndexError:
        print(f"no fit for {crime_code}: \t'{crime_name}'")
        log.write(f"no fit for {crime_code}:\t'{crime_name}'\n")
log.close()




