"""
date: 2024-01-15 18:41
author: swekia
"""

import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
for crime, crimes in crime2019DF.groupby(by="iccs_code"):
    tmp = pd.pivot(data=crimes, index="geo_code", columns="unit_code", values="OBS_VALUE")
    tmp.rename(columns={"NR": ""})
    allDF = pd.concat([allDF, tmp.add_prefix(f"{crime}_")], axis=1)

# append columns of the factors
tmp = pd.pivot(data=gini2019DF, index="geo_code", columns="indic_il_code", values="OBS_VALUE")
allDF = pd.merge(allDF, tmp, left_index=True, right_index=True)
tmp = pd.pivot(data=socNetPart2019DF, index="geo_code", columns="indic_is_name", values="OBS_VALUE")
allDF = pd.merge(allDF, tmp, left_index=True, right_index=True)
print(".")

# Linear Regression
mdl = LinearRegression()
# predict ICCS09051_P_HTHAB
# Internet use GINI_HND
X = allDF[["Internet use", "GINI_HND"]]
y = allDF["ICCS09051_P_HTHAB"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

mdl.fit(X_train, y_train)

y_test = mdl.predict(X_test)