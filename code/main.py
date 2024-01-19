"""
date: 2024-01-15 18:41
author: swekia
"""

import pandas as pd
import json5 as json
from utils import *
import numpy as np

import statsmodels.formula.api as smf
import statsmodels.api as sma
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from statsmodels.graphics.factorplots import interaction_plot

# old = pd.read_csv("data/old.csv")
# new = pd.read_csv("data/new.csv")

def get_metro_region2(strin):
    if re.match(r"\w{2}_NM", strin):
        return "NM";  # Non-Metropole
    elif re.match(r"^\w{2}$", strin):
        return "WC";  # Whole Country
    elif re.match(r"\w{2}\d{3}(M|MC)$", strin):
        return "M";  # Metropole
    else:
        raise ValueError(f"not pattern match for {strin.split(':')[0]}")

df = pd.read_csv("data/df_allmetro.csv")
crimes = list(df["iccs"].unique())
df["MvsNM"] = df["metroreg"].apply(lambda x: get_metro_region2(x));

if not os.path.exists("plots"):
    os.makedirs("plots")

unempDF = pd.read_csv("raw_data/Unemployment_met_lfu3pers_linear.csv")
crimeDF = pd.read_csv("raw_data/Crime_in_Metropoles_estat_met_crim_gen_en.csv")
gdpDF = pd.read_csv("raw_data/GDP_metro_10r_3gdp_linear.csv")
popDF = pd.read_csv("raw_data/Population_met_pjangrp3_linear.csv")
unempDF = unempDF[unempDF["OBS_VALUE"].notna()]
crimeDF = crimeDF[crimeDF["OBS_VALUE"].notna()]
gdpDF = gdpDF[gdpDF["OBS_VALUE"].notna()]

def get_metro_region(strin):
    if re.match(r"\w{2}_NM", strin.split(':')[0]):
        return "NM";  # Non-Metropole
    elif re.match(r"^\w{2}$", strin.split(':')[0]):
        return "WC";  # Whole Country
    elif re.match(r"\w{2}\d{3}(M|MC)$", strin.split(':')[0]):
        return "M";  # Metropole
    else:
        raise ValueError(f"not pattern match for {strin.split(':')[0]}")


# gdpDF["MvsNM"] = gdpDF["metroreg"].apply(lambda x: get_metro_region(x));
# unempDF["MvsNM"] = gdpDF["metroreg"].apply(lambda x: get_metro_region(x));
# crimeDF["MvsNM"] = crimeDF["metroreg"].apply(lambda x: get_metro_region(x));
crimeDF["nCrimes"] = crimeDF["OBS_VALUE"]
for df in [crimeDF]:
    del df["DATAFLOW"], df["LAST UPDATE"], df["OBS_FLAG"], df["OBS_VALUE"]

gdpDF["gdp"] = gdpDF["OBS_VALUE"]
del gdpDF["OBS_VALUE"], gdpDF["DATAFLOW"], gdpDF["LAST UPDATE"], gdpDF["OBS_FLAG"], gdpDF["freq"]
popDF["population"] = popDF["OBS_VALUE"]
del popDF["OBS_VALUE"], popDF["age"], popDF["DATAFLOW"], popDF["LAST UPDATE"], popDF["OBS_FLAG"], popDF["freq"]
# sinngem. gdpDF["gdp"] * popDF["population"] / 1000
# merge gdp and population
allDF = pd.merge(left=gdpDF, right=popDF, on=["metroreg", "TIME_PERIOD"], suffixes=["_gdp", "_pop"])
# normalise gdp on per thousand persons
allDF = allDF[allDF["population"] != 0]
allDF["gdp_norm"] = allDF["gdp"] * allDF["population"] / 1000

# split unemployment data into male and female for interaction term
for df in [unempDF]:
    del df["DATAFLOW"], df["LAST UPDATE"], df["OBS_FLAG"], df["age"]
allDF = pd.merge(left=allDF, right=unempDF, on=["metroreg", "TIME_PERIOD", "sex"])
allDF["unit_unemp"] = allDF["unit"]
del allDF["unit"]
allDFf = allDF[allDF["sex"] == "F:Females"]
allDFf.loc[allDFf.index, "unemp_F"] = allDFf["OBS_VALUE"]
del allDFf["sex"]
allDFm = allDF[allDF["sex"] == "M:Males"]
allDFm.loc[allDFm.index, "unemp_M"] = allDFm["OBS_VALUE"]
del allDFm["sex"]
# unempDFf = unempDF[unempDF["sex"] == "F:Females"]
# unempDFf.loc[unempDFf.index, "unemp_F"] = unempDFf["OBS_VALUE"]
# del unempDFf["sex"]
# unempDFm = unempDF[unempDF["sex"] == "M:Males"]
# unempDFm.loc[unempDFm.index, "unemp_M"] = unempDFm["OBS_VALUE"]
# del unempDFm["sex"]

# a = pd.merge(left=unempDFf, right=unempDFm, on=["metroreg", "TIME_PERIOD", "unit", "freq"])
a = pd.merge(left=allDFf, right=allDFm, on=["metroreg", "TIME_PERIOD"])

c = pd.merge(left=a, right=crimeDF, on=["metroreg", "TIME_PERIOD"])
c["MvsNM"] = c["metroreg"].apply(lambda x: get_metro_region(x));
c.to_csv("data/table4regression.csv")
exit()
# 2do: make this more bulletproof
c["country"] = c["metroreg"].apply(lambda x: x.split(':')[0][0:2])
c["iccs_c"] = c["iccs"].apply(lambda x: x.split(':')[0])
c = c[c["MvsNM"] != "WC"]
factors = ["MvsNM", "gdp"]
interactions = ["unemp_F", "unemp_M"]
c[["gdp", "unemp_F", "unemp_M"]].corr()

c = c[c["iccs_c"].isin(crimes)]
model = smf.mixedlm(f"nCrimes ~ {' + '.join(factors)} + {':'.join(interactions)}", data=c, groups=c["country"])
result = model.fit()
# model = smf.mixedlm(f"Nr_Crimes ~ TIME_PERIOD + socnet", data=df, groups=df["geo"], re_formula="~TIME_PERIOD")
result = model.fit()
# df["crimes_pred"] = result.predict()
result.summary()

sns.boxplot(c, hue="MvsNM", y="nCrimes", x="TIME_PERIOD")
plt.show()

sns.scatterplot(c, x="unemp_F", y="nCrimes", hue="MvsNM")
plt.savefig("plots/nCrimes_unemp_F_MvsNM.png")
plt.show()
sns.scatterplot(c, x="unemp_M", y="nCrimes", hue="MvsNM")
plt.savefig("plots/nCrimes_unemp_M_MvsNM.png")
plt.show()
# sma.graphics.influence_plot(model, criterion="cooks")

print('.')
# with open("code/config.json", 'r') as f:
#     cfg = json.load(f)
#
# # store merged data in one dataframe
# allDF = pd.DataFrame()
# keys = {}
# # 2do: multiple years for random effect
# for key, params in cfg.items():
#     # load table
#     tmpDF = pd.read_csv(f"raw_data/{params['fName']}.csv")
#     # tmpDF = pd.read_csv(f"data/{params['fName']}.csv")
#     # transform columns to make them more readable
#     for column in params['columns'].split('|'):
#         tmpDF = split_column(tmpDF, column)
#
#     # filter data from 2019
#     tmp2019DF = data_from_year(tmpDF, 2019)
#
#     # sum up data for males and females
#     if key == "alcDeath":
#         tmp2019DF = tmp2019DF[tmp2019DF["sex"] == "T:Total"]
#     elif key == "povertyRisk":
#         tmp2019DF = tmp2019DF[tmp2019DF["age"] == "TOTAL:Total"]
#     elif key == "unempRate":
#         tmp2019DF = tmp2019DF[tmp2019DF["age_code"] == "Y25-74"]
#     elif key == "healthYears":
#         tmp2019DF = tmp2019DF[(tmp2019DF["sex"] == "T:Total") & (tmp2019DF["indic_he_code"] == "HLY_0")]
#
#     # process data
#     prms = params["processing"]
#     if key in ["crime", "alcDeath"]:
#         keys.update({key: {}})
#         for var, rows in tmp2019DF.groupby(by=f'{prms["groupby"]}_code'):
#             tmp = pd.pivot(data=rows, index=prms["index"], columns=prms["columns"], values=prms["values"])
#             # tmp.rename(columns={"NR": ""})
#             allDF = pd.concat([allDF, tmp.add_prefix(f"{var}_")], axis=1)
#             keys[key].update({var: rows[f'{prms["groupby"]}_name'].unique()[0]})
#     elif key in ["socNetPart", "Gini", "povertyRisk", "unempRate", "healthYears"]:
#         tmp = pd.pivot(data=tmp2019DF, index=prms["index"], columns=f"{prms['columns']}_code", values=prms["values"])
#         if key in ["povertyRisk", "unempRate", "healthYears"]:
#             allDF = pd.concat([allDF, tmp.add_prefix(f"{key}_")], axis=1)
#         else:
#             allDF = pd.merge(allDF, tmp, left_index=True, right_index=True)
#         keys.update({key: tmp.columns.values[0]})
#     else:
#         print(key)
#
# # throw lines with NaNs only
# allDF = allDF.dropna(how='all')
# # allDF.to_csv("data/new.csv")
#
# # Linear regression
# log = open("data/regression.log", 'w');
# factors = ["I_IUSNET", "GINI_HND", "povertyRisk_PC", "unempRate_PC_ACT", "healthYears_YR", "F10_RT"]
# # first of all: check correlations
# corrs = allDF[factors].corr()
#
# for crime_code, crime_name in keys["crime"].items():
#     try:
#         # crime_code = "ICCS09051"
#         df = allDF[allDF[f"{crime_code}_P_HTHAB"].notna()]
#         # throw NaNs from the rows
#         # countries = df.index.tolist()
#         for factor in factors:
#             df = df[df[factor].notna()]
#         # df = new[new[f"{crime_code}_P_HTHAB"].notna()]
#         # model = smf.mixedlm(f"{crime_code}_P_HTHAB ~ I_IUSNET + GINI_HND", data=df) #, groups=df.index) # (1|index)
#         model = smf.mixedlm(f"{crime_code}_P_HTHAB ~ {' + '.join(factors)}", data=df, groups=df.index) #, groups=df.index) # (1|index)
#         result = model.fit()
#         log.write(f"{crime_code}\t{crime_name}:\t\n"
#                   f"threw countries: {set(allDF.index) - set(df.index)}\n")
#
#         for idx, table in enumerate(result.summary().tables):
#             # log.writelines(table)
#             table.to_csv(f"data/{crime_name}-table-{idx}.csv")
#         result.params.to_csv(f"data/{crime_name}-params.csv")
#         # log.write("\n---\n")
#         # 2do: change x-axis
#         sns.regplot(x="GINI_HND", y=f"{crime_code}_P_HTHAB", data=df)
#         plt.suptitle(crime_name)
#         plt.savefig(f"plots/{crime_name}.png")
#         plt.show()
#     except IndexError:
#         print(f"no fit for {crime_code}: \t'{crime_name}'")
#         log.write(f"no fit for {crime_code}:\t'{crime_name}'\n")
# log.close()

# Regression with years (Marina data)
all_DF = pd.read_csv("data/df_all.csv")
print(all_DF.columns)

factors = ["socnet",  "h_age", "unemp", "d_alc", "poverty", "gini"]

# tab = pd.pivot_table(all_DF, columns=["iccs_d"], index="geo", values=["Nr_Crimes", "socnet"], aggfunc={"Nr_Crimes": "mean"})
# tab = pd.pivot_table(all_DF, columns="iccs_d", index="geo", values=["Nr_Crimes", "socnet"])
tab = all_DF.melt(id_vars=["geo", "TIME_PERIOD", "iccs", "iccs_d", "socnet",  "h_age", "unemp", "d_alc", "poverty", "gini"],
                  value_vars=['Nr_Crimes'], value_name="nCrimes")
tab.insert(1, "year", tab["TIME_PERIOD"])
del tab["TIME_PERIOD"]
tab["prepostpandemic"] = tab["year"].apply(lambda x: "pre" if x <= 2019 else "post")

for (crime, country), rows in tab.groupby(by=["iccs_d", "geo"]):
    corr = rows[['socnet', 'h_age', 'unemp', 'd_alc', 'poverty',
                 'gini', "year", 'nCrimes']].corr(method="pearson")
    print(f"----------------------------------\n{crime} -- {country} \n")
    print(corr)
    print('.')

tab2 = tab.groupby(['year', 'iccs'])['nCrimes'].mean()
tab3 = tab.groupby(['year', 'geo'])['nCrimes'].mean()

tab3 = tab.groupby('year')['nCrimes'].mean()
tab = pd.merge(tab, tab3, left_on="year", right_index=True)
tab["nCrimesPerYear"] = tab["nCrimes_y"]

ICCS0903 = tab[tab["iccs"] == "ICCS0903"]

sns.scatterplot(ICCS0903, x="year", y="nCrimesPerYear", hue="geo")
plt.show()

sns.scatterplot(tab, x="year", y="nCrimesPerYear", hue="iccs")
plt.show()


NUM_COLORS = len(ICCS0903["geo"].unique())
import numpy as np
cm = plt.get_cmap('gist_rainbow')
fig, ax = plt.subplots()
ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
for idx, country in enumerate(ICCS0903["geo"].unique()):
    df = ICCS0903[(ICCS0903["geo"] == country)]
    plt.plot(df["year"].values, df["nCrimes"].values, label=country)
plt.legend(ncol=5)
plt.show()
# for country in df["geo"].unique():
#     plt.plot(df["year"].values, df["nCrimes"].values)
# plt.show()

sns.lineplot(tab, x="year", y="nCrimes", hue="geo")
plt.plot(tab["year"].values, df["nCrimes"].values, label=country)

for country, grp in df.groupby(by="geo"):
    grp.plot(grp.year.values, grp.nCrimes.values, ax=ax, label=country)
plt.show()

# for year, rows in ICCS0903.groupby(by="year"):
for year, rows in ICCS0903.groupby(by="prepostpandemic"):
    df = rows
    # print(year)
    # break
    # sns.scatterplot(df, y="nCrimes", hue="geo", x="socnet")
    # plt.suptitle(year)
    # plt.show()
    #
    # sns.relplot(df, x="unemp", y="nCrimes", hue="geo", size="socnet")
    # plt.suptitle(year)
    # plt.show()
    #
    # sns.histplot(df, x="nCrimes", hue="geo", multiple="dodge")
    # plt.suptitle(year)
    # plt.show()


    plt.pie(df["nCrimes"], labels=df["geo"])
    plt.suptitle(f"{rows['iccs_d'].unique()[0]} ({year})")
    plt.show()

    # sns.scatterplot(df, y="nCrimes", hue="geo", x="poverty")
    # plt.suptitle(year)
    # plt.show()
    #
    # sns.scatterplot(df, y="nCrimes", hue="geo", x="gini")
    # plt.suptitle(year)
    # plt.show()

for crime, rows in all_DF.groupby(by="iccs_d"):
    if crime == 'Acts against computer systems':
        print("d")
    # break
    df = rows[rows["iccs_d"] == crime]
    df = df[df["Nr_Crimes"].notna()]
    for factor in factors:
        df = df[df[factor].notna()]
    model = smf.ols(f"Nr_Crimes ~ TIME_PERIOD + {' + '.join(factors)}", data=df, groups=df["geo"], re_formula="~TIME_PERIOD")
    result = model.fit()
    # model = smf.mixedlm(f"Nr_Crimes ~ TIME_PERIOD + socnet", data=df, groups=df["geo"], re_formula="~TIME_PERIOD")
    result = model.fit()
    df["crimes_pred"] = result.predict()
    # fig = sma.graphics.influence_plot(result, criterion="cooks")
    sns.scatterplot(data=df, x="Nr_Crimes", y="crimes_pred", hue="geo")
    plt.show()

    print(f"------------------------------------\n{crime.upper()}\n")
    print(result.params)
    print(result.summary())

    print('\n')
