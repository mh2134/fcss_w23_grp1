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




popDF = pd.read_csv("raw_data/Population_met_pjangrp3_linear.csv")
unempDF = pd.read_csv("raw_data/Unemployment_met_lfu3pers_linear.csv")
unempDF = unempDF[unempDF["OBS_VALUE"].notna()]
crimeDF = pd.read_csv("raw_data/Crime_in_Metropoles_estat_met_crim_gen_en.csv")
crimeDF = crimeDF[crimeDF["OBS_VALUE"].notna()]
gdpDF = pd.read_csv("raw_data/GDP_metro_10r_3gdp_linear.csv")
gdpDF = gdpDF[gdpDF["OBS_VALUE"].notna()]

def get_metro_region(strin, from_country=False):
    if not from_country:
        if re.match(r"\w{2}_NM", strin.split(':')[0]):
            return "NM";  # Non-Metropole
        elif re.match(r"^\w{2}$", strin.split(':')[0]):
            return "WC";  # Whole Country
        elif re.match(r"\w{2}\d{3}(M|MC)$", strin.split(':')[0]):
            return "M";  # Metropole
        else:
            raise ValueError(f"not pattern match for {strin.split(':')[0]}")
    else:
        if re.match("DEG1:Cities", strin):
            return "M";
        elif re.match("DEG2:Towns and suburbs", strin) or re.match("DEG3:Rural areas", strin):
            return "NM";
        else:
            raise ValueError(f"not pattern match for {strin}")

for df in [unempDF]:
    del df["DATAFLOW"], df["LAST UPDATE"], df["OBS_FLAG"]

# gdpDF["MvsNM"] = gdpDF["metroreg"].apply(lambda x: get_metro_region(x));
# unempDF["MvsNM"] = gdpDF["metroreg"].apply(lambda x: get_metro_region(x));
# crimeDF["MvsNM"] = crimeDF["metroreg"].apply(lambda x: get_metro_region(x));
crimeDF["nCrimes"] = crimeDF["OBS_VALUE"]
for df in [crimeDF]:
    del df["DATAFLOW"], df["LAST UPDATE"], df["OBS_FLAG"], df["OBS_VALUE"]

gdpDF["gdp"] = gdpDF["OBS_VALUE"]
del gdpDF["OBS_VALUE"], gdpDF["DATAFLOW"], gdpDF["LAST UPDATE"], gdpDF["OBS_FLAG"], gdpDF["freq"]
# isolate only the GDP values we need
gdpDF = gdpDF[gdpDF["unit"].isin(["PPS_EU27_2020_HAB:Purchasing power standard (PPS, EU27 from 2020), per inhabitant",
                                  "EUR_HAB:Euro per inhabitant"])]

popDF["population"] = popDF["OBS_VALUE"]
popDF = popDF[popDF["age"] == "TOTAL:Total"]
del popDF["OBS_VALUE"], popDF["age"], popDF["DATAFLOW"], popDF["LAST UPDATE"], popDF["OBS_FLAG"], popDF["freq"]
# sinngem. gdpDF["gdp"] * popDF["population"] / 1000
# merge gdp and population
allDF = pd.merge(left=gdpDF, right=popDF[popDF["sex"] == "T:Total"], on=["metroreg", "TIME_PERIOD"],
                 suffixes=["_gdp", "_pop"])
# normalise gdp on per thousand persons
allDF = allDF[allDF["population"] != 0]
allDF.loc[allDF.index, "gdp_norm"] = allDF["gdp"] * allDF["population"] / 1000
# merge crimes on population
crimeDF = pd.merge(left=crimeDF, right=popDF[popDF["sex"] == "T:Total"], on=["metroreg", "TIME_PERIOD"],
                   suffixes=["_crime", "_pop"])
crimeDF = crimeDF[crimeDF["population"] != 0]
crimeDF["nCrimes_norm"] = crimeDF["nCrimes"] / crimeDF["population"]
del crimeDF["sex"], crimeDF["unit_crime"], crimeDF["unit_pop"], \
    crimeDF["freq"], crimeDF["population"], crimeDF["nCrimes"]
del allDF["unit_pop"]
allDF = pd.merge(left=allDF, right=crimeDF, on=["metroreg", "TIME_PERIOD"])

# split unemployment data into male and female for interaction term
# 2do: hier könnte man die 20 bis
unempDF = unempDF[unempDF["age"] == "Y_GE15:15 years or over"]
unempDF["unemp"] = unempDF["OBS_VALUE"]
del unempDF["freq"], unempDF["OBS_VALUE"], unempDF["age"]
unempDF = pd.merge(left=popDF, right=unempDF, on=["metroreg", "TIME_PERIOD", "sex"], suffixes=["_pop", "_unemp"])
unempDF["unemp_norm"] = unempDF["unemp"] * 1000 / unempDF["population"]
del unempDF["population"], unempDF["unemp"]
femalesUnempDF = unempDF[unempDF["sex"] == "F:Females"]
del femalesUnempDF["sex"]
malesUnempDF = unempDF[unempDF["sex"] == "M:Males"]
del malesUnempDF["sex"]

for df in [femalesUnempDF, malesUnempDF]:
    del df["unit_pop"], df["unit_unemp"]
# a = pd.merge(left=unempDFf, right=unempDFm, on=["metroreg", "TIME_PERIOD", "unit", "freq"])
unempDF = pd.merge(left=femalesUnempDF, right=malesUnempDF, on=["metroreg", "TIME_PERIOD"],
                   suffixes=["_F", "_M"])
allDF = pd.merge(left=allDF, right=unempDF, on=["metroreg", "TIME_PERIOD"])

# now, create some helper variables to make values more accessible
allDF["MvsNM"] = allDF["metroreg"].apply(lambda x: get_metro_region(x));
# 2do: make this more bulletproof
allDF["country"] = allDF["metroreg"].apply(lambda x: x.split(':')[0][0:2])
allDF["iccs_code"] = allDF["iccs"].apply(lambda x: x.split(':')[0])
allDF["iccs_name"] = allDF["iccs"].apply(lambda x: ' '.join(x.split(':')[1:]))
allDF.loc[allDF.index, "unit_gdp_code"] = allDF["unit_gdp"].apply(lambda x: x.split(':')[0])
allDF.loc[allDF.index, "unit_gdp_name"] = allDF["unit_gdp"].apply(lambda x: ' '.join(x.split(':')[1:]))
del allDF["unit_gdp"]

incomeDF = pd.read_csv("raw_data/Income_urbanisation_ilc_di17_linear.csv");
incomeDF = incomeDF[(incomeDF["age"] == "TOTAL:Total") &
                    incomeDF["unit"].isin(["EUR:Euro", "PPS:Purchasing power standard (PPS)"]) &
                    (incomeDF["sex"] == "T:Total")]
incomeDF["unit_income"] = incomeDF["unit"]
incomeDF["income"] = incomeDF["OBS_VALUE"]
incomeDF["MvsNM"] = incomeDF["deg_urb"].apply(lambda x: get_metro_region(x, from_country=True))
incomeDF["country"] = incomeDF["geo"].apply(lambda x: x.split(':')[0])
del incomeDF["geo"], incomeDF["OBS_VALUE"], incomeDF["DATAFLOW"], incomeDF["LAST UPDATE"], incomeDF["OBS_FLAG"], \
    incomeDF["freq"], incomeDF["age"], \
    incomeDF["sex"], incomeDF["unit"], incomeDF["deg_urb"]

cols = list(incomeDF.columns)
cols.remove("income")
# merge values of non-metropole regions together (NM)
incomeDF = incomeDF.groupby(cols).mean().reset_index()
allDF = pd.merge(left=allDF, right=incomeDF, on=['TIME_PERIOD', 'MvsNM', 'country'])

allDF.to_csv("data/table4regression.csv")

regDF = allDF[(allDF["MvsNM"] != "WC") & (allDF["unit_gdp_code"] == "PPS_EU27_2020_HAB") &
              (allDF["unit_income"] == "PPS:Purchasing power standard (PPS)") &
              (allDF["indic_il"] == "MED_E:Median equivalised net income")]
factors = ["MvsNM", "gdp_norm", "income"]
interactions = ["unemp_norm_F", "unemp_norm_M"]
# correlation of factors
sns.heatmap(regDF[["nCrimes_norm", "gdp_norm", "income", "unemp_norm_F", "unemp_norm_M"]].corr(method="spearman"),
            cmap="seismic", vmin=-1, vmax=1)
plt.tight_layout()
plt.savefig("plots/correlation_factors.png")
plt.show()

# allDF = allDF[allDF["iccs_c"].isin(crimes)]
for crime, data in regDF.groupby(by="iccs_code"):
    print(f"regression for crime {crime} ({data['iccs_name'].unique()[0]})")
    model = smf.mixedlm(f"nCrimes_norm ~ {' + '.join(factors)} + {':'.join(interactions)}",
                        data=data, groups=data["country"])
    # model = smf.mixedlm(f"nCrimes_norm ~ TIME_PERIOD + {' + '.join(factors)} + {':'.join(interactions)}",
    #                     data=data, groups=data["country"], re_formula="~ TIME_PERIOD")
    result = model.fit()
    # df["crimes_pred"] = result.predict()
    print(result.summary())
    with open(f"data/Model-{'-'.join(data['iccs_name'].unique()[0].split(' '))}.txt", 'w') as f:
        # Write the summary to the file
        f.write(result.summary().as_text())
    result.params.to_csv(f"data/Params-{'-'.join(data['iccs_name'].unique()[0].split(' '))}.txt", sep="\t")

    sns.boxplot(data, hue="MvsNM", y="nCrimes_norm", x="TIME_PERIOD")
    plt.suptitle(f"{data['iccs_name'].unique()[0]} ({crime})")
    plt.savefig(f"plots/Timeline_nCrimes_MvsNM_{data['iccs_name'].unique()[0]}.png")
    plt.show()

    sns.scatterplot(data, x="unemp_norm_F", y="nCrimes_norm", hue="MvsNM")
    plt.suptitle(f"{data['iccs_name'].unique()[0]} ({crime})")
    plt.savefig(f"plots/nCrimes_unemp_F_MvsNM_{data['iccs_name'].unique()[0]}.png")
    plt.show()
    sns.scatterplot(data, x="unemp_norm_M", y="nCrimes_norm", hue="MvsNM")
    plt.suptitle(f"{data['iccs_name'].unique()[0]} ({crime})")
    plt.savefig(f"plots/nCrimes_unemp_M_MvsNM_{data['iccs_name'].unique()[0]}.png")
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
