
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from pymer4.models import Lmer
import pymer4.stats as pymers
from scipy.stats import zscore


def print_min_max(dframe):
    print("\t\t\tmin\t\tmax\n"
          f'unemp:\t\t{dframe["unemp_per_k"].min()}\t\t{dframe["unemp_per_k"].max()}\n'
          f'density:\t{dframe["density"].min():.2f}\t{dframe["density"].max():.2f}\n'
          f'gdp:\t\t{dframe["gdp"].min()}\t{dframe["gdp"].max()}\n')
    return;

df = pd.read_csv("data/normalized_data.csv")
df.loc[:, "year"] = df["TIME_PERIOD"]
del df["TIME_PERIOD"]
df["crimes"] = df['crime/population']
df["unemp_per_k"] = df['unemp per Tsd.']
df = df[~df["unemp_per_k"].isna()]
factors = ["gdp", "unemp_per_k", "density"]

# # throw countries that have too few values
# countries = []
# for country, data in df.groupby(by="country"):
#     if len(data["year"].unique()) <= 6:
#         countries.append(country)
# print(countries)
# df = df[~df["country"].isin(countries)]

# # z-score
# df_z = df
# for var in ["unemp_per_k", "density", "gdp"]:
#     df_z.loc[:, var] = zscore(df_z[var])

# # just in case
# train, test = train_test_split(df, test_size=0.2, random_state=42)

# make a separate model for each crime type
for crime, data in df.groupby(by="iccs"):

    crime_name = data["iccs_d"].unique()[0]

    # throw outlier countries
    lenbef = len(data)
    data = data[data["crimes"] <= (data["crimes"].median() + 3 * data["crimes"].std())]
    print(f"throwing {lenbef-len(data)} rows with outliers")
    # throw countries that have too few data points (now)
    countries = []
    for country, rows in data.groupby(by="country"):
        if len(rows["year"].unique()) <= 3:
            countries.append(country)
    print(f"throwing countries {countries} with too few datapoints")
    data = data[~data["country"].isin(countries)]

    # boxplots
    # plot number of crimes by country
    sns.boxplot(data, x="crimes", hue="country")
    plt.legend(ncol=2)
    plt.xlabel(f"Number of Crimes: {crime_name}")
    plt.tight_layout()
    plt.savefig(f"plots/{crime}_{crime_name.split(' ')[0]}_Boxplots_Country.png")
    plt.show()
    # plot number of crimes in total
    sns.boxplot(data, x="crimes")
    plt.legend(ncol=2)
    plt.xlabel(f"Number of Crimes: {crime_name}")
    plt.tight_layout()
    plt.savefig(f"plots/{crime}_{crime_name.split(' ')[0]}_Boxplots.png")
    plt.show()

    # overview on the data
    print(f"----------- {crime_name} -----------")
    print_min_max(data)

    # model
    model = Lmer("crimes ~ gdp + unemp_per_k + density + (1|country)", data=data)
    result = model.fit() # print(model.coefs) # print(model.fixef.head(5))
    print(model.summary())
    # model.plot_summary()
    # plt.show()

    # R^2
    R_squared = pymers.rsquared(model.data.crimes, model.residuals)
    print(f"R2c:\t\t{R_squared}\n")
    # df_res = len(data) - len(model.fixef.columns) - 1
    # R_squared_adj = pymers.rsquared_adj(R_squared, nobs=len(data), df_res=df_res)
    # print(f"R2:\t\t{R_squared}\n"
    #       f"R2_adj:\t{R_squared_adj}")

    # plot regression lines
    for (var, colour) in [("unemp_per_k", "tab:green"), ("gdp", "tab:blue"), ("density", "tab:orange")]:
        sns.regplot(x=var, y="crimes", data=model.data, fit_reg=True, color=colour)
        plt.suptitle(crime_name)
        plt.savefig(f"plots/{crime}_{crime_name.split(' ')[0]}_{var}_reg.png")
        plt.show()
        sns.scatterplot(data, x=var, y="crimes", hue="country")
        plt.suptitle(crime_name)
        plt.savefig(f"plots/{crime}_{crime_name.split(' ')[0]}_{var}_scatter.png")
        plt.show()

        # plot regression lines by country
        model.plot(param=var, ylabel="crimes", xlabel=var, grps=model.fixef.index.values.tolist())
        plt.suptitle(crime_name)
        plt.legend(model.fixef.index.values.tolist())
        plt.savefig(f"plots/{crime}_{crime_name.split(' ')[0]}_{var}_reg_byCountry.png")
        plt.show()

    print('')

