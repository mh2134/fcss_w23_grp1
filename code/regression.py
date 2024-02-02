"""Linear Mixed Effects Model"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from pymer4.models import Lmer
import pymer4.stats as pymers
import numpy as np

# axis limits for the final plots
from fcss_w23_grp1.code.utils import throw_outliers

axlims = {
    "ICCS0401": {  # robbery
        "unemployment": {
            "x0": -5, "x1": 1200, "y0": -30, "y1": 2300
        },
        "density": {
            "x0": -70/10, "x1": 4200/10, "y0": -70, "y1": 2300
        },
        "gdp": {
            "x0": 20, "x1": 400, "y0": -15, "y1": 2300
        }
    },
    "ICCS050211": {  # theft
        "unemployment": {
            "x0": -20, "x1": 1200, "y0": -100, "y1": 5000
        },
        "density": {
            "x0": -100/10, "x1": 4200/10, "y0": -100, "y1": 5000
        },
        "gdp": {
            "x0": 20, "x1": 400, "y0": -100, "y1": 5000
        }
    },
    "ICCS05012": {  # burglary
        "unemployment": {
            "x0": -22, "x1": 1200, "y0": 0, "y1": 7100
        },
        "density": {
            "x0": -70/10, "x1": 4200/10, "y0": 0, "y1": 7100
        },
        "gdp": {
            "x0": 20, "x1": 200, "y0": 0, "y1": 7100
        }
    },
    "ICCS0101": {  # homicide
        "unemployment": {
            "x0": -12, "x1": 1200, "y0": -2, "y1": 40
        },
        "density": {
            "x0": -2/10, "x1": 2700/10, "y0": -2, "y1": 40
        },
        "gdp": {
            "x0": 20, "x1": 420, "y0": -2, "y1": 40
        }
    }
}

# axis labels of the predictors
predictor_labels = {
    "unemployment": "Unemployment Rate (per 1000 citizens)",
    "density": "Urban Density (inhabitants per 10 km²)",
    "gdp": "GDP (Purchasing Power Standard, % EU average)"
}

# short names for the crime types
crime_labels = {
    "ICCS0101": "Homicides",
    "ICCS0401": "Robberies",
    "ICCS05012": "Burglaries",
    "ICCS050211": "Thefts"
}

# read in preprocessed data
df = pd.read_csv("data/normalized_data.csv")
# rename colums
df.rename(columns={"TIME_PERIOD": "year", "crime/population": "crimes", "unemp per Tsd.": "unemployment"}, inplace=True)
# df.loc[:, "year"] = df["TIME_PERIOD"]
# del df["TIME_PERIOD"]
# df["crimes"] = df['crime/population']
# df["unemployment"] = df['unemp per Tsd.']
# throw NaN values in unemployment
df = df[~df["unemployment"].isna()]

# define the predictor variables
predictors = ["gdp", "unemployment", "density"]

# rescale urban density
df.loc[:, "density"] = df["density"] / 10

# define unique colour palette for each country with persistent colour across plots
palette = sns.color_palette(cc.glasbey_bw_minc_20, n_colors=len(df.country.unique()))
color_dict = {}
for index, country in enumerate(df.country.unique()):
    color_dict.update({country: palette[index]})
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[color_dict[c] for c in df.country.unique()])

# make a separate model for each crime type
# homicide: "ICCS0101" | robbery: "ICCS0401" | theft: "ICCS050211" | burglary: "ICCS05012"
for crime, data in df.groupby(by="iccs"):
    crime_name = data["iccs_d"].unique()[0]
    # log results, model outputs etc.
    log = open(f"data/{crime}_{crime_name}.log", 'w')
    log.write(f"{crime_name}\n")

    # throw outliers for the data corresponding to this crime type
    data, loginfo = throw_outliers(data, crime_name)
    log.writelines(loginfo)

    # save dataframe
    data.to_csv(f"data/{crime}_{crime_name.split(' ')[0]}_RegressionDataframe.csv")
    # print(crime_name)

    # plot the number of crimes
    # plot number of crimes in total
    plt.figure(figsize=(5, 1))
    sns.boxplot(data, x="crimes", color="gray")
    plt.legend(ncol=2)
    plt.xlabel(f"Number of {crime_labels[crime]}")
    plt.tight_layout()
    plt.savefig(f"plots/{crime}_{crime_name.split(' ')[0]}_Boxplots.svg", transparent=True)
    plt.show()
    # plot number of crimes by country
    hue_order = sorted(data["country"].unique())
    colours = [code for col, code in color_dict.items() if col in hue_order]
    plt.figure(figsize=(5, 4))
    sns.boxplot(data, x="crimes", hue="country", hue_order=hue_order, palette=colours)
    plt.legend(ncol=2)
    plt.xlabel(f"Number of {crime_labels[crime]}")
    plt.tight_layout()
    plt.savefig(f"plots/{crime}_{crime_name.split(' ')[0]}_Boxplots_Country.svg", transparent=True)
    plt.show()

    log.write(f"----------- regression for {crime_name} -----------\n")

    # equation and model
    equation = "crimes ~ (1|country) + " \
               "unemployment + (unemployment|country) + " \
               "density + (density|country) + " \
               "gdp + (gdp|country)"
    model = Lmer(equation, data=data)
    result = model.fit()

    log.write(f"{equation}\n")
    log.writelines(model.summary())

    # save model summary to table
    model.summary().to_csv(f"data/{crime}_{crime_name.split(' ')[0]}_LMER.csv")

    # save R^2 to table
    R_squared = pymers.rsquared(model.data.crimes, model.residuals)
    with open(f"data/{crime}_{crime_name.split(' ')[0]}_LMER.csv", 'a') as fd:
        fd.write(f"\nR2c:,{R_squared}\n")
    log.write(f"\nR2c:,{R_squared}\n")

    # plot regression lines
    for (var, colour) in [("unemployment", "tab:green"), ("gdp", "tab:blue"), ("density", "tab:orange")]:
        plt.figure(figsize=(5, 4))
        # noinspection PyArgumentEqualDefault
        sns.regplot(x=var, y="crimes", data=model.data, fit_reg=True, color=colour)
        x0, _ = plt.xlim()
        plt.xlim(x0, model.data[var].max())
        plt.xlabel(predictor_labels[var])
        plt.ylabel(f"Number of {crime_labels[crime]}")
        plt.tight_layout()
        plt.savefig(f"plots/{crime}_{crime_name.split(' ')[0]}_{var}_reg.svg", transparent=True)
        plt.show()
        plt.figure(figsize=(5, 4))
        sns.scatterplot(data, x=var, y="crimes", hue="country", hue_order=hue_order, palette=colours)
        x0, _ = plt.xlim()
        plt.xlim(x0, model.data[var].max())
        # plt.suptitle(crime_name)
        plt.xlabel(predictor_labels[var])
        plt.ylabel(f"Number of {crime_labels[crime]}")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f"plots/{crime}_{crime_name.split(' ')[0]}_{var}_scatter.svg", transparent=True)
        plt.show()

        # # plot regression lines by country

        x = np.linspace(start=data[var].min() * 0.99, stop=data[var].max())

        fig = plt.figure(figsize=(10, 7))
        ncols = 8
        nrows = len(data["country"].unique()) // ncols + (len(data["country"].unique()) % ncols > 0)
        for idx, country in enumerate(data["country"].unique()):
            subset = model.data[model.data["country"] == country]
            var1, var2 = list(set(predictors) - set([var]))
            x1 = subset[var1]
            x2 = subset[var2]
            y = model.fixef.loc[country, var1] * 0 + x1.mean() \
                + model.fixef.loc[country, var2] * 0 + x2.mean() \
                + model.fixef.loc[country, var] * x \
                + model.fixef.loc[country, "(Intercept)"]
            ax = plt.subplot(nrows, ncols, idx + 1)
            sns.scatterplot(subset, x=var, y="crimes", color=color_dict[country])# , ax=axs[row,idx-row])
            # plt.plot(np.array(x), np.array(y), color=color_dict[country])
            plt.tick_params(
                axis='x',   which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
            plt.tick_params(
                axis='y', which='both', left=False, right=False, labelleft=False)
            plt.xlabel(subset["country"].unique()[0])
            plt.ylabel("")
            plt.xlim(data[var].min()*0.99, data[var].max())
            plt.ylim((data["crimes"].min(), data["crimes"].max()))
        plt.tight_layout()
        plt.savefig(f"plots/{crime}_{crime_name}_{var}_ScatterIndividualCountries.svg", transparent=True)
        plt.show()

        # plt.figure(figsize=(5, 4))
        x = np.linspace(start=data[var].min() * 0.99, stop=data[var].max())
        for country in data["country"].unique():
            subset = model.data[model.data["country"] == country]
            var1, var2 = list(set(predictors) - set([var]))
            x1 = subset[var1]
            x2 = subset[var2]
            y = model.fixef.loc[country, var1] * 0 + x1.mean() \
                + model.fixef.loc[country, var2] * 0 + x2.mean() \
                + model.fixef.loc[country, var] * x \
                + model.fixef.loc[country, "(Intercept)"]
            plt.plot(np.array(x), np.array(y), color=color_dict[country])
        plt.xlabel(predictor_labels[var])
        plt.ylabel(f"Number of {crime_labels[crime]}")
        plt.legend(sorted(model.fixef.index.values.tolist()), ncol=2)
        plt.xlim(x.min(), x.max())
        if crime == "ICCS0401": # robbery
            y1 = 2500
        elif crime == "ICCS05012": # burglary
            y1 = 7200
        elif crime == "ICCS050211": # theft
            y1 = 4900
        elif crime == "ICCS0101": # homicide
            _, y1 = plt.ylim()
        plt.ylim(0, y1)
        plt.tight_layout()
        plt.savefig(f"plots/{crime}_{crime_name.split(' ')[0]}_{var}_regAdj_byCountry.svg", transparent=True)
        plt.show()
        print(crime_name)
    log.close()

    print('')

