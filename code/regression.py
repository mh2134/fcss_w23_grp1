"""Linear Mixed Effects Model"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from pymer4.models import Lmer
import pymer4.stats as pymers
import numpy as np

# axis limits for the final plots
from fcss_w23_grp1.code.utils import *

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
color_dict, palette = define_persistent_colours(df)

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

    # plot the number of crimes (boxplots)
    make_boxplots(data, crime, crime_name, color_dict)

    # equation and model
    log.write(f"----------- regression for {crime_name} -----------\n")
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
    plot_regression_lines(data, model, crime, crime_name, color_dict)

    log.close()

    return;

