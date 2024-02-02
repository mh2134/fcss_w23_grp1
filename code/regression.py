"""Linear Mixed Effects Model"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from pymer4.models import Lmer
import pymer4.stats as pymers
import numpy as np
from fcss_w23_grp1.code.utils import *

# read in preprocessed data
df = pd.read_csv("data/normalized_data.csv")
# rename colums
df.rename(columns={"TIME_PERIOD": "year", "crime/population": "crimes", "unemp per Tsd.": "unemployment"}, inplace=True)
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


