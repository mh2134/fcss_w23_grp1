"""
Functions.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import numpy as np

# short names for the crime types
crime_labels = {
    "ICCS0101": "Homicides",
    "ICCS0401": "Robberies",
    "ICCS05012": "Burglaries",
    "ICCS050211": "Thefts"
}

# axis labels of the predictors
predictor_labels = {
    "unemployment": "Unemployment Rate (per 1000 citizens)",
    "density": "Urban Density (inhabitants per 10 km²)",
    "gdp": "GDP (Purchasing Power Standard, % EU average)"
}

def data_from_year(df, year):
    """Get data from specific year."""
    return df[df["TIME_PERIOD"] == year];


def split_column(df, col):
    """Split columns of type CODE:LABEL into more readable separate columns."""
    df[f"{col}_code"] = df[col].apply(lambda x: x.split(':')[0])
    df[f"{col}_name"] = df[col].apply(lambda x: x.split(':')[1])
    return df;


def define_persistent_colours(df):
    pltte = sns.color_palette(cc.glasbey_bw_minc_20, n_colors=len(df.country.unique()))
    colDict = {}
    for index, country in enumerate(df.country.unique()):
        colDict.update({country: pltte[index]})
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[colDict[c] for c in df.country.unique()])
    return colDict, pltte


def throw_outliers(df, cName, thesevars=["crimes", "gdp", "density", "unemployment"]):
    """Throw outliers from the predictors, the predicted variable and countries with too few data points."""
    loginfo = []

    tmp = df.copy(deep=True)
    for var in thesevars:
        len_before = len(df)
        df = df[df[var] <= (tmp[var].median() + 3 * tmp[var].std())]
        # print(f"Threw {len_before-len(data)} rows with outliers in '{var}'\n")
        loginfo.append(f"Threw {len_before - len(df)} rows with outliers in '{var}'\n")
    del tmp

    # throw countries that have too few data points (now)
    countries = []
    for country, rows in df.groupby(by="country"):
        if len(rows["year"].unique()) <= 3:
            countries.append(country)
    loginfo.append(f"throwing countries {countries} with too few datapoints (<=3)\n")
    df = df[~df["country"].isin(countries)]

    return df, loginfo;


def make_boxplots(df, crime, cName, colDict):
    """Make boxplots (1) for all the data, (2) for the individual countries."""
    # (1) plot number of crimes in total
    plt.figure(figsize=(5, 1))
    sns.boxplot(df, x="crimes", color="gray")
    plt.legend(ncol=2)
    plt.xlabel(f"Number of {crime_labels[crime]}")
    plt.tight_layout()
    plt.savefig(f"plots/{crime}_{cName.split(' ')[0]}_Boxplots.svg", transparent=True)
    plt.show()

    # (2) plot number of crimes by country
    hue_order = sorted(df["country"].unique())
    colours = [code for col, code in colDict.items() if col in hue_order]
    plt.figure(figsize=(5, 4))
    sns.boxplot(df, x="crimes", hue="country", hue_order=hue_order, palette=colours)
    plt.legend(ncol=2)
    plt.xlabel(f"Number of {crime_labels[crime]}")
    plt.tight_layout()
    plt.savefig(f"plots/{crime}_{cName.split(' ')[0]}_Boxplots_Country.svg", transparent=True)
    plt.show()

    return;


def plot_regression_lines(df, mdl, crime, cName, colDict):
    """Plot regression lines for the fitted model."""
    hue_order = sorted(df["country"].unique())
    colours = [code for col, code in colDict.items() if col in hue_order]
    predictors = ["gdp", "unemployment", "density"]
    for (var, colour) in [("unemployment", "tab:green"), ("gdp", "tab:blue"), ("density", "tab:orange")]:
        # regression line for the data in total for the current predictor
        plt.figure(figsize=(5, 4))
        # noinspection PyArgumentEqualDefault
        sns.regplot(x=var, y="crimes", data=mdl.data, fit_reg=True, color=colour)
        x0, _ = plt.xlim()
        plt.xlim(x0, mdl.data[var].max())
        plt.xlabel(predictor_labels[var])
        plt.ylabel(f"Number of {crime_labels[crime]}")
        plt.tight_layout()
        plt.savefig(f"plots/{crime}_{cName.split(' ')[0]}_{var}_reg.svg", transparent=True)
        plt.show()

        # coloured scatter plot for the individual countries
        plt.figure(figsize=(5, 4))
        sns.scatterplot(df, x=var, y="crimes", hue="country", hue_order=hue_order, palette=colours)
        x0, _ = plt.xlim()
        plt.xlim(x0, mdl.data[var].max())
        plt.xlabel(predictor_labels[var])
        plt.ylabel(f"Number of {crime_labels[crime]}")
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(f"plots/{crime}_{cName.split(' ')[0]}_{var}_scatter.svg", transparent=True)
        plt.show()

        # individual scatter plot for each country
        x = np.linspace(start=df[var].min() * 0.99, stop=df[var].max())
        plt.figure(figsize=(10, 7))
        ncols = 8
        nrows = len(df["country"].unique()) // ncols + (len(df["country"].unique()) % ncols > 0)
        for idx, country in enumerate(df["country"].unique()):
            subset = mdl.data[mdl.data["country"] == country]
            plt.subplot(nrows, ncols, idx + 1)
            sns.scatterplot(subset, x=var, y="crimes", color=colDict[country])
            plt.tick_params(
                axis='x',   which='both', bottom=False, top=False, labelbottom=False, labeltop=False)
            plt.tick_params(
                axis='y', which='both', left=False, right=False, labelleft=False)
            plt.xlabel(subset["country"].unique()[0])
            plt.ylabel("")
            plt.suptitle(f"Number of {crime_labels[crime]} by {predictor_labels[var]}")
            plt.xlim(df[var].min()*0.99, df[var].max())
            plt.ylim((df["crimes"].min(), df["crimes"].max()))
        plt.tight_layout()
        plt.savefig(f"plots/{crime}_{cName}_{var}_ScatterIndividualCountries.svg", transparent=True)
        plt.show()

        # regression lines for the individual countries
        x = np.linspace(start=df[var].min() * 0.99, stop=df[var].max())
        for country in df["country"].unique():
            subset = mdl.data[mdl.data["country"] == country]
            var1, var2 = list(set(predictors) - set([var]))
            x1 = subset[var1]
            x2 = subset[var2]
            y = mdl.fixef.loc[country, var1] * 0 + x1.mean() \
                + mdl.fixef.loc[country, var2] * 0 + x2.mean() \
                + mdl.fixef.loc[country, var] * x \
                + mdl.fixef.loc[country, "(Intercept)"]
            plt.plot(np.array(x), np.array(y), color=colDict[country])
        plt.xlabel(predictor_labels[var])
        plt.ylabel(f"Number of {crime_labels[crime]}")
        plt.legend(sorted(mdl.fixef.index.values.tolist()), ncol=2)
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
        plt.savefig(f"plots/{crime}_{cName.split(' ')[0]}_{var}_regAdj_byCountry.svg", transparent=True)
        plt.show()

    return;