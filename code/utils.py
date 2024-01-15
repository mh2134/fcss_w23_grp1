"""
Functions.
"""

def data_from_year(df, year):
    """Get data from specific year."""
    return df[df["TIME_PERIOD"] == year];


def split_column(df, col):
    """Split columns of type CODE:LABEL into more readable separate columns."""
    df[f"{col}_code"] = df[col].apply(lambda x: x.split(':')[0])
    df[f"{col}_name"] = df[col].apply(lambda x: x.split(':')[1])
    return df;