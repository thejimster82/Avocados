# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from patsy import dmatrices, dmatrix

# %%
# help from here https://stackoverflow.com/questions/12604909/pandas-how-to-change-all-the-values-of-a-column


def convertDate(dateStr):
    newDate = datetime.strptime(dateStr, '%Y-%m-%d')
    return newDate


# %%
avocados = pd.read_csv("avocado.csv")
avocados['Date'] = avocados['Date'].apply(convertDate)

# %%
plt.plot_date(avocados.Date, avocados.AveragePrice)
# %%
plt.plot(avocados['AveragePrice'][(avocados.region == "West") & (avocados.type == "organic")],
         avocados['Total Volume'][(avocados.region == "West") & (avocados.type == "organic")], "ro")
plt.plot(avocados['AveragePrice'][(avocados.region == "West") & (avocados.type == "conventional")],
         avocados['Total Volume'][(avocados.region == "West") & (avocados.type == "conventional")], "bo")
plt.show()

# %%
plt.plot(avocados['AveragePrice'][(avocados.region == "California") & (avocados.type == "organic")].transform('log'),
         avocados['Total Volume'][(avocados.region == "California") & (avocados.type == "organic")].transform('log'), "ro")
plt.plot(avocados['AveragePrice'][(avocados.region == "California") & (avocados.type == "conventional")].transform('log'),
         avocados['Total Volume'][(avocados.region == "California") & (avocados.type == "conventional")].transform('log'), "bo")
plt.show()

# %%
avocados['month'] = pd.DatetimeIndex(avocados['Date']).month
plt.plot(avocados['month'][(avocados.region == "California") & (avocados.type == "organic")],
         avocados['Total Volume'][(avocados.region == "California") & (avocados.type == "organic")], "bo")
plt.plot(avocados['month'][(avocados.region == "California") & (avocados.type == "conventional")],
         avocados['Total Volume'][(avocados.region == "California") & (avocados.type == "conventional")], "ro")

# %%
result, predictors = dmatrices(
    "Q('Total Volume') ~ AveragePrice + region + type + Date + AveragePrice:type", avocados)
avocMod1 = sm.OLS(result, predictors
                  ).fit()
avocMod1.summary()
# %%
result, predictors = dmatrices(
    "np.log(Q('Total Volume')) ~ AveragePrice + region + type + Date + AveragePrice:type + AveragePrice:region", avocados)
avocMod2 = sm.OLS(result, predictors
                  ).fit()
avocMod2.summary()

# %%
result, predictors = dmatrices(
    "np.log(Q('Total Volume')) ~ np.log(AveragePrice) + region + type + Date + np.log(AveragePrice):type + np.log(AveragePrice):region", avocados)
avocMod3 = sm.OLS(result, predictors
                  ).fit()
avocMod3.summary()
