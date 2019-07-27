# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from patsy import dmatrices, dmatrix

# %%
# help from here https://stackoverflow.com/questions/12604909/pandas-how-to-change-all-the-values-of-a-column
# https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python

def convertDate(dateStr):
    newDate = datetime.strptime(dateStr, '%Y-%m-%d')
    return newDate


# %%
avocados = pd.read_csv("avocado.csv")
avocados['Date'] = avocados['Date'].apply(convertDate)

# %%
plt.plot_date(avocados.Date, avocados.AveragePrice)
# %%
x1=avocados['AveragePrice'][(avocados.region == "West") & (avocados.type == "organic")]
y1=avocados['Total Volume'][(avocados.region == "West") & (avocados.type == "organic")]
plt.plot(x1,y1, "ro")
plt.plot(np.unique(x1), np.poly1d(np.polyfit(x1, y1, 1))(np.unique(x1)))

x2=avocados['AveragePrice'][(avocados.region == "West") & (avocados.type == "conventional")]
y2=avocados['Total Volume'][(avocados.region == "West") & (avocados.type == "conventional")]
plt.plot(x2,y2, "bo")
plt.plot(np.unique(x2), np.poly1d(np.polyfit(x2, y2, 1))(np.unique(x2)))

plt.show()

# %%
x1=avocados['AveragePrice'][(avocados.region == "California") & (avocados.type == "organic")].transform('log')
y1=avocados['Total Volume'][(avocados.region == "California") & (avocados.type == "organic")].transform('log')
plt.plot(x1, y1, "ro")
plt.plot(np.unique(x1), np.poly1d(np.polyfit(x1, y1, 1))(np.unique(x1)))

x2=avocados['AveragePrice'][(avocados.region == "California") & (avocados.type == "conventional")].transform('log')
y2=avocados['Total Volume'][(avocados.region == "California") & (avocados.type == "conventional")].transform('log')
plt.plot(x2,y2, "bo")
plt.plot(np.unique(x2), np.poly1d(np.polyfit(x2, y2, 1))(np.unique(x2)))

plt.show()

# %%
avocados['month'] = pd.DatetimeIndex(avocados['Date']).month
x1=avocados['month'][(avocados.region == "California") & (avocados.type == "organic")]
y1=avocados['Total Volume'][(avocados.region == "California") & (avocados.type == "organic")]
plt.plot(x1,y1, "bo")
plt.plot(np.unique(x1), np.poly1d(np.polyfit(x1, y1, 1))(np.unique(x1)))

x2=avocados['month'][(avocados.region == "California") & (avocados.type == "conventional")]
y2=avocados['Total Volume'][(avocados.region == "California") & (avocados.type == "conventional")]
plt.plot(x2,y2, "ro")
plt.plot(np.unique(x2), np.poly1d(np.polyfit(x2, y2, 1))(np.unique(x2)))

plt.show()

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
