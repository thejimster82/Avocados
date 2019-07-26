install.packages("lubridate")
library(dplyr)
library(ggplot2)
library(car)
library(lubridate)
avocados <- read.csv("avocado.csv")
avocados$Date <- as.Date(avocados$Date)
ggplot(avocados,aes(x=Date,y=AveragePrice,color=type)) + geom_point() + geom_smooth() + facet_wrap(avocados$type)

ggplot(filter(avocados,region=='West'),aes(x=AveragePrice,y=Total.Volume,color=type)) + geom_point() 
organic = filter(avocados,type=="organic")
conv = filter(avocados,type=='conventional')

ggplot(filter(conv,region!='TotalUS'),aes(x=AveragePrice,y=Total.Volume,color=type)) + geom_point() + facet_wrap(filter(conv,region!='TotalUS')$region)

#Region is def really important
#Same with type
avo.cali <- filter(avocados,region=="California") 

ggplot(avo.cali,aes(x=log(AveragePrice),y=log(Total.Volume),color=type)) + geom_point() + geom_smooth(method='lm')

cor(avo.cali$Total.Bags,avo.cali$Total.Bags)
avos_mod = lm(data=avocados, Total.Volume ~ AveragePrice + region + type + Date + AveragePrice:type + AveragePrice:region)
avos_mod2 = lm(data=avocados, log(Total.Volume) ~ AveragePrice + region + type + Date + AveragePrice:type + AveragePrice:region)
avos_mod3 = lm(data=avocados, log(Total.Volume) ~ log(AveragePrice) + region + type + Date + log(AveragePrice):type + log(AveragePrice):region)
summary(avos_mod)
summary(avos_mod2)
summary(avos_mod3)

vif(avos_mod)
vif(avos_mod2)
vif(avos_mod3)

avo.cali$Month = month(as.POSIXct(avo.cali$Date,format="%Y-%m-%d"))

ggplot(avo.cali,aes(x=Month,y=Total.Volume,color=type)) + geom_point()
 

