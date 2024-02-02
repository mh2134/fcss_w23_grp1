library(lmerTest) 
library(MuMIn)
library(tidyr)
setwd(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))


for (crime in c("ICCS0401_Robbery_RegressionDataframe.csv", 
                "ICCS05012_Burglary_RegressionDataframe.csv",
                "ICCS0101_Intentional_RegressionDataframe.csv",
                "ICCS050211_Theft_RegressionDataframe.csv"))
{
  cname <- substr(crime, 1, 16)
  cname <- paste(cname,".txt", sep="")
  data <- read.csv(paste("data",crime, sep="/"))
  data$country <- as.factor(data$country)
  sink(paste("data", cname, sep="/"))
  mdl <- lmer(crimes ~ gdp + (gdp|country) +
                unemployment + (1|country) + (unemployment|country) +
                density + (density|country), # (gdp:density) + (gdp:unemployment),
              data=data) 
  
  print(summary(mdl))
  R2 <- r.squaredGLMM(mdl) 
  print(R2)
  print(R2[2] - R2[1])
  
  ranef(mdl) # random effect
  coef(mdl) # schon berechnete
  sink()
  
}


data <- read.csv("data/ICCS0401_Robbery_dataframe.csv")
data$country <- as.factor(data$country)
sink(paste("data", "ICCS0401_Robbery.txt", sep="/"))
mdl <- lmer(crimes ~ (1|country) + gdp + (gdp|country) +
              unemployment + (unemployment|country) +
              density + (density|country), # (gdp:density) + (gdp:unemployment),
            data=data) 

print(summary(mdl))
R2 <- r.squaredGLMM(mdl) 
print(R2)
print(R2[2] - R2[1])


mdl <- lmer(crimes ~ (1|country) + density + (density|country), 
            data=data) 
print(summary(mdl))
R2 <- r.squaredGLMM(mdl) 
print(R2)
print(R2[2] - R2[1])

mdl <- lm(crimes ~ density + gdp + unemployment,
          data=data) 

print(summary(mdl))
R2 <- r.squaredGLMM(mdl) 
print(R2)
print(R2[2] - R2[1])

mdl <- lm(crimes ~ density,
          data=data) 

print(summary(mdl))
R2 <- r.squaredGLMM(mdl) 
print(R2)
print(R2[2] - R2[1])
sink()


# r squared M: nur fixed effects (varianz durch die predictors)
# r squared C: fixed + random effects 
# spielt random slope rolle? 1 modell mit, eins ohne, dann r^2 vgl (beim r^2 C 
# sieht man den unterschied)