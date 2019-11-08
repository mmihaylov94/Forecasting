## @author Mihail Mihaylov
## @date 03/03/2019

library(quantmod)
library(neuralnet)
library(forecast)
options("scipen"=10, "digits"=4)


#Data source: http://faculty.chicagobooth.edu/ruey.tsay/teaching/fts/

#Get the monthly log stock returns for Disney
disney <- scan("http://faculty.chicagobooth.edu/ruey.tsay/teaching/fts/m-dis6299.dat")

#Put it into a time series specifying the frequency (12 means monthly)
disneyts <- ts(disney,frequency=12)

#plot the time series
plot.ts(disneyts)

#Using a smoothing method to estimate the trend component
disneytsSMA <- SMA(disneyts, 12)
disneytsSMA <- na.omit(disneytsSMA)

#Plot the original data set versus the smoothed values
ts.plot(disneyts, disneytsSMA, gpars = list(col = c("black", "red")))
legend("bottomright", legend = c("Disney Original", "Disney Smoothed"), col = 1:2, lty=1)

#Create a lagged matrix
disneyLaggedTS <- cbind(disneytsSMA, disneyts1=Lag(disneytsSMA,k=1), disneyts2=Lag(disneytsSMA,k=2), disneyts3=Lag(disneytsSMA,k=3)
                        , disneyts4=Lag(disneytsSMA,k=4), disneyts5=Lag(disneytsSMA,k=5), disneyts6=Lag(disneytsSMA,k=6)
                        , disneyts7=Lag(disneytsSMA,k=7), disneyts8=Lag(disneytsSMA,k=8), disneyts9=Lag(disneytsSMA,k=9)
                        , disneyts10=Lag(disneytsSMA,k=10))

colnames(disneyLaggedTS) <- c("Input1", "Input2", "Input3", "Input4", "Input5", "Input6", "Input7", "Input8", "Input9", "Input10", "Output")

#Remove rows with null values
disneyLaggedTS <- na.omit(disneyLaggedTS)

#Transfer the matrix back to a time series
disneyLaggedTS <- ts(disneyLaggedTS)

#Disney lagged time series time slice set
#Perform an 80/20 Train/test set split
index <- ceiling(0.8*nrow(disneyLaggedTS))

#Disney lagged time series time slice training and testing set
disneyLTStrain <- disneyLaggedTS[1:index,]
disneyLTStest <- disneyLaggedTS[(index+1):nrow(disneyLaggedTS),1:(ncol(disneyLaggedTS)-1)]

disneyLTSExpResults <- disneyLTStrain[,ncol(disneyLTStrain)]
disneyLTSTestExpResults <- disneyLaggedTS[(index+1):nrow(disneyLaggedTS),ncol(disneyLaggedTS)]

#Run the neural net
disneyForecastNN <- neuralnet(Output ~ Input1 + Input2 + Input3 + 
                                Input4 + Input5 + Input6 + Input7 + 
                                Input8 + Input9 + Input10, disneyLTStrain,
                              hidden = c(15,15,15),algorithm = "rprop+",stepmax=1e+06, threshold = 0.1)

#Plot the neural net
plot(disneyForecastNN)

#Display the result matrix containing the MSE
disneyForecastNN$result.matrix[1:3,]

#Display the results from the training set
cleanOutputTrain <- cbind(disneyLTSExpResults, as.data.frame(disneyForecastNN$net.result))
colnames(cleanOutputTrain) <- c("Expected Output", "Neural Net Output")
print(cleanOutputTrain)


###Testing
#Test the neural network on the testing set
disneyLTStest.results <- compute(disneyForecastNN,disneyLTStest) #Run them through the neural network

#See the results
cleanOutputTest <- cbind(disneyLTSTestExpResults, as.data.frame(disneyLTStest.results$net.result))
colnames(cleanOutputTest) <- c("Expected Output", "Neural Net Output")

#MSE for the test set
custom <- function(x,y){sqrt((y-x)^2)}
mean(custom(disneyLTSTestExpResults,disneyLTStest.results$net.result))

### Get the predictions for the test set and plot it against the actual values from the test set
ts.plot(ts(cleanOutputTest[,1]),ts(cleanOutputTest[,2]), gpars = list(col = c("red", "blue")))
legend("bottomleft", legend = c("NN Predictions", "Disney Smoothed"), col = 1:2, lty=1)

###Holt-Winters
##Expected values from the Holt Winters
expected <- ts(c(disneyLTStrain[,11]), frequency=12)
hwExpected <- ts(forecast((HoltWinters(expected,gamma=FALSE)),h=nrow(disneyLTStest))$mean)


### Baseline mean performance
bMean <- function(x){return(mean(x))}
bMeanExpected <- ts(apply(disneyLTStest, 1, bMean))

expectedResults <- cbind(cleanOutputTest[,1], cleanOutputTest[,2], bMeanExpected, hwExpected)
colnames(expectedResults) <- c("Disney Smoothed", "NN Predicted", "Mean Predicted", "HW Predicted")
expectedResults

ts.plot(ts(cleanOutputTest[,1]),ts(cleanOutputTest[,2]), bMeanExpected, hwExpected, gpars = list(col = c("blue", "red", "green", "yellow")))
legend("bottomleft", legend = c("Disney Smoothed", "NN Predicted", "Baseline Mean Predicted", "Holt-Winters Predicted"), col = c("blue", "red", "green", "yellow"), lty=1)
