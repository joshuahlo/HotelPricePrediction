# Hotel Price Prediction (class final project)

### source functions
source("DataAnalyticsFunctions.R") 
source("PerformanceCurves.R")

### install packages
installpkg("tree")
library(tree)
installpkg("partykit")
library(partykit)
installpkg("randomForest")
library(randomForest)
installpkg("readr")
library(readr)
installpkg('mclust')
library(mclust)
installpkg("glmnet")
library(glmnet)
installpkg('dplyr')
library(dplyr)
installpkg("ggplot2")
installpkg("GGally")
library(ggplot2)
library(GGally)

### This will turn off warning messages
options(warn=-1)
############################
set.seed(1)

###########################
# DATA PROCESSING
data <- read_csv('hotels-4.csv')

# Remove duplicates
data <- unique(data)
# variables we are going to use, selected from business understanding
data <- data %>%
  select(hotel, 
         is_canceled, 
         lead_time, 
         arrival_date_month, 
         stays_in_weekend_nights, 
         stays_in_week_nights, 
         market_segment, 
         distribution_channel, 
         is_repeated_guest, 
         previous_cancellations, 
         booking_changes, 
         deposit_type, 
         customer_type, 
         adr, 
         total_of_special_requests, 
         reservation_status_date, 
         adults, 
         children, 
         babies)

# Replace month names with number of month
data$month <- match(data$arrival_date_month, 
                    c("January", "February", "March", "April", "May", 
                      "June", "July", "August", "September", 
                      "October", "November", "December"))

data <- subset(data, select = -arrival_date_month)

# Remove NA values
data<-na.omit(data)

##########################
# END OF DATA PROCESSING

################################
mean(data$is_canceled==1)

### correlations
ggpairs(data[,c(2,3,5,18,20)])

# DATA EXPLORATION  
plot(is_canceled ~ lead_time, data=data, col=c(8,2), ylab="Cancelation Rate", xlab="Lead Time") 


# Calculate the percentage of each category
percentage_data <- as.data.frame(prop.table(table(data$is_canceled)) * 100)

# Rename columns for clarity
colnames(percentage_data) <- c("is_canceled", "percentage")

# Plot the bar graph with percentages
ggplot(percentage_data, aes(x = factor(is_canceled), y = percentage, fill = factor(is_canceled))) +
  geom_bar(stat = "identity") +
  labs(title = "Percentage of Cancellations", x = "Is Canceled", y = "Percentage") +
  theme_minimal() +
  scale_fill_discrete(name = "Is Canceled", labels = c("Not Canceled", "Canceled"))

### DATA EXPLORATION, UNSUPERVISED LEARNING
### k-means 
xdata <- model.matrix(is_canceled ~ ., data=data)[,-1]
xdata <- scale(xdata)

# Define the range of clusters to evaluate
cluster_range <- 1:30

# Compute k-means for each number of clusters
kfit <- lapply(cluster_range, function(k) kmeans(xdata, centers = k, nstart = 10))

# Use kIC function defined in DataAnalyticsFunctions.R
# Calculate AIC, BIC, and HDIC for each k
kaic <- sapply(kfit, kIC)          # AIC
kHDic <- sapply(kfit, kIC, "C")    # HDIC

# Now we plot the results
par(mar=c(1,1,1,1))
par(mai=c(1,1,1,1))

# Plot AIC
plot(kaic, xlab="k (# of clusters)", ylab="IC (Deviance + Penalty)", 
     ylim=range(c(kaic, kHDic)), 
     type="l", lwd=2, col="black")

# Vertical line where AIC is minimized
abline(v=which.min(kaic), col="black", lty=2)

# Plot HDIC
lines(kHDic, col="green", lwd=2)
# Vertical line where HDIC is minimized
abline(v=which.min(kHDic), col="green", lty=2)

# Insert labels
text(c(which.min(kaic), which.min(kHDic)), 
     c(mean(kaic), mean(kHDic)), 
     c("AIC", "HDIC"), pos=2.75)


### 23 CLUSTERS
twentythreeCenters <- kmeans(xdata,23,nstart=30)
### Sizes of clusters
twentythreeCenters$size

### variation explained with 23 clusters
1 - twentythreeCenters$tot.withinss/ twentythreeCenters$totss
### near 50%
aggregate( data$lead_time ~ twentythreeCenters$cluster, FUN = 'mean' )
aggregate( data$stays_in_week_nights ~ twentythreeCenters$cluster, FUN = 'mean' )
aggregate( data$is_repeated_guest ~ twentythreeCenters$cluster, FUN = 'mean' )
aggregate( data$is_repeated_guest ~ twentythreeCenters$cluster, FUN = 'mean' )

aggregate(data$is_canceled~twentythreeCenters$cluster, FUN='mean')

# Calculate the means for each variable by cluster
lead_time_means <- aggregate(data$lead_time ~ twentythreeCenters$cluster, FUN = 'mean')
stays_means <- aggregate(data$stays_in_week_nights ~ twentythreeCenters$cluster, FUN = 'mean')
repeated_guest_means <- aggregate(data$is_repeated_guest ~ twentythreeCenters$cluster, FUN = 'mean')
cancellation_means <- aggregate(data$is_canceled ~ twentythreeCenters$cluster, FUN = 'mean')

# Rename columns
colnames(lead_time_means) <- c("Cluster", "Mean_Lead_Time")
colnames(stays_means) <- c("Cluster", "Mean_Stays_In_Week_Nights")
colnames(repeated_guest_means) <- c("Cluster", "Mean_Repeated_Guest")
colnames(cancellation_means) <- c("Cluster", "Mean_Cancellation")

# Merge the data frames by Cluster
final_table <- lead_time_means %>%
  merge(stays_means, by = "Cluster") %>%
  merge(repeated_guest_means, by = "Cluster") %>%
  merge(cancellation_means, by = "Cluster")

# Print the final table
print(final_table)


#########################################
### Need to estimate probability of cancellation
### Compare different models 
### m.lr : logistic regression
### m.lr.l : logistic regression with interaction using lasso
### m.lr.pl : logistic regression with interaction using post lasso
### m.lr.tree : classification tree
### m.rf: random forest
###

# More data cleaning for market segment (k fold had some trouble without converting to dummy variables)
Direct <- ifelse(data$market_segment == "Direct", 1, 0)
Corporate <- ifelse(data$market_segment == "Corporate", 1, 0)
Online_TA <- ifelse(data$market_segment == "Online TA", 1, 0)
Offline_TA_TO <- ifelse(data$market_segment == "Offline TA/TO", 1, 0)
Complementary <- ifelse(data$market_segment == "Complementary", 1, 0)
Groups <- ifelse(data$market_segment == "Groups", 1, 0)
Undefined <- ifelse(data$market_segment == "Undefined", 1, 0)
Aviation <- ifelse(data$market_segment == "Aviation", 1, 0)

data <- data.frame(data, 'Direct' = Direct, 'Corporate' = Corporate, 'Online_TA' = Online_TA,
                   'Offline_TA_TO'= Offline_TA_TO, 'Complementary' = Complementary, 
                   'Groups'=Groups, 'Undefined' = Undefined, 'Aviation'= Aviation)

data <- subset(data, select = -market_segment)

dis_Direct <- ifelse(data$distribution_channel == "Direct", 1, 0)
dis_Corporate <- ifelse(data$distribution_channel == "Corporate", 1, 0)
TA_TO <- ifelse(data$distribution_channel == "TA/TO", 1, 0)
GDS <- ifelse(data$distribution_channel == "GDS", 1, 0)

data<- data.frame(data, 'Direct' = dis_Direct, 'Corporate' = dis_Corporate, 'TA/TO' = TA_TO, 'GDS'=GDS)

data <- subset(data, select = -distribution_channel)

# First need to run Lasso and cross validate lasso
Mx<- model.matrix(is_canceled ~ .^2, data=data)[,-1]
My<- data$is_canceled == 1
lasso <- glmnet(Mx,My, family="binomial")
# This will take a while to run. I ran it overnight and it worked. That's why its 5 folds and not 10
lassoCV <- cv.glmnet(Mx,My, family="binomial", nfolds=5)

# Theoretical value of lasso
num.features <- ncol(Mx)
num.n <- nrow(Mx)
num.can <- sum(My)
w <- (num.can/num.n)*(1-(num.can/num.n))
lambda.theory <- sqrt(w*log(num.features/0.05)/num.n)
lassoTheory <- glmnet(Mx,My, family="binomial",lambda = lambda.theory)
summary(lassoTheory)
support(lassoTheory$beta)

features.min <- support(lasso$beta[,which.min(lassoCV$cvm)])
length(features.min)
data.min <- data.frame(Mx[,features.min],My)


### MAE
PerformanceMeasure <- function(actual, prediction, threshold=.5) {
  #1-mean( abs( (prediction>threshold) - actual ) )  
  #R2(y=actual, pred=prediction, family="binomial")
  mean(abs(actual - prediction))
}

n <- nrow(data)
nfold <- 10
OOS <- data.frame(m.lr=rep(NA,nfold), m.lr.l=rep(NA,nfold), m.lr.pl=rep(NA,nfold), m.tree=rep(NA,nfold), m.rf=rep(NA,nfold), m.average=rep(NA,nfold)) 
#names(OOS)<- c("Logistic Regression", "Lasso on LR with Interactions", "Post Lasso on LR with Interactions", "Classification Tree", "Average of Models")
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]

# This takes about 1.5-2.5 hours to run! 
for(k in 1:nfold){  
  
  train <- which(foldid != k) # train on all but fold `k`
  
  ### Logistic regression
  m.lr <- glm(is_canceled ~ ., data = data, subset = train, family = "binomial")
  pred.lr <- predict(m.lr, newdata = data[-train,], type = "response")
  OOS$m.lr[k] <- PerformanceMeasure(actual = My[-train], pred = pred.lr)
  
  ### the Post Lasso Estimates
  m.lr.pl <- glm(My ~ ., data = data.min, subset = train, family = "binomial")
  pred.lr.pl <- predict(m.lr.pl, newdata = data.min[-train,], type = "response")
  OOS$m.lr.pl[k] <- PerformanceMeasure(actual = My[-train], prediction = pred.lr.pl)
  
  ### the Lasso estimates
  m.lr.l <- glmnet(Mx[train,], My[train], family = "binomial", lambda = lassoCV$lambda.min)
  pred.lr.l <- predict(m.lr.l, newx = Mx[-train,], type = "response")
  OOS$m.lr.l[k] <- PerformanceMeasure(actual = My[-train], prediction = pred.lr.l)
  
  ### the classification tree
  m.tree <- tree(is_canceled ~ ., data = data, subset = train)
  pred.tree <- predict(m.tree, newdata = data[-train,], type = "vector")
  OOS$m.tree[k] <- PerformanceMeasure(actual = My[-train], prediction = pred.tree)
  
  ### random forest
  # Convert is_canceled to a factor to treat it as a classification problem
  data$is_canceled <- as.factor(data$is_canceled)
  
  m.rf <- randomForest(is_canceled ~ ., data = data, subset = train, nodesize = 5, ntree = 300, mtry = 4)
  pred.rf <- predict(m.rf, newdata = data[-train,], type = "prob")[,2]  # Probability for class 1
  OOS$m.rf[k] <- PerformanceMeasure(actual = My[-train], prediction = pred.rf)
  
  ### average predictions
  pred.m.average <- rowMeans(cbind(pred.tree, pred.lr.l, pred.lr.pl, pred.lr, pred.lr))
  OOS$m.average[k] <- PerformanceMeasure(actual = My[-train], prediction = pred.m.average)
  
  print(paste("Iteration", k, "of", nfold, "completed"))
}

par(mar=c(7,5,.5,1)+0.3)
# Create the barplot with the new title "MAE"
barplot(colMeans(OOS), 
        las = 2, 
        xpd = FALSE, 
        xlab = "", 
        ylim = c(0.975 * min(colMeans(OOS)), max(colMeans(OOS))), 
        ylab = "", 
        main = "MAE")

######## ACCURACY
PerformanceMeasure <- function(actual, prediction, threshold=.5) {
  1-mean( abs( (prediction>threshold) - actual ) )  
}

n <- nrow(data)
nfold <- 10
OOS <- data.frame(m.lr=rep(NA,nfold), m.lr.l=rep(NA,nfold), m.lr.pl=rep(NA,nfold), m.tree=rep(NA,nfold), m.rf=rep(NA,nfold), m.average=rep(NA,nfold)) 
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]


# This takes about 1.5-2.5 hours to run! 
for(k in 1:nfold){  
  
  train <- which(foldid != k) # train on all but fold `k`
  
  ### Logistic regression
  m.lr <- glm(is_canceled ~ ., data = data, subset = train, family = "binomial")
  pred.lr <- predict(m.lr, newdata = data[-train,], type = "response")
  OOS$m.lr[k] <- PerformanceMeasure(actual = My[-train], pred = pred.lr)
  
  ### the Post Lasso Estimates
  m.lr.pl <- glm(My ~ ., data = data.min, subset = train, family = "binomial")
  pred.lr.pl <- predict(m.lr.pl, newdata = data.min[-train,], type = "response")
  OOS$m.lr.pl[k] <- PerformanceMeasure(actual = My[-train], prediction = pred.lr.pl)
  
  ### the Lasso estimates
  m.lr.l <- glmnet(Mx[train,], My[train], family = "binomial", lambda = lassoCV$lambda.min)
  pred.lr.l <- predict(m.lr.l, newx = Mx[-train,], type = "response")
  OOS$m.lr.l[k] <- PerformanceMeasure(actual = My[-train], prediction = pred.lr.l)
  
  ### the classification tree
  m.tree <- tree(is_canceled ~ ., data = data, subset = train)
  pred.tree <- predict(m.tree, newdata = data[-train,], type = "vector")
  OOS$m.tree[k] <- PerformanceMeasure(actual = My[-train], prediction = pred.tree)
  
  ### random forest
  # Convert is_canceled to a factor to treat it as a classification problem
  data$is_canceled <- as.factor(data$is_canceled)
  
  m.rf <- randomForest(is_canceled ~ ., data = data, subset = train, nodesize = 5, ntree = 300, mtry = 4)
  pred.rf <- predict(m.rf, newdata = data[-train,], type = "prob")[,2]  # Probability for class 1
  OOS$m.rf[k] <- PerformanceMeasure(actual = My[-train], prediction = pred.rf)
  
  ### average predictions
  pred.m.average <- rowMeans(cbind(pred.tree, pred.lr.l, pred.lr.pl, pred.lr, pred.lr))
  OOS$m.average[k] <- PerformanceMeasure(actual = My[-train], prediction = pred.m.average)
  
  print(paste("Iteration", k, "of", nfold, "completed"))
}

par(mar=c(7,5,.5,1)+0.3)
barplot(colMeans(OOS), 
        las = 2, 
        xpd = FALSE, 
        xlab = "", 
        ylim = c(0.975 * min(colMeans(OOS)), max(colMeans(OOS))), 
        ylab = "", 
        main = "OOS Accuracy")  # Add title here
### Post Lasso very similar to Average of models.
### Post Lasso better in OOS MAD in |Y-Prob|
### Average better in OOS Accuracy and OOS R2
###
data$is_canceled <- as.factor(data$is_canceled)
train <- which(foldid!=1)
m.rf <- randomForest(is_canceled~., data=data, subset=train, nodesize=5, ntree = 300, mtry = 4)
pred.rf <- predict(m.rf,type = "prob")[,2]
hist(pred.rf, breaks = 40,main="Prediction for Random Forest")
sum(pred.rf==0)


train <- which(foldid!=1)
### Logistic regression
m.lr <-glm(is_canceled~., data=data, subset=train,family="binomial")
pred.lr <- predict(m.lr, newdata=data[-train,], type="response")
par(mar=c(1.5,1.5,1.5,1.5))
par(mai=c(1.5,1.5,1.5,1.5))
hist(pred.lr, breaks = 40,main="Prediction for Logistic Regression")
### the Post Lasso Estimates
m.lr.pl <- glm(My~., data=data.min, subset=train, family="binomial")
pred.lr.pl <- predict(m.lr.pl, newdata=data.min[-train,], type="response")
par(mar=c(1.5,1.5,1.5,1.5))
par(mai=c(1.5,1.5,1.5,1.5))
hist(pred.lr.pl, breaks = 40, main="Predictions for Post Lasso")

### FEATURE IMPORTANCE
# Extract importance
importance(m.rf)

# Plotting feature importance
varImpPlot(m.rf)


# Convert reservation_status_date to Date type if it's not already
data$reservation_status_date <- as.Date(data$reservation_status_date)

# Group data by reservation_status_date and count cancellations and non-cancellations
data_summary <- data %>%
  group_by(reservation_status_date) %>%
  summarise(canceled = sum(is_canceled == 1), 
            not_canceled = sum(is_canceled == 0))

# Identify the top 5 peak points (max cancellations per day)
top_5_peaks <- data_summary %>%
  arrange(desc(canceled)) %>%
  head(5)

# Create a line plot with ggplot and label top 5 peaks with their months
ggplot(data_summary, aes(x = reservation_status_date)) +
  geom_line(aes(y = canceled, color = "Canceled"), size = 1) +
  geom_line(aes(y = not_canceled, color = "Not Canceled"), size = 1) +
  geom_text(data = top_5_peaks, aes(x = reservation_status_date, y = canceled, 
                                    label = format(reservation_status_date, "%B")), 
            vjust = -1, color = "red", size = 3) +  # Labeling the peaks with month names
  labs(title = "Reservations: Cancellations vs Non-Cancellations Over Time",
       x = "Date",
       y = "Count of Reservations",
       color = "Reservation Status") +
  theme_minimal()


# By hotel type
data$reservation_status_date <- as.Date(data$reservation_status_date)

# Split the data by hotel type
resort_data <- data %>% filter(hotel == "Resort Hotel")
city_data <- data %>% filter(hotel == "City Hotel")

# Function to create a plot for each hotel type
plot_hotel_cancellations <- function(hotel_data, hotel_name) {
  # Group data by reservation_status_date and count cancellations and non-cancellations
  data_summary <- hotel_data %>% 
    group_by(reservation_status_date) %>% 
    summarise(canceled = sum(is_canceled == 1), 
              not_canceled = sum(is_canceled == 0))
  
  # Identify the single highest peak (max cancellations per day)
  highest_peak <- data_summary %>% 
    slice_max(canceled, n = 1)
  
  # Create a line plot label highest peak with its month
  ggplot(data_summary, aes(x = reservation_status_date)) +
    geom_line(aes(y = canceled, color = "Canceled"), size = 1) +
    geom_line(aes(y = not_canceled, color = "Not Canceled"), size = 1) +
    geom_text(data = highest_peak, aes(x = reservation_status_date, y = canceled, 
                                       label = format(reservation_status_date, "%B")), 
              vjust = -1, color = "red", size = 3) +  # Labeling the peak with the month name
    labs(title = paste("Cancellations vs Non-Cancellations -", hotel_name),
         x = "Date",
         y = "Count of Reservations",
         color = "Reservation Status") +
    theme_minimal()
}

# Create two separate plots for Resort Hotel and City Hotel
plot_resort <- plot_hotel_cancellations(resort_data, "Resort Hotel")
plot_city <- plot_hotel_cancellations(city_data, "City Hotel")

plot_resort
plot_city


# Convert reservation_status_date to Date type if it's not already
data$reservation_status_date <- as.Date(data$reservation_status_date)

# Filter the data for the month of October
october_data <- data_summary %>%
  filter(format(reservation_status_date, "%B") == "October")

# Find the exact date in October with the highest number of cancellations
october_peak <- october_data %>%
  filter(canceled == max(canceled))

# Display the exact date with the highest cancellations in October
october_peak


### We use the Post Lasso Estimates for simplicity
### and we run the method in the whole sample

m.lr.pl <- glm(My~., data=data.min, family="binomial")
summary(m.lr.pl)$coef[,1]
pred.lr.pl <- predict(m.lr.pl, newdata=data.min, type="response")
### To create the confusion matrix, we set the threshold to 0.5
### If threshold >0.5 then it's regarded as cancelled, if <0.5 it's regarded as not cancelled
PL.performance <- FPR_TPR(pred.lr.pl>=0.5 , My)
PL.performance
confusion.matrix <- c( sum(pred.lr.pl>=0.5) *PL.performance$TP,  sum(pred.lr.pl>=0.5) * PL.performance$FP,  sum(pred.lr.pl<0.5) * (1-PL.performance$TP),  sum(pred.lr.pl<0.5) * (1-PL.performance$FP) )
confusion.matrix <- c( sum( (pred.lr.pl>=0.5) * My ),  sum( (pred.lr.pl>=0.5) * !My ) , sum( (pred.lr.pl<0.5) * My ),  sum( (pred.lr.pl<0.5) * !My))

# Calculations for cost benefit matrix
# Hotels make a profit of on average $20 per person per night

# Calculation for average nights stayed
total_stays <- data$stays_in_weekend_nights + data$stays_in_week_nights
average_stays <- mean(total_stays)
print(average_stays) # 3.630687

# Calculation for average number of people (adults + children) staying
total_people <- data$adults + data$children
average_people <- mean(total_people)
print(average_people) # 2.014418

# TOTAL REVENUE PER RESERVATION THAT IS NOT CANCELLED
total_revenue <- 20*average_stays*average_people # 146.2744

# Assume $25 for how much it costs for our vouchers/free cab/free breakfast per reservation
total_profit <- total_revenue-25 # 121.2744

cost.benefit.matrix <- c( 0, 0 , -25 , total_profit )

par(mar=c(6,6,2,6))


### Expected profit
t(cost.benefit.matrix) %*% confusion.matrix

# scale by time and hotel
sum(data$hotel == "Resort Hotel") # 33968
sum(data$hotel == "City Hotel") # 53424

(t(cost.benefit.matrix) %*% confusion.matrix)/87392
##### You make a profit of $78.17 for every reservation #####

### Baseline of majority rule (when we predict that reservation cancels)
cost.benefit.matrix %*% c( 0, 0, sum(My), sum(!My) )

### for post Lasso predictions
### we will use functions in PerformanceCurves.R files (load earlier)
par(mar=c(5,5,3,5))
profit <- profitcurve(p=pred.lr.pl,y=My,cost.benefit.m=cost.benefit.matrix)

profictcurveOrder(score=pred.lr.pl,y=My,cost.benefit.m=cost.benefit.matrix)
profitcurveAll(p=pred.lr.pl,y=My,cost.benefit.m=cost.benefit.matrix)

# ROC curve and cumulative curve for model performance
roccurve <-  roc(p=pred.lr.pl, y=My, bty="n")
cumulative <- cumulativecurve(p=pred.lr.pl,y=My)

