library(caret)
library(ROCR) 
library(tidyverse)
library(Amelia)
library(mlbench)
library(ggplot2)
library(reshape2)
library(e1071)
library(car)        #<-- used to get Prestige dataset; and 'symbox' function
library(EnvStats)   #<-- used to get "boxcox" function
library(ggbiplot)
library(MASS)
library(devtools)
library(ggord)
library(mice)
library(VIM)
library(Amelia)
library(randomForest)
library(rpart.plot)
library(gbm)
library(GGally)
library(pROC)

# Function in R, using precision, recall and F statistics
# For multiclass, average across all classes 
model.accuracy <- function(predicted.class, actual.class){
  result.tbl <- as.data.frame(table(predicted.class,actual.class ) ) 
  result.tbl$Var1 <- as.character(result.tbl$predicted.class)
  result.tbl$Var2 <- as.character(result.tbl$actual.class)
  colnames(result.tbl)[1:2] <- c("Pred","Act")
  cntr <- 0  
  
  for (pred.class in unique(result.tbl$Pred) ){
    cntr <- cntr+ 1
    tp <- sum(result.tbl[result.tbl$Pred==pred.class & result.tbl$Act==pred.class, "Freq"])
    tp.fp <- sum(result.tbl[result.tbl$Pred == pred.class , "Freq" ])
    tp.fn <- sum(result.tbl[result.tbl$Act == pred.class , "Freq" ])
    presi <- tp/tp.fp 
    rec <- tp/tp.fn
    F.score <- 2*presi*rec/(presi+rec)
    if (cntr == 1 ) F.score.row <- cbind(pred.class, presi,rec,F.score)
    if (cntr > 1 ) F.score.row <- rbind(F.score.row,cbind(pred.class,presi,rec,F.score))
  }
  F.score.row <- as.data.frame(F.score.row) 
  return(F.score.row)
}
#read the data
read_train <- read.csv('../../Project/train.csv')
forest_train <- as.tibble(read_train)
forest_train <- forest_train[,-1]
forest_train[,11:55] <- lapply(forest_train[,11:55], as.factor)
#Missing Values
colSums(is.na(forest_train))
which(is.na(forest_train), arr.ind=TRUE)
#forest_train <- forest_train[-c(6501,10740,14893,2242,12250,12411),]
#Renaming the variables
forest_train <- dplyr::rename(forest_train,  HD_hydro=Horizontal_Distance_To_Hydrology , VD_hydro=Vertical_Distance_To_Hydrology , HD_road=Horizontal_Distance_To_Roadways,
                              HD_fire=Horizontal_Distance_To_Fire_Points ,  WA_1 =Wilderness_Area1 ,  WA_2 =Wilderness_Area2 ,  WA_3 =Wilderness_Area3 , WA_4 =Wilderness_Area4 )

####################
#Visualizations
#plot-1
#check no of instances belonging to each output class
percentage <- prop.table(table(forest_train$Cover_Type)) * 100
cbind(freq=table(forest_train$Cover_Type), percentage=percentage)

#Plot 2
#plot density of numeric features by covertype
label <- colnames(forest_train[,1:54], do.NULL = TRUE, prefix = "col")
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=forest_train[,1:10],y=forest_train$Cover_Type,plot="density",scales=scales, labels=label)

#Plot 3
#Correlation plot
correlationplot<-ggcorr(forest_train[,1:10], label = TRUE, label_size = 3, label_round = 2, label_alpha = TRUE, hjust = 00.75, size = 3,layout.exp = 1)
correlationplot

#plot-4
#Scatter plot between two correlated variables
ggplot(forest_train, aes(x=Aspect)) + 
  geom_point(aes(y=Hillshade_9am, color="Hillshade_9am"), alpha=.1) +
  geom_point(aes(y=Hillshade_3pm, color="Hillshade_3pm"), alpha=.1)

# Plot 5
# Boxplot of Elevation by Cover Type
ggplot(forest_train, aes(x=Cover_Type, y=Elevation, fill = as.factor(Cover_Type))) +
  geom_boxplot() +
  labs(title="Elevation by Cover Type", x="Cover Type", y="Elevation") +
  scale_fill_discrete(name = "Cover Type")

#Plot 6
# Density plot of elevation by Cover Type
ggplot(forest_train, aes(Elevation, fill=as.factor(Cover_Type))) +
  geom_density(alpha=0.4) +
  labs(title="Elevation Density by Cover Type", x="", y="") +
  scale_fill_discrete(name="Cover Type")

#PCA
numerical_var <- c('Elevation','Aspect','Slope','HD_hydro','VD_hydro',
                   'HD_road','Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
                   'HD_fire')
forest_num_var <- dplyr :: select(forest_train, one_of(numerical_var))
#Principal Component analysis
pc <- prcomp(forest_num_var, scale = T)
summary(pc) # To get summary of principal components.
plot(pc) # plots pc
plot(prcomp(forest_num_var, scale. = TRUE), type = "l" , main = "Variance explained by each component")
#First two columns itself  explains 51% of the variance in the whole matrix.
ggbiplot(pc, obs.scale = 1, var.scale = 1, ellipse = TRUE, circle = TRUE, alpha = 0.005) + ylim(-3.5,4) + xlim(-3,4)

#Feature Engineering

#SKEW Transformation
par(mfrow= c(1,1))
Val  <-   c(3,2.75,2.50,2.25,2,1.75,1.50,1.25,1,0.75,0.50,0.4,0.25,0,-0.25,-0.5,-0.75,-1,-1.25,-1.50,-1.75,-2)
symbox(forest_train$HD_hydro+0.01, powers=Val)
symbox(forest_train$HD_road+0.01, powers=Val)  #and look at symbox output   0.4
symbox(forest_train$HD_fire+0.01, powers=Val)  #and look at symbox output
symbox(forest_train$Hillshade_9am+0.01, powers=Val)  #and look at symbox output
symbox(forest_train$Hillshade_Noon+0.01, powers=Val)  #and look at symbox output


hist((forest_train$HD_hydro+0.01)^0.3)
hist((forest_train$HD_road+0.01)^0.4)
hist((forest_train$HD_fire+0.01)^0.4)
hist((forest_train$Hillshade_9am+0.01)^4.5)
hist((forest_train$Hillshade_Noon+0.01)^5)

forest_train$HD_hydro <- (forest_train$HD_hydro)^0.3
forest_train$HD_road <- log(forest_train$HD_road)^0.4
forest_train$HD_fire <- log(forest_train$HD_fire)^0.4
forest_train$Hillshade_9am <- log(forest_train$Hillshade_9am)^4.5
forest_train$Hillshade_Noon <- log(forest_train$Hillshade_Noon)^5

#Outlier Detection & treatment
#Detected Oultiers for two columns VD_hydro and Hillshade_9am we found 1804, 1893, 11939 for VD_hydro and 2721, 2790, 4928, 12807   for 
par(mfrow= c(1,1))
grubbs.test(forest_train$VD_hydro) ## Grubbs test to detect the outliers
plot(forest_train$VD_hydro, main = "Plotting Vertical Distance to Hydrology", ylab = "Distance to Hydrology") 
v<-identify(forest_train$VD_hydro,labels=row.names(forest_train)) # Identify the outliers
forest_train<-forest_train[-v,] # deleting the outliers

grubbs.test(forest_train$Hillshade_9am)    #Grubbs test to detect the outliers
plot(forest_train$Hillshade_9am, main = "Plotting Hillshade at 9 AM", ylab = "Hillshade Index")
v<-identify(forest_train$Hillshade_9am,labels=row.names(forest_train)) # Identify Outliers
forest_train<-forest_train[-v,] #Delete the outliers


#Feature Engineering-1
#Remove Columns with zero Variance
x <- forest_train[,1:54]
x <- data.frame(x)
f_covertype <- forest_train[,55]
y <- data.frame(f_covertype)
varnames <- colnames(x)

#measure std dev of col, drop if 0
to.drop = c()
for (i in 1:length(x)){
  stddev <- sd(x[,i], na.rm=TRUE)
  if (stddev == 0){
    to.drop <- c(to.drop,i)
    print(varnames[i])
  }
}
for (i in 1:length(to.drop)){
  x[,to.drop[i]-(i-1)] <- list(NULL)
}
varnames <- colnames(x)


drops <- c("Soil_Type7","Soil_Type15")
forest_train <- forest_train[ , !(names(forest_train) %in% drops)]


#Fature Engineering 2
#Pythogoral distance from from hydrology

HD = as.vector(forest_train$HD_hydro)
VD = as.vector(forest_train$VD_hydro)
euc.dist <- function(x1, x2) sqrt(((x1)^2+  (x2)^ 2))
forest_train$PythDist <- euc.dist(HD,VD)

#Fature Engineering 3
#MEan Hill Shade
forest_train$Hillshade_mean = (forest_train$Hillshade_9am + forest_train$Hillshade_3pm + forest_train$Hillshade_Noon) / 3

#Fature Engineering 4
#interaction between hillshade noon and 9 am 
#Train
forest_train$interaction_9amnoon= forest_train$Hillshade_9am*forest_train$Hillshade_Noon
forest_train$interaction_noon3pm=forest_train$Hillshade_Noon*forest_train$Hillshade_3pm
forest_train$interaction_9am3pm= forest_train$Hillshade_9am*forest_train$Hillshade_3pm

#Fature Engineering 5
#Cosine of Slope
forest_train$cosine_slope=cos((forest_train$Slope))

#Cosine of Aspect
forest_train$cosine_Aspect=cos((forest_train$Aspect))

#-----------------------------------------------------------
correlationplot<-ggcorr(forest_train[,c(1:10,54:60)], label = TRUE, label_size = 3, label_round = 2, label_alpha = TRUE, hjust = 00.75, size = 3,layout.exp = 1)
correlationplot
forest_train <- forest_train[,-c(5,7,8,56,57,58)]
correlationplot<-ggcorr(forest_train[,c(1:7,51:54)], label = TRUE, label_size = 3, label_round = 2, label_alpha = TRUE, hjust = 00.75, size = 3,layout.exp = 1)
correlationplot
#Partition
train_index<-createDataPartition(forest_train$Cover_Type, p = 0.7, list=F)
f_train<-forest_train[train_index,]
f_valid<-forest_train[-train_index,]
f_train_bk <- f_train
f_valid_bk <- f_valid
####################################
#BASIC GLM
par(mfrow=c(2,2))
f_trainglm <- lapply(f_train, as.numeric)
fit <- glm(Cover_Type~., data=f_trainglm,  family="gaussian")
summary(fit)
plot(fit)
par(mfrow=c(1,1))
#NULL DEVIANCE is deviance if you have an empty model
#RESIUDAL DEVIANCE: deviance based on actual model

#now let's take a look at the residuals

pearsonRes <-residuals(fit,type="pearson")
devianceRes <-residuals(fit,type="deviance")
rawRes <-residuals(fit,type="response")
studentDevRes<-rstudent(fit)
fv<-fitted(fit)

influence.measures(fit)
influencePlot(fit)
#From InfluencePlot we can see that 775, 3154 are the outliers
plot(predict(fit),residuals(fit))
vif(fit)
#Variance Inflation factor is a standard error which is very high for Bill1 to Bill6
#the "fit" glm object has a lot of useful information
names(fit)

head(fit$data)             # all of the data is stored
head(fit$y)                # the "true" value of the binary target  
head(fit$fitted.values)    # the predicted probabilities
fit$deviance               # the residual deviance

f_validglm <- lapply(f_valid, as.numeric)
pred_glm <- predict(fit, type = "response", newdata = f_validglm)


#-------------------------------------------------------------
#----------------------------------
#Random Forest  
set.seed(123)
which(is.na(f_train), arr.ind=TRUE)
p2tune <- tuneRF(f_train[,-50], f_train$Cover_Type, stepFactor = 0.35, plot = TRUE, ntreeTry = 100, trace = TRUE, improve=0.01, doBest=TRUE)
mtr <- p2tune$mtry
ntre <- p2tune$ntree
#Run RF again for optimized parameters (mtry and ntree) 
rf <- randomForest(factor(Cover_Type) ~ ., data=f_train, ntree=ntre,  mtry = mtr, importance = TRUE, proximity = TRUE, metric = "ROC", do.trace = 25)
summary(rf)
#prediction
pred_rf <- predict(rf, f_valid, type="class", norm.votes = TRUE, proximity = FALSE)
#Confusion Matrix
p1cm <- confusionMatrix(pred_rf, f_valid$Cover_Type)
#Overall details
print(p1cm$overall)
importance(rf, type = 1)
varImpPlot(rf, type=1, main="Feature Importance", col="steelblue", pch=20)
#Area Under Curve
multiclass.roc(as.numeric(pred_rf), as.numeric(f_valid$Cover_Type))

###################################################################
#GBM
#GBM Without Hyper Parameter tuning
gbmfit <- gbm(factor(Cover_Type) ~ ., data = f_train, distribution="gaussian", 
            n.trees = 5000, shrinkage=0.05, interaction.depth=3,
            bag.fraction = 0.5, verbose = TRUE, train.fraction = 0.7,
            cv.folds = 10, repeats = 5)
predgbm <- predict(gbmfit, f_valid, type="response")
best.iter <- gbm.perf(gbmfit,method="cv")

#GBM with Hyper Parameter tuning
#fit control
fitControl <- trainControl(method = "repeatedcv",number = 10, repeats = 5)
#creating grid for gbm
gbmGrid <-  expand.grid(interaction.depth = 2,n.trees = c(3000), shrinkage = 0.15,n.minobsinnode = 10)

gbmFit <- train(factor(Cover_Type) ~ . ,data = f_train,
                method = "gbm",
                trControl = fitControl,
                verbose = TRUE,
                tuneGrid = gbmGrid)

pred_gbm <- predict(gbmFit, f_valid)
gbmFit$bestTune

cnf_gbm <- confusionMatrix(pred_gbm, f_valid$Cover_Type)
cnf_gbm$overall


#######################################
#SVM

fitControl <- trainControl(method = "repeatedcv",number = 10, repeats = 5)
model <- svm(Cover_Type ~ ., f_train, cost = 64, epsilon = 0.01, probability = TRUE, cross = 10)
summary(model)
pred_svm <- predict(model, f_valid)
cm_svm <- confusionMatrix(pred_svm, f_valid$Cover_Type)
cm_svm
#Overall details
print(cm_svm$overall)
#Area Under Curve
multiclass.roc(as.numeric(pred_svm), as.numeric(f_valid$Cover_Type))
dat <- f_train
dat$th <- rnorm(nrow(f_train))
plot(model, dat, Elevation~th)
tune_svm <- tune(svm, Cover_Type ~ ., f_train, 
                                     kernel='radial', 
                                     ranges = list(cost=c(0.1,1,10),
                                                   gamma=c(0.5, 1,2)))

summary(tune.out)

#########################################################################
#Decision Tree
####################
#Decision Tree rpart Model form Caret package and prepare resampling method

fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  repeats = 5)

#Information Gain
model_dt_ig <- train(factor(Cover_Type)~., data=f_train, 
                        method="rpart", 
                        parms = list(split = "information"),  #information gain,
                        tuneLength = 10,
                        trControl=fitControl)

pred_dt_ig <- predict(model_dt_ig, f_valid)
cm <- confusionMatrix(pred_dt_ig, f_valid$Cover_Type)
print(cm$overall)
#Area Under Curve
multiclass.roc(as.numeric(pred_dt_ig), as.numeric(f_valid$Cover_Type))

#Gini Index
model_dt_gini <- train(factor(Cover_Type)~., data=f_train, 
                             method = "rpart",
                             parms = list(split = "gini"),   #gini index
                             trControl=fitControl,
                             tuneLength = 10)

pred_dt_gini =predict(model_dt_gini, f_valid)
cm <- confusionMatrix(pred_dt_gini, f_valid$Cover_Type)
print(cm$overall)
multiclass.roc(as.numeric(pred_dt_gini), as.numeric(f_valid$Cover_Type))

model.accuracy(pred_svm, f_valid$Cover_Type)
model.accuracy(pred_rf, f_valid$Cover_Type)
model.accuracy(pred_dt_ig, f_valid$Cover_Type)
model.accuracy(pred_dt_gini, f_valid$Cover_Type)

# display results
plot(model_dt_gini)


#K- Means
#let's scale the data..
fscaled<-data.frame(scale(f_train[,1:7]))  #default is mean centering with scaling by standard deviation

#kmeans is a function in the standard R package "stats"
forest_knn <- kmeans(fscaled,7, nstart=10)          


clusplot(fscaled, forest_knn$cluster, main='2D representation of the Cluster solution',
         color=TRUE, shade=TRUE,labels=2, lines=0)

#confusin matrix of clusters compared to the target
table(f_train$Cover_Type,forest_knn$cluster)

#HIERARCHICAL CLUSTERING
#@ k = 7
d <- dist(fscaled, method = "euclidean") # Euclidean distance matrix.
H.fit <- hclust(d, method="ward.D")
plot(H.fit) # display dendogram
rect.hclust(H.fit, k=7, border="blue")

