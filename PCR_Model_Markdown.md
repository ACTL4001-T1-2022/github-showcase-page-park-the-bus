PCR Model
================
Nathan Ng
2022-04-10

Team: Part the bus 10/04/2022

The following code used the data in the spreadsheet attached below:  
[Data_Spreadsheet](CleanedDataset.xlsx)

## Training and Predicting using Tournament Dataset

#### Setting file directory and random seed

``` r
setwd("C:/Users/natha/OneDrive/University/4th Year/ACTL4001/Group Project")
set.seed(8)
```

#### Loading Packages

``` r
library(dplyr)
library(MASS)
library(readxl)
library(ggplot2)
library(ggfortify)
library(ISLR)
library(pls)
library(tidyr)
library(plyr)
```

#### Uploading of data and filtering of positions

``` r
TournamentSet <- read_excel("Cleaned Dataset.xlsx", sheet = 12, n_max = 550)
```

    ## New names:
    ## * League -> League...25
    ## * League -> League...26

``` r
FWSet <- TournamentSet[which(TournamentSet$Pos == "FW"),]
MWSet <- TournamentSet[which(TournamentSet$Pos == "MF"),]
DFSet <- TournamentSet[which(TournamentSet$Pos == "DF"),]
GKSet <- read_excel("Cleaned Dataset.xlsx", sheet = 13, n_max = 75)
```

#### Creating PCA plot to identify possible clusters

``` r
Dataset <- read_excel("Cleaned Dataset.xlsx", sheet = 12, n_max = 550)
```

    ## New names:
    ## * League -> League...25
    ## * League -> League...26

``` r
Dataset$Ranking <- floor(Dataset$`Nation Rank`/4)
Dataset_PCA <- prcomp(Dataset[,-c(1:3,26:28)], center = TRUE, scale. = TRUE)
summary(Dataset_PCA)
```

    ## Importance of components:
    ##                           PC1    PC2     PC3     PC4     PC5     PC6     PC7
    ## Standard deviation     1.6805 1.5188 1.46530 1.37994 1.27474 1.23754 1.14615
    ## Proportion of Variance 0.1228 0.1003 0.09335 0.08279 0.07065 0.06659 0.05712
    ## Cumulative Proportion  0.1228 0.2231 0.31644 0.39924 0.46989 0.53647 0.59359
    ##                            PC8     PC9    PC10    PC11   PC12    PC13    PC14
    ## Standard deviation     1.06891 1.02723 0.99165 0.97676 0.9312 0.90921 0.83802
    ## Proportion of Variance 0.04968 0.04588 0.04275 0.04148 0.0377 0.03594 0.03053
    ## Cumulative Proportion  0.64327 0.68915 0.73190 0.77338 0.8111 0.84702 0.87756
    ##                           PC15    PC16   PC17    PC18    PC19    PC20    PC21
    ## Standard deviation     0.79367 0.75740 0.7241 0.62031 0.52773 0.46952 0.34661
    ## Proportion of Variance 0.02739 0.02494 0.0228 0.01673 0.01211 0.00958 0.00522
    ## Cumulative Proportion  0.90494 0.92989 0.9527 0.96941 0.98152 0.99111 0.99633
    ##                           PC22    PC23
    ## Standard deviation     0.28565 0.05282
    ## Proportion of Variance 0.00355 0.00012
    ## Cumulative Proportion  0.99988 1.00000

``` r
autoplot(Dataset_PCA, data = Dataset, colour = "Ranking", loadings = TRUE, loadings.label = TRUE, loadings.label.size = 3, 
         loadings.colour = "blue", loadings.label.colour = "blue")
```

[Click here to see the plot.](PCA_Plot.png)

#### Clean Non-numeric Variables for filtered datasets

``` r
FWSetClean <- FWSet[,-c(1:3,5,26,27)]
MWSetClean <- MWSet[,-c(1:3,5,26,27)]
DFSetClean <- DFSet[,-c(1:3,5,26,27)]
TournamentSetClean <- TournamentSet[,-c(1:3,5,26,27)] 
```

### Modelling Entire Tournament Dataset

### Split data into a training and test set

``` r
sample <- sample(nrow(TournamentSet),nrow(TournamentSet)*0.7)
trainset <- TournamentSet[sample,]
testset <- TournamentSet[-sample,]
```

#### Removing non-numerical data

``` r
train <- trainset[,-c(1:3,5,26,27)]
test <- testset[,-c(1:3,5,26,27)]
```

#### Applying PCR to create model using training dataset

``` r
pcrmodel <- pcr(train$`Nation Rank` ~. , data = train, scale = TRUE, validation = "CV")
```

#### Validation plot to identify best number of principle components

``` r
validationplot(pcrmodel, val.type = "MSE")
```

![](PCR_Model_Markdown_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

#### Creating predictor vectors for training and testing set

``` r
x_train <- model.matrix(train$`Nation Rank` ~., train)[,-1]
x_test <- model.matrix(test$`Nation Rank` ~., test)[,-1]
```

#### Creating response vectors from training and testing set

``` r
y_train <- as.numeric(unlist(train[,22]))
y_test <- as.numeric(unlist(test[,22]))
```

#### Predicting training and testing sets with PCR model

``` r
pcrtrainpredict <- predict(pcrmodel, x_train, ncomp = 7)
pcrpredict <- predict(pcrmodel, x_test, ncomp = 7)
```

#### Computing the MSE of model

``` r
mean((pcrpredict-y_test)^2)
```

    ## [1] 48.78963

#### Tabulating results for both train and test datasets

``` r
TournamentTrain <- data.frame(trainset$Nation,y_train,pcrtrainpredict)
colnames(TournamentTrain) <- c("Nation", "True Value", "Predicted Value")
aggregate(TournamentTrain[,2:3], list(TournamentTrain$Nation), mean)
```

    ##                    Group.1 True Value Predicted Value
    ## 1               Bernepamar          8        13.02220
    ## 2            Byasier Pujan         15        12.54440
    ## 3                 Djipines         16        12.35515
    ## 4                  Dosqaly          9        12.27539
    ## 5         Eastern Niasland         23        12.69846
    ## 6         Eastern Sleboube         19        12.85592
    ## 7                     Esia         14        12.39948
    ## 8                 Galamily          7        12.58255
    ## 9          Giumle Lizeibon         10        12.53657
    ## 10      Greri Landmoslands         11        12.45793
    ## 11                  Ledian         18        12.80523
    ## 12        Leoneku Guidisia         17        12.50569
    ## 13          Manlisgamncent         13        12.52879
    ## 14                    Mico          4        12.47833
    ## 15                 New Uwi         20        12.64152
    ## 16                 Nganion          3        12.03130
    ## 17           Ngoque Blicri         21        12.79091
    ## 18      Nkasland Cronestan         22        12.77348
    ## 19 People's Land of Maneau          2        12.30215
    ## 20                Quewenia          5        12.64646
    ## 21          Sobianitedrucy          1        11.94942
    ## 22         Southern Ristan          6        12.16116
    ## 23         Varijitri Isles         24        12.45472
    ## 24                  Xikong         12        12.29793

``` r
TournamentTest <- data.frame(testset$Nation,y_test,pcrpredict)
colnames(TournamentTest) <- c("Nation", "True Value", "Predicted Value")
aggregate(TournamentTest[,2:3], list(TournamentTest$Nation), mean)
```

    ##                    Group.1 True Value Predicted Value
    ## 1               Bernepamar          8        11.93771
    ## 2            Byasier Pujan         15        12.61356
    ## 3                 Djipines         16        12.45277
    ## 4                  Dosqaly          9        12.32160
    ## 5         Eastern Niasland         23        13.50850
    ## 6         Eastern Sleboube         19        12.41023
    ## 7                     Esia         14        12.48081
    ## 8                 Galamily          7        12.55892
    ## 9          Giumle Lizeibon         10        12.80406
    ## 10      Greri Landmoslands         11        12.77021
    ## 11                  Ledian         18        12.17644
    ## 12        Leoneku Guidisia         17        12.70318
    ## 13          Manlisgamncent         13        12.99142
    ## 14                    Mico          4        12.44623
    ## 15                 New Uwi         20        12.91264
    ## 16                 Nganion          3        11.32861
    ## 17           Ngoque Blicri         21        12.55356
    ## 18      Nkasland Cronestan         22        12.24403
    ## 19 People's Land of Maneau          2        11.96760
    ## 20                Quewenia          5        13.07435
    ## 21          Sobianitedrucy          1        11.88655
    ## 22         Southern Ristan          6        12.22338
    ## 23         Varijitri Isles         24        12.24488
    ## 24                  Xikong         12        12.59960

### PCR Model for Forward Players

#### Split data into a training and test set

``` r
sampleF <- sample(nrow(FWSet),nrow(FWSet)*0.7)
trainsetF <- FWSet[sampleF,]
testsetF <- FWSet[-sampleF,]
```

#### Removing non-numerical data

``` r
trainF <- trainsetF[,-c(1:3,5,26,27)]
testF <- testsetF[,-c(1:3,5,26,27)]
```

#### Applying PCR to create model using training dataset

``` r
pcrmodelF <- pcr(trainF$`Nation Rank` ~. , data = trainF, scale = TRUE, validation = "CV")
```

#### Validation plot to identify best number of principle components

``` r
validationplot(pcrmodelF, val.type = "MSE")
```

![](PCR_Model_Markdown_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

#### Creating predictor vectors for training and testing set

``` r
x_trainF <- model.matrix(trainF$`Nation Rank` ~., trainF)[,-1]
x_testF <- model.matrix(testF$`Nation Rank` ~., testF)[,-1]
```

#### Creating response vectors from training and testing set

``` r
y_trainF <- as.numeric(unlist(trainF[,22]))
y_testF <- as.numeric(unlist(testF[,22]))
```

#### Predicting training and testing sets with PCR model

``` r
pcrtrainpredictF <- predict(pcrmodelF, x_trainF, ncomp = 6)
pcrpredictF <- predict(pcrmodelF, x_testF, ncomp = 6)
```

#### Computing the MSE of model

``` r
mean((pcrpredictF-y_testF)^2)
```

    ## [1] 40.33471

#### Tabulating results for both train and test datasets and identifying MSE

``` r
TournamentTrainF <- data.frame(trainsetF$Nation,y_trainF,pcrtrainpredictF)
colnames(TournamentTrainF) <- c("Nation", "True Value", "Predicted Value")
TourTrainF <- aggregate(TournamentTrainF[,2:3], list(TournamentTrainF$Nation), mean)
mean((TourTrainF$`True Value`-TourTrainF$`Predicted Value`)^2)
```

    ## [1] 41.95876

``` r
TournamentTestF <- data.frame(testsetF$Nation,y_testF,pcrpredictF)
colnames(TournamentTestF) <- c("Nation", "True Value", "Predicted Value")
TourTestF <-aggregate(TournamentTestF[,2:3], list(TournamentTestF$Nation), mean)
mean((TourTestF$`True Value`-TourTestF$`Predicted Value`)^2)
```

    ## [1] 40.65784

### PCR Model for Midfield Players

#### Split data into a training and test set

``` r
sampleM <- sample(nrow(MWSet),nrow(MWSet)*0.7)
trainsetM <- MWSet[sampleM,]
testsetM <- MWSet[-sampleM,]
```

#### Removing non-numerical data

``` r
trainM <- trainsetM[,-c(1:3,5,26,27)]
testM <- testsetM[,-c(1:3,5,26,27)]
```

#### Applying PCR to create model using training dataset

``` r
pcrmodelM <- pcr(trainM$`Nation Rank` ~. , data = trainM, scale = TRUE, validation = "CV")
```

#### Validation plot to identify best number of principle components

``` r
validationplot(pcrmodelM, val.type = "MSE")
```

![](PCR_Model_Markdown_files/figure-gfm/unnamed-chunk-27-1.png)<!-- -->

#### Creating predictor vectors for training and testing set

``` r
x_trainM <- model.matrix(trainM$`Nation Rank` ~., trainM)[,-1]
x_testM <- model.matrix(testM$`Nation Rank` ~., testM)[,-1]
```

#### Creating response vectors from training and testing set

``` r
y_trainM <- as.numeric(unlist(trainM[,22]))
y_testM <- as.numeric(unlist(testM[,22]))
```

#### Predicting training and testing sets with PCR model

``` r
pcrtrainpredictM <- predict(pcrmodelM, x_trainM, ncomp = 5)
pcrpredictM <- predict(pcrmodelM, x_testM, ncomp = 5)
```

#### Computing the MSE of model

``` r
mean((pcrpredictM-y_testM)^2)
```

    ## [1] 53.18518

#### Tabulating results for both train and test datasets and identifying MSE

``` r
TournamentTrainM <- data.frame(trainsetM$Nation,y_trainM,pcrtrainpredictM)
colnames(TournamentTrainM) <- c("Nation", "True Value", "Predicted Value")
TourTrainM <- aggregate(TournamentTrainM[,2:3], list(TournamentTrainM$Nation), mean)
mean((TourTrainM$`True Value`-TourTrainM$`Predicted Value`)^2)
```

    ## [1] 41.71186

``` r
TournamentTestM <- data.frame(testsetM$Nation,y_testM,pcrpredictM)
colnames(TournamentTestM) <- c("Nation", "True Value", "Predicted Value")
TourTestM <-aggregate(TournamentTestM[,2:3], list(TournamentTestM$Nation), mean)
mean((TourTestM$`True Value`-TourTestM$`Predicted Value`)^2)
```

    ## [1] 58.59819

### PCR Model for Defense Players

#### Split data into a training and test set

``` r
sampleD <- sample(nrow(DFSet),nrow(DFSet)*0.7)
trainsetD <- DFSet[sampleD,]
testsetD <- DFSet[-sampleD,]
```

#### Removing non-numerical data

``` r
trainD <- trainsetD[,-c(1:3,5,26,27)]
testD <- testsetD[,-c(1:3,5,26,27)]
```

#### Applying PCR to create model using training dataset

``` r
pcrmodelD <- pcr(trainD$`Nation Rank` ~. , data = trainD, scale = TRUE, validation = "CV")
```

#### Validation plot to identify best number of principle components

``` r
validationplot(pcrmodelD, val.type = "MSE")
```

![](PCR_Model_Markdown_files/figure-gfm/unnamed-chunk-36-1.png)<!-- -->

#### Creating predictor vectors for training and testing set

``` r
x_trainD <- model.matrix(trainD$`Nation Rank` ~., trainD)[,-1]
x_testD <- model.matrix(testD$`Nation Rank` ~., testD)[,-1]
```

#### Creating response vectors from training and testing set

``` r
y_trainD <- as.numeric(unlist(trainD[,22]))
y_testD <- as.numeric(unlist(testD[,22]))
```

#### Predicting training and testing sets with PCR model

``` r
pcrtrainpredictD <- predict(pcrmodelD, x_trainD, ncomp = 3)
pcrpredictD <- predict(pcrmodelD, x_testD, ncomp = 3)
```

#### Computing the MSE of model

``` r
mean((pcrpredictD-y_testD)^2)
```

    ## [1] 42.97234

#### Tabulating results for both train and test datasets and identifying MSE

``` r
TournamentTrainD <- data.frame(trainsetD$Nation,y_trainD,pcrtrainpredictD)
colnames(TournamentTrainD) <- c("Nation", "True Value", "Predicted Value")
TourTrainD <- aggregate(TournamentTrainD[,2:3], list(TournamentTrainD$Nation), mean)
mean((TourTrainD$`True Value`-TourTrainD$`Predicted Value`)^2)
```

    ## [1] 29.90225

``` r
TournamentTestD <- data.frame(testsetD$Nation,y_testD,pcrpredictD)
colnames(TournamentTestD) <- c("Nation", "True Value", "Predicted Value")
TourTestD <-aggregate(TournamentTestD[,2:3], list(TournamentTestD$Nation), mean)
mean((TourTestD$`True Value`-TourTestD$`Predicted Value`)^2)
```

    ## [1] 40.21384

## PCR Model for Goalkeeping Players

#### Split data into a training and test set

``` r
sampleG <- sample(nrow(GKSet),nrow(GKSet)*0.7)
trainsetG <- GKSet[sampleG,]
testsetG <- GKSet[-sampleG,]
```

#### Removing non-numerical data

``` r
trainG <- trainsetG[,-c(1:5)]
testG <- testsetG[,-c(1:5)]
```

#### Applying PCR to create model using training dataset

``` r
pcrmodelG <- pcr(trainG$`Nation Rank` ~. , data = trainG, scale = TRUE, validation = "CV")
```

#### Validation plot to identify best number of principle components

``` r
validationplot(pcrmodelG, val.type = "MSE")
```

![](PCR_Model_Markdown_files/figure-gfm/unnamed-chunk-45-1.png)<!-- -->

#### Creating predictor vectors for training and testing set

``` r
x_trainG <- model.matrix(trainG$`Nation Rank` ~., trainG)[,-1]
x_testG <- model.matrix(testG$`Nation Rank` ~., testG)[,-1]
```

#### Creating response vectors from training and testing set

``` r
y_trainG <- as.numeric(unlist(trainG[,7]))
y_testG <- as.numeric(unlist(testG[,7]))
```

#### Predicting training and testing sets with PCR model

``` r
pcrtrainpredictG <- predict(pcrmodelG, x_trainG, ncomp = 5)
pcrpredictG <- predict(pcrmodelG, x_testG, ncomp = 5)
```

#### Computing the MSE of model

``` r
mean((pcrpredictG-y_testG)^2)
```

    ## [1] 93.55432

#### Tabulating results for both train and test datasets and identifying MSE

``` r
TournamentTrainG <- data.frame(trainsetG$Nation,y_trainG,pcrtrainpredictG)
colnames(TournamentTrainG) <- c("Nation", "True Value", "Predicted Value")
TourTrainG <- aggregate(TournamentTrainG[,2:3], list(TournamentTrainG$Nation), mean)
mean((TourTrainG$`True Value`-TourTrainG$`Predicted Value`)^2)
```

    ## [1] 31.61848

``` r
TournamentTestG <- data.frame(testsetG$Nation, y_testG,pcrpredictG)
colnames(TournamentTestG) <- c("Nation", "True Value", "Predicted Value")
TourTestG <-aggregate(TournamentTestG[,2:3], list(TournamentTestG$Nation), mean)
mean((TourTestG$`True Value`-TourTestG$`Predicted Value`)^2)
```

    ## [1] 50.51594

## Results

#### Creating results table for training dataset

``` r
df <- merge(TourTrainD, TourTrainF, by = "Group.1", all = TRUE)
colnames(df) <- c("Group.1",2,3,4)
df <- merge(df, TourTrainG, by = "Group.1", all = TRUE)
df <- merge(df, TourTrainM, by = "Group.1", all = TRUE)
df<- df[-c(4,6,8)]
colnames(df) <- c("Nation", "True Rank", "D", "F", "G", "M")
df$AggregateMean <- rowMeans(df[,c(3:6)], na.rm = TRUE)
mean((df$`True Rank`-df$AggregateMean)^2)
```

    ## [1] 33.3284

#### Creating results table for test dataset

``` r
df2 <- merge(TourTestD, TourTestF, by = "Group.1", all = TRUE)
colnames(df2) <- c("Group.1",2,3,4)
df2 <- merge(df2, TourTestG, by = "Group.1", all = TRUE)
df2 <- merge(df2, TourTestM, by = "Group.1", all = TRUE)
df2<- df2[-c(4,6,8)]
colnames(df2) <- c("Nation", "True Rank", "D", "F", "G", "M")
df2$AggregateMean <- rowMeans(df2[,c(3:6)], na.rm = TRUE)
df2[19,2]<- 2
mean((df2$`True Rank`-df2$AggregateMean)^2)
```

    ## [1] 41.04369
