GBM model
================
Team: Park the bus
06/04/2022

## Section 1: Both training and predicting using tournament dataset

#### Loading packages

``` r
library(xgboost)
library(caTools)
library(dplyr)
library(writexl)
```

#### Loading Data

``` r
nongk_data <- read.csv("/Users/williamli/Documents/Uni/2022 Sem 1/ACTL4001/New_Non_GK.csv",na.strings = 'Unknown')
gk_data <- read.csv("/Users/williamli/Documents/Uni/2022 Sem 1/ACTL4001/New_GK.csv", na.strings = 'Unknown')
```

#### Filter player positions

``` r
df_data <- filter(nongk_data, Pos == "DF")
mf_data <- filter(nongk_data, Pos == "MF")
fw_data <- filter(nongk_data, Pos == "FW")
```

### Defenders

#### Split into training and testing data sets, independent and dependent variables

``` r
set.seed(1)
df_split <- sample.split(Y = df_data$Nation.Rank, SplitRatio = 0.7)
df_train <- subset(x = df_data, df_split == TRUE)
df_test <- subset(x = df_data, df_split == FALSE)

df_y_train <- as.integer(df_train$Nation.Rank) - 1
df_y_test <- as.integer(df_test$Nation.Rank) - 1
df_X_train_full <- df_train %>% select(-Nation.Rank)
df_X_test_full <- df_test %>% select(-Nation.Rank)
```

#### Make data frame without characters

``` r
df_X_train=df_X_train_full[,-which(sapply(df_X_train_full, class)=="character")]
df_X_test=df_X_test_full[,-which(sapply(df_X_test_full, class)=="character")]
```

#### Constructing DMatrix data structures and parameters

``` r
df_xgb_train <- xgb.DMatrix(data = as.matrix(df_X_train), label = df_y_train)
df_xgb_test <- xgb.DMatrix(data = as.matrix(df_X_test), label = df_y_test)
```

#### Cross validation to choose the best number of rounds

``` r
df_cv <- xgb.cv(data = df_xgb_train, nfold = 5, nrounds = 100, metrics = "rmse", showsd = T) #53 boosting rounds gave lowest test rmse
df_xgb_model <- xgb.train(data = df_xgb_train, nrounds = 53, verbose = 1) #set parameter of 53 boosts
```

#### Prediction on each player

``` r
df_xgb_preds <- predict(df_xgb_model, df_xgb_test, reshape = TRUE)
df_xgb_preds
```

    ##  [1]  6.4184856  7.5234089 12.9889631 11.2612600 14.2458162 14.1030846
    ##  [7] 13.6249676 13.5946379  6.7469358 14.7526026 15.0818768  8.7644758
    ## [13] 20.8942852 22.1835613  7.6875262  7.3347015  6.4774804 12.6664906
    ## [19] 16.1983471  8.2328138  8.0269423  3.6363149 10.7488785  9.9816799
    ## [25] 17.2412472 16.7149792 16.9240227 10.6960850 12.4387589  9.9206524
    ## [31]  8.2203102 18.6267452 15.3833132  2.5231082  6.3963852  9.9383755
    ## [37] 18.5824013 11.8745861  9.7581100 13.0261602 13.1366549  2.3506474
    ## [43]  5.9735637 12.3692646  0.9779601 15.5460930 10.8641882  8.9397354
    ## [49]  8.8017149  6.3864484 12.0863037 11.4338045  8.3119984 18.0731010

``` r
df_test$df_Pred <-  df_xgb_preds + 1 #put this back into the test set 
```

#### For every nation, take the average output of its defenders to give an overall defense rank. The same will be done for MF, FW and GK.

``` r
df_pred <- df_test %>%
  group_by(Nation) %>%  
  summarize_at(vars(Nation.Rank:df_Pred), mean)
df_pred
```

    ## # A tibble: 24 x 3
    ##    Nation             Nation.Rank df_Pred
    ##    <chr>                    <dbl>   <dbl>
    ##  1 Bernepamar                   8    7.97
    ##  2 Byasier Pujan               15   13.8 
    ##  3 Djipines                    16   14.9 
    ##  4 Dosqaly                      9   11.2 
    ##  5 Eastern Niasland            23   15.9 
    ##  6 Eastern Sleboube            19   18.3 
    ##  7 Esia                        14    8.17
    ##  8 Galamily                     7   15.4 
    ##  9 Giumle Lizeibon             10    9.13
    ## 10 Greri Landmoslands          11    4.64
    ## # … with 14 more rows

#### Mean squared error

``` r
mean((df_pred$Nation.Rank - df_pred$df_Pred)^2)
```

    ## [1] 30.30965

#### Investigating variable importance

``` r
df_boost_model <- xgboost(data = df_xgb_train, nrounds = 9, verbose = 0)
df_importance <- xgb.importance(feature_names = colnames(df_xgb_train), model = df_boost_model)
df_importance
```

    ##                    Feature         Gain       Cover   Frequency
    ##  1:                   X90s 0.1981718877 0.125619579 0.104838710
    ##  2:             Medium.Att 0.1471455937 0.030824040 0.012096774
    ##  3:              Total.Cmp 0.0993855193 0.054213135 0.040322581
    ##  4:        Pressures.Press 0.0688377150 0.068773234 0.020161290
    ##  5:              Total.Att 0.0466689350 0.040582404 0.012096774
    ##  6:      Pressures.Mid.3rd 0.0327635408 0.014869888 0.008064516
    ##  7:       Expected.npxG.Sh 0.0324103016 0.017657993 0.016129032
    ##  8:              Short.Cmp 0.0294135578 0.022459727 0.024193548
    ##  9:            Medium.Cmp. 0.0291961157 0.019671623 0.020161290
    ## 10:        Vs.Dribbles.Tkl 0.0288968700 0.035470880 0.028225806
    ## 11:      Pressures.Att.3rd 0.0237518459 0.025402726 0.016129032
    ## 12:        Tackles.Def.3rd 0.0235079683 0.009138786 0.012096774
    ## 13:               Long.Cmp 0.0211913370 0.008983891 0.016129032
    ## 14:      Pressures.Def.3rd 0.0154202493 0.015799257 0.012096774
    ## 15:       Vs.Dribbles.Tkl. 0.0147963151 0.056536555 0.040322581
    ## 16:                   Prog 0.0129699440 0.012236679 0.016129032
    ## 17:       Vs.Dribbles.Past 0.0125373160 0.032682776 0.024193548
    ## 18:            Expected.xG 0.0122264342 0.006505576 0.016129032
    ## 19:         Pressures.Succ 0.0100779066 0.017193309 0.016129032
    ## 20:            Pressures.. 0.0092157321 0.014869888 0.016129032
    ## 21:                   X1.3 0.0090164816 0.015024783 0.008064516
    ## 22:             Short.Cmp. 0.0082130802 0.010377943 0.020161290
    ## 23:         Standard.Sh.90 0.0081838486 0.014869888 0.020161290
    ## 24:          Total.PrgDist 0.0077904910 0.026641884 0.016129032
    ## 25:            Tackles.Tkl 0.0070304529 0.004337051 0.004032258
    ## 26:              Short.Att 0.0069084959 0.006350682 0.008064516
    ## 27:            Standard.Sh 0.0068085103 0.009603470 0.032258065
    ## 28:             Total.Cmp. 0.0066943866 0.005731103 0.016129032
    ## 29: X2020.League.Indicator 0.0052997444 0.016728625 0.012096774
    ## 30:           Standard.SoT 0.0049914303 0.003097893 0.012096774
    ## 31:          Standard.Dist 0.0046406551 0.011771995 0.008064516
    ## 32:        Tackles.Mid.3rd 0.0045716031 0.016883519 0.012096774
    ## 33:              Long.Cmp. 0.0045645194 0.012391574 0.012096774
    ## 34:               Long.Att 0.0042760597 0.014405204 0.020161290
    ## 35:                Tkl.Int 0.0042129494 0.006815366 0.004032258
    ## 36:                    Clr 0.0039245226 0.008209418 0.008064516
    ## 37:                   A.xA 0.0035225420 0.009448575 0.004032258
    ## 38:                   Born 0.0029320025 0.042596035 0.012096774
    ## 39:                    PPA 0.0025948172 0.009448575 0.008064516
    ## 40:       Expected.np.G.xG 0.0023042828 0.011462206 0.016129032
    ## 41:          Standard.SoT. 0.0021348731 0.005885998 0.004032258
    ## 42:                    Ast 0.0020723479 0.010223048 0.012096774
    ## 43:             Medium.Cmp 0.0019262398 0.007125155 0.012096774
    ## 44:            Blocks.Pass 0.0017942091 0.006040892 0.008064516
    ## 45:                    Age 0.0017632609 0.014714994 0.096774194
    ## 46:         Performance.PK 0.0016833548 0.006195787 0.020161290
    ## 47:                    Int 0.0016788299 0.005111524 0.012096774
    ## 48:        Standard.SoT.90 0.0016708719 0.004491945 0.012096774
    ## 49:            Standard.FK 0.0015034028 0.002633209 0.004032258
    ## 50:            Blocks.ShSv 0.0014757075 0.005576208 0.008064516
    ## 51:                    Gls 0.0013506911 0.005576208 0.024193548
    ## 52:                     KP 0.0009127699 0.006350682 0.008064516
    ## 53:        Vs.Dribbles.Att 0.0008167928 0.005885998 0.004032258
    ## 54:                  CrsPA 0.0007811101 0.007744734 0.012096774
    ## 55:          Expected.G.xG 0.0003392695 0.003097893 0.008064516
    ## 56:                    Err 0.0002785532 0.004956629 0.004032258
    ## 57:          Total.TotDist 0.0002451054 0.004027261 0.004032258
    ## 58:          Expected.npxG 0.0001634336 0.002633209 0.008064516
    ## 59:          Standard.G.Sh 0.0001300297 0.001394052 0.004032258
    ## 60:           Tackles.TklW 0.0001110989 0.002788104 0.004032258
    ## 61:         Standard.G.SoT 0.0001020876 0.001858736 0.004032258
    ##                    Feature         Gain       Cover   Frequency

#### Plot

``` r
xgb.plot.tree(model = df_boost_model, trees = 1)
```

[Click here to see the plot](GBM-Plot.png)

### Same code for midfielders

``` r
##Split into training and testing data sets, independent and dependent variables
set.seed(1)
mf_split <- sample.split(Y = mf_data$Nation.Rank, SplitRatio = 0.7)
mf_train <- subset(x = mf_data, mf_split == TRUE)
mf_test <- subset(x = mf_data, mf_split == FALSE)

mf_y_train <- as.integer(mf_train$Nation.Rank) - 1
mf_y_test <- as.integer(mf_test$Nation.Rank) - 1
mf_X_train_full <- mf_train %>% select(-Nation.Rank)
mf_X_test_full <- mf_test %>% select(-Nation.Rank)

#mf_X_train[sapply(mf_X_train, is.character)] <- lapply(mf_X_train[sapply(mf_X_train, is.character)], as.numeric) #convert character types to numeric types for Xgb
#mf_X_test[sapply(mf_X_test, is.character)] <- lapply(mf_X_test[sapply(mf_X_test, is.character)], as.numeric) #convert character types to numeric types for Xgb
mf_X_train=mf_X_train_full[,-which(sapply(mf_X_train_full, class)=="character")]
mf_X_test=mf_X_test_full[,-which(sapply(mf_X_test_full, class)=="character")]

##Constructing DMatrix data structures and parameters
mf_xgb_train <- xgb.DMatrix(data = as.matrix(mf_X_train), label = mf_y_train)
mf_xgb_test <- xgb.DMatrix(data = as.matrix(mf_X_test), label = mf_y_test) 

###Cross validation to choose the best nrounds
mf_cv <- xgb.cv(data = mf_xgb_train, nfold = 5, nrounds = 100, metrics = "rmse", showsd = T) #53 boosting rounds gave lowest test rmse

mf_xgb_model <- xgb.train(data = mf_xgb_train, nrounds = 53, verbose = 1) #set parameter of 9 boosts

##prediction and comparison to actual Rank
mf_xgb_preds <- predict(mf_xgb_model, as.matrix(mf_X_test), reshape = TRUE)

mf_xgb_preds

mf_test$mf_Pred <-  mf_xgb_preds + 1 #put this back into the test set 


mf_pred <- mf_test %>%
  group_by(Nation) %>%  
  summarize_at(vars(Nation.Rank:mf_Pred), mean) #take the average of each nation's players to predict the overall nation's rank 
mf_pred

mean((mf_pred$Nation.Rank - mf_pred$mf_Pred)^2) #mse

##Investigating variable importance
mf_boost_model <- xgboost(data = mf_xgb_train, nrounds = 9, verbose = 1)
mf_importance <- xgb.importance(feature_names = colnames(mf_xgb_train), model = mf_boost_model)
mf_importance

#plot 

xgb.plot.tree(model = mf_boost_model, trees = 1)
```

### Same code for forwards

``` r
##Split into training and testing data sets, independent and dependent variables
set.seed(1)
fw_split <- sample.split(Y = fw_data$Nation.Rank, SplitRatio = 0.7)
fw_train <- subset(x = fw_data, fw_split == TRUE)
fw_test <- subset(x = fw_data, fw_split == FALSE)

fw_y_train <- as.integer(fw_train$Nation.Rank) - 1
fw_y_test <- as.integer(fw_test$Nation.Rank) - 1
fw_X_train_full <- fw_train %>% select(-Nation.Rank)
fw_X_test_full <- fw_test %>% select(-Nation.Rank)

#fw_X_train[sapply(fw_X_train, is.character)] <- lapply(fw_X_train[sapply(fw_X_train, is.character)], as.numeric) #convert character types to numeric types for Xgb
#fw_X_test[sapply(fw_X_test, is.character)] <- lapply(fw_X_test[sapply(fw_X_test, is.character)], as.numeric) #convert character types to numeric types for Xgb
fw_X_train=fw_X_train_full[,-which(sapply(fw_X_train_full, class)=="character")]
fw_X_test=fw_X_test_full[,-which(sapply(fw_X_test_full, class)=="character")]

##Constructing DMatrix data structures and parameters
fw_xgb_train <- xgb.DMatrix(data = as.matrix(fw_X_train), label = fw_y_train)
fw_xgb_test <- xgb.DMatrix(data = as.matrix(fw_X_test), label = fw_y_test) 

###Cross validation to choose the best nrounds
fw_cv <- xgb.cv(data = fw_xgb_train, nfold = 5, nrounds = 100, metrics = "rmse", showsd = T) #23 boosting rounds gave lowest test rmse

fw_xgb_model <- xgb.train(data = fw_xgb_train, nrounds = 23, verbose = 1) #set parameter of 23 boosts

##prediction and comparison to actual Rank
fw_xgb_preds <- predict(fw_xgb_model, as.matrix(fw_X_test), reshape = TRUE)

fw_xgb_preds

fw_test$fw_Pred <-  fw_xgb_preds + 1 #put this back into the test set 


fw_pred <- fw_test %>%
  group_by(Nation) %>%  
  summarize_at(vars(Nation.Rank:fw_Pred), mean) #take the average of each nation's players to predict the overall nation's rank 
fw_pred

mean((fw_pred$Nation.Rank - fw_pred$fw_Pred)^2) #mse

##Investigating variable importance
fw_boost_model <- xgboost(data = fw_xgb_train, nrounds = 9, verbose = 1)
fw_importance <- xgb.importance(feature_names = colnames(fw_xgb_train), model = fw_boost_model)
fw_importance

#plot 

xgb.plot.tree(model = fw_boost_model, trees = 1)
```

### Same code for goalkeepers

``` r
##Split into training and testing data sets, independent and dependent variables
set.seed(1)
gk_split <- sample.split(Y = gk_data$Nation.Rank, SplitRatio = 0.7)
gk_train <- subset(x = gk_data, gk_split == TRUE)
gk_test <- subset(x = gk_data, gk_split == FALSE)

gk_y_train <- as.integer(gk_train$Nation.Rank) - 1
gk_y_test <- as.integer(gk_test$Nation.Rank) - 1
gk_X_train_full <- gk_train %>% select(-Nation.Rank)
gk_X_test_full <- gk_test %>% select(-Nation.Rank)

#gk_X_train[sapply(gk_X_train, is.character)] <- lapply(gk_X_train[sapply(gk_X_train, is.character)], as.numeric) #convert character types to numeric types for Xgb
#gk_X_test[sapply(gk_X_test, is.character)] <- lapply(gk_X_test[sapply(gk_X_test, is.character)], as.numeric) #convert character types to numeric types for Xgb
gk_X_train=gk_X_train_full[,-which(sapply(gk_X_train_full, class)=="character")]
gk_X_test=gk_X_test_full[,-which(sapply(gk_X_test_full, class)=="character")]

##Constructing DMatrix data structures and parameters
gk_xgb_train <- xgb.DMatrix(data = as.matrix(gk_X_train), label = gk_y_train)
gk_xgb_test <- xgb.DMatrix(data = as.matrix(gk_X_test), label = gk_y_test) 

###Cross validation to choose the best nrounds
gk_cv <- xgb.cv(data = gk_xgb_train, nfold = 5, nrounds = 100, metrics = "rmse", showsd = T) #14 boosting rounds gave lowest test rmse

gk_xgb_model <- xgb.train(data = gk_xgb_train, nrounds = 14, verbose = 1) #set parameter of 14 boosts

##prediction and comparison to actual Rank
gk_xgb_preds <- predict(gk_xgb_model, as.matrix(gk_X_test), reshape = TRUE)

gk_xgb_preds

gk_test$gk_Pred <-  gk_xgb_preds + 1 #put this back into the test set 


gk_pred <- gk_test %>%
  group_by(Nation) %>%  
  summarize_at(vars(Nation.Rank:gk_Pred), mean) #take the average of each nation's players to predict the overall nation's rank 
gk_pred

mean((gk_pred$Nation.Rank - gk_pred$gk_Pred)^2) #mse

##Investigating variable importance
gk_boost_model <- xgboost(data = gk_xgb_train, nrounds = 9, verbose = 1)
gk_importance <- xgb.importance(feature_names = colnames(gk_xgb_train), model = gk_boost_model)
gk_importance

#plot 

xgb.plot.tree(model = mf_boost_model, trees = 1)
```

### Team Prediction.

#### Method 1: Using average of the position, then weighting all positions equally

``` r
position1 <- merge(df_pred, fw_pred, by.y= "Nation", by.x = "Nation",all.x=TRUE, all.y= TRUE)
position2 <- merge(position1, mf_pred, by.y= "Nation", by.x = "Nation",all.x=TRUE, all.y= TRUE)
position3 <- merge(position2, gk_pred, by.y= "Nation", by.x = "Nation",all.x=TRUE, all.y= TRUE)
position4 <- position3[c(1,2,3,5,7,9)]
position4$mean_rank <- rowMeans(position4[,3:6], na.rm=TRUE)
```

#### Method 2: Equally weighting across all players in a team.

``` r
df_test$Pred <- df_test$df_Pred
mf_test$Pred <- mf_test$mf_Pred
fw_test$Pred <- fw_test$fw_Pred
gk_test$Pred <- gk_test$gk_Pred
df_test2 <- df_test[c('Player', 'Nation', 'Nation.Rank', 'Pred')]
mf_test2 <- mf_test[c('Player', 'Nation', 'Nation.Rank', 'Pred')]
fw_test2 <- fw_test[c('Player', 'Nation', 'Nation.Rank', 'Pred')]
gk_test2 <- gk_test[c('Player', 'Nation', 'Nation.Rank', 'Pred')]

player <- rbind(df_test2, mf_test2, fw_test2, gk_test2)

mean_vec <- c()
for (nation in position4$Nation){
  mean_vec=append(mean_vec, mean(player$Pred[player$Nation==nation]))
}
position4$player_mean <- mean_vec

results <- position4[order(position4$Nation.Rank.x),]
```

#### Method 3: Using average of the position, but weighting different positions differently

``` r
results[is.na(results$gk_Pred),]$gk_Pred<-results[is.na(results$gk_Pred),]$df_Pred*4/10+results[is.na(results$gk_Pred),]$mf_Pred*4/10+results[is.na(results$gk_Pred),]$fw_Pred*2/10
results$weighted_rank=results$df_Pred*4/11+results$mf_Pred*4/11+results$fw_Pred*2/11+results$gk_Pred*1/11
```

``` r
results
```

    ##                     Nation Nation.Rank.x   df_Pred   fw_Pred  mf_Pred   gk_Pred
    ## 21          Sobianitedrucy             1 10.129414  8.518067 12.49957  2.283109
    ## 19 People's Land of Maneau             2  8.743651 10.918339 15.03746  3.145949
    ## 16                 Nganion             3  5.459747  6.794359 10.54327  4.349906
    ## 14                    Mico             4 10.070481 12.639655 10.16527  5.954307
    ## 20                Quewenia             5 10.171414 12.594725 14.47560 18.931726
    ## 22         Southern Ristan             6  9.042633 10.088250 12.72812  5.454830
    ## 8                 Galamily             7 15.432419 14.898273 14.69915 13.376601
    ## 1               Bernepamar             8  7.970947 13.331272 11.67979  5.228739
    ## 4                  Dosqaly             9 11.170787 12.766638 13.88907 12.577270
    ## 9          Giumle Lizeibon            10  9.129878 11.409662 13.25450  8.137301
    ## 10      Greri Landmoslands            11  4.636315 14.365783 11.29976 15.382756
    ## 24                  Xikong            12 14.192550 10.107752 12.78920 17.214275
    ## 13          Manlisgamncent            13 12.567422 12.955068 13.62423  8.484883
    ## 7                     Esia            14  8.166569 12.731366 11.03689  9.322652
    ## 2            Byasier Pujan            15 13.832013 14.065174 11.76516 10.839200
    ## 3                 Djipines            16 14.864026 13.641050 15.70379 13.978896
    ## 12        Leoneku Guidisia            17 17.960083 13.834584 11.36806  6.164259
    ## 11                  Ledian            18 11.365279 15.614917 13.05738 20.300558
    ## 6         Eastern Sleboube            19 18.280774 12.289407 12.14424  6.958354
    ## 15                 New Uwi            20 18.005029 13.353114 19.46367 20.256823
    ## 17           Ngoque Blicri            21 14.465121 18.218481 15.57854 16.848222
    ## 18      Nkasland Cronestan            22 12.392135 15.327122 14.31505 13.030108
    ## 5         Eastern Niasland            23 15.917240 12.744478 14.86414 17.325787
    ## 23         Varijitri Isles            24 12.760054 11.814470 17.43018 17.388887
    ##    mean_rank player_mean weighted_rank
    ## 21  8.357539    9.338327      9.985015
    ## 19  9.461350   10.363550     10.918825
    ## 16  6.786819    7.134950      7.450061
    ## 14  9.707429   10.243589     10.197875
    ## 20 14.043366   13.486350     12.973566
    ## 22  9.328458   10.106917     10.246758
    ## 8  14.601612   14.776614     14.881767
    ## 1   9.552686   10.359067     10.044929
    ## 4  12.608831   12.791722     12.577270
    ## 9  10.482835   11.154393     10.954012
    ## 10 11.421154   11.807132      9.805330
    ## 24 13.575944   13.439246     13.214252
    ## 13 11.907902   12.466675     12.651058
    ## 7  10.314368   10.169858     10.145382
    ## 2  12.625388   12.999490     12.850751
    ## 3  14.546941   14.762552     14.866569
    ## 12 12.331746   14.154533     13.740544
    ## 11 15.084533   14.339387     13.565548
    ## 6  12.418194   13.833497     13.930658
    ## 15 17.769659   18.091223     17.894350
    ## 17 16.277590   15.659877     15.769074
    ## 18 13.766104   13.871246     13.683008
    ## 5  15.212911   14.665124     15.085478
    ## 23 14.848398   15.287715     14.707160

#### Analysis of errors

Method 1

``` r
mean((results$Nation.Rank.x-results$mean_rank)^2)
```

    ## [1] 27.55403

Method 2

``` r
mean((results$Nation.Rank.x-results$player_mean)^2)
```

    ## [1] 28.38349

Method 3

``` r
mean((results$Nation.Rank.x-results$weighted_rank)^2)
```

    ## [1] 29.57723

Choosing Method 1 as it has best performance

mean error fairly close to zero (which is what we want)

``` r
(mean_error=mean(results$Nation.Rank.x-results$mean_rank))
```

    ## [1] 0.1236768

Std dev of errors

``` r
sd(results$Nation.Rank.x-results$mean_rank)
```

    ## [1] 5.360603

Distribution of errors

``` r
hist(results$Nation.Rank.x-results$mean_rank,breaks = 8)
```

![](SOA-GBM-Model_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

## Section 2: Using tournament model to predict on league data

### Data Prep

#### Load data

``` r
nongk_league_data <- read.csv("/Users/williamli/Documents/Uni/2022 Sem 1/ACTL4001/League_non_GK_scaled.csv", na.strings = 'Unknown')
gk_league_data <- read.csv("/Users/williamli/Documents/Uni/2022 Sem 1/ACTL4001/League_GK_scaled.csv", na.strings = 'Unknown')
```

#### Compute 2020 and 2021 Indicator based on which years the player played

``` r
nongk_league_data$X2020.League.Indicator=0
nongk_league_data$X2021.League.Indicator=0
nongk_league_data$X2020.League.Indicator[nongk_league_data$Year==2020]=1
nongk_league_data$X2021.League.Indicator[nongk_league_data$Year==2021]=1
nongk_league_data$X2021.League.Indicator[nongk_league_data$Year==2020]=
  nongk_league_data$Player[nongk_league_data$Year==2020] %in% nongk_league_data$Player[nongk_league_data$Year==2021]
nongk_league_data$X2020.League.Indicator[nongk_league_data$Year==2021]=
  nongk_league_data$Player[nongk_league_data$Year==2021] %in% nongk_league_data$Player[nongk_league_data$Year==2020]

gk_league_data$X2020.League.Indicator=0
gk_league_data$X2021.League.Indicator=0
gk_league_data$X2020.League.Indicator[gk_league_data$Year==2020]=1
gk_league_data$X2021.League.Indicator[gk_league_data$Year==2021]=1
gk_league_data$X2021.League.Indicator[gk_league_data$Year==2020]=
  gk_league_data$Player[gk_league_data$Year==2020] %in% gk_league_data$Player[gk_league_data$Year==2021]
gk_league_data$X2020.League.Indicator[gk_league_data$Year==2021]=
  gk_league_data$Player[gk_league_data$Year==2021] %in% gk_league_data$Player[gk_league_data$Year==2020]
```

#### Setting dummy column because xgboost requires label

``` r
nongk_league_data$Nation.Rank=0
gk_league_data$Nation.Rank=0
```

#### Filter player positions

``` r
df_l_data <- filter(nongk_league_data, Pos == "DF")
mf_l_data <- filter(nongk_league_data, Pos == "MF")
fw_l_data <- filter(nongk_league_data, Pos == "FW")
```

### Predict League defenders

#### Split the independent and dependent variables

``` r
df_l_y_test <- as.integer(df_l_data$Nation.Rank) 
df_l_X_test <- df_l_data %>% select(-Nation.Rank, -Squad, -Salary)
```

#### Remove character types for Xgb and set up Xgb matrix

``` r
df_l_X_test=df_l_X_test[,-which(sapply(df_l_X_test, class)=="character")]
df_l_xgb_test <- xgb.DMatrix(data = as.matrix(df_l_X_test), label = df_l_y_test)
colnames(df_l_xgb_test) <- NULL 
```

#### We are predicting on the league data now, so the whole tournament dataset can be used for training, no need for train test split

``` r
df_y_both <- as.integer(df_data$Nation.Rank) - 1
df_X_both <- df_data %>% select(-Nation.Rank)
df_X_both=df_X_both[,-which(sapply(df_X_both, class)=="character")]
df_xgb_both <- xgb.DMatrix(data = as.matrix(df_X_both), label = df_y_both)
```

#### Train the model on tournament data

``` r
df_xgb_model <- xgb.train(data = df_xgb_both, nrounds = 53, verbose = 1) #set parameter of 53 boosts
```

#### Predict on league dataset

``` r
df_l_xgb_preds <- predict(df_xgb_model, df_l_xgb_test, reshape = TRUE)
df_l_data$df_l_Pred <-  df_l_xgb_preds
head(df_l_data)
```

    ##          Player                  Nation Pos             Squad Age Born
    ## 1     I. Winter        Danan Seekeeling  DF Fanatical Outlaws  27 1991
    ## 2  P. Nakubulwa                 Dosqaly  DF Fanatical Outlaws  22 1997
    ## 3   M. Mahlangu          Imaar Vircoand  DF Fanatical Outlaws  34 1985
    ## 4      I. Huber          Lenia Gerdanho  DF Fanatical Outlaws  25 1993
    ## 5 A. Kobusingye People's Land of Maneau  DF Fanatical Outlaws  18 2000
    ## 6     F. Kizito People's Land of Maneau  DF Fanatical Outlaws  19 2000
    ##        X90s        Gls Standard.Sh Standard.SoT Standard.SoT. Standard.Sh.90
    ## 1 2.1547673 0.06216787   0.7525640   0.33391887   38.41363681     0.80871867
    ## 2 5.1001685 0.12433574   0.4007159   0.10274427   38.45202166     0.52951818
    ## 3 1.0823562 0.03552450   0.1172827   0.03424809    0.00000000     0.08664843
    ## 4 1.7387315 0.00000000   0.5179986   0.21405056   38.29848227     0.38510413
    ## 5 0.8171541 0.02664337   0.1563769   0.09418225   95.96212044     0.24069008
    ## 6 0.4989115 0.00000000   0.4007159   0.01712405    0.03838485     0.25994529
    ##   Standard.SoT.90 Standard.G.Sh Standard.G.SoT Standard.Dist Standard.FK
    ## 1      0.27203424    0.05766596    0.271592188      7.122742  0.00000000
    ## 2      0.25448365    0.12494292    0.504385492      7.621725  0.00000000
    ## 3      0.04387649    0.00000000    0.276539112     11.926679  0.00000000
    ## 4      0.24570835    0.00000000    0.009699721     28.647512  0.00000000
    ## 5      0.26325895    0.00000000    0.000000000     24.371910  0.01215740
    ## 6      0.03510119    0.00000000    0.276539112     27.766953  0.03647221
    ##   Performance.PK Performance.PKatt Expected.xG Expected.npxG Expected.npxG.Sh
    ## 1     0.00000000        0.02800001  0.19741236    0.09271732       0.07565463
    ## 2     0.09984084        0.04200001  0.00000000    0.00000000       0.15130925
    ## 3     0.02995225        0.12600004  0.08973289    0.01854346       0.03782731
    ## 4     0.07987267        0.00000000  0.08075960    0.03708693       0.00000000
    ## 5     0.00000000        0.00000000  0.00000000    0.00000000       0.03782731
    ## 6     0.03993634        0.00000000  0.00000000    0.00000000       0.00000000
    ##   Expected.G.xG Expected.np.G.xG Total.Cmp Total.Att Total.Cmp. Total.TotDist
    ## 1   -0.12087816      -0.17637244 12.477754  16.47320   77.03705      273.3562
    ## 2    0.10744725       0.10853689 11.072765  13.68937   82.85097      222.2414
    ## 3    0.04029272      -0.09496978 16.886983  21.84644   78.84675      271.9389
    ## 4    0.02686181      -0.09496978 20.478516  24.98819   83.55821      319.1099
    ## 5    0.00000000      -0.02713422  8.235679  14.53778   57.56718      126.6668
    ## 6    0.08058544      -0.04070133 16.751453  23.76861   71.57675      285.6584
    ##   Total.PrgDist Short.Cmp Short.Att Short.Cmp. Medium.Cmp Medium.Att
    ## 1      89.66457  3.666858  4.291140   84.93101   6.000839   6.918755
    ## 2      74.60537  4.107450  4.424949   93.34546   4.889736   5.527176
    ## 3     113.93721  9.385073 10.284893   92.16557   6.437344   8.058109
    ## 4     117.31710 12.218124 13.154881   93.63525   6.507890   7.888511
    ## 5      61.34581  5.192347  6.547449   81.12226   2.447072   4.344334
    ## 6     130.00794 10.252044 10.921643   94.77374   4.726598   7.097051
    ##   Medium.Cmp.  Long.Cmp Long.Att Long.Cmp.        Ast          xA        A.xA
    ## 1    88.78584 2.4567817 4.511112  57.06263 0.00000000 0.004634908  0.06003684
    ## 2    89.02301 1.7983478 2.996522  63.81866 0.04670873 0.037079264 -0.08004911
    ## 3    80.39192 1.1646051 2.584194  47.58732 0.00000000 0.064888713 -0.11006753
    ## 4    83.87735 1.5061677 2.767904  57.77934 0.07473396 0.023174540  0.04002456
    ## 5    56.78788 0.7037013 3.098583  22.70279 0.04670873 0.013904724  0.08004911
    ## 6    67.42980 1.6049328 4.168186  40.84183 0.00000000 0.023174540 -0.14008595
    ##           KP      X1.3       PPA      CrsPA      Prog Tackles.Tkl Tackles.TklW
    ## 1 0.00000000 0.8346223 0.0446409 0.03263753 0.8495328    1.275241    0.8416557
    ## 2 0.01775681 0.7166390 0.0000000 0.00000000 0.4324894    0.985825    0.5520537
    ## 3 0.26191295 1.8527740 0.3368359 0.12122510 1.6488659    1.962606    1.3484591
    ## 4 0.22196013 1.1273955 0.2353793 0.13987512 1.3746985    3.771459    2.2987156
    ## 5 0.00000000 0.2403363 0.1947967 0.06993756 0.4402124    3.554396    2.2987156
    ## 6 0.26635215 2.0800010 0.5722152 0.45226289 2.2242312    2.469085    1.1312577
    ##   Tackles.Def.3rd Tackles.Mid.3rd Tackles.Att.3rd Vs.Dribbles.Tkl
    ## 1       1.0082365       0.2393269      0.06424947       0.5370829
    ## 2       0.7986032       0.3218534      0.00000000       0.5663783
    ## 3       1.3376604       0.6024436      0.20192691       0.6444994
    ## 4       2.6054429       0.9985709      0.24781939       1.0351051
    ## 5       3.2742731       0.4869065      0.00000000       1.4354760
    ## 6       1.7369620       0.6272015      0.30289036       1.2499383
    ##   Vs.Dribbles.Att Vs.Dribbles.Tkl. Vs.Dribbles.Past Pressures.Press
    ## 1        1.658061         25.90585        1.2585628        8.863509
    ## 2        0.905166         52.69765        0.4620041        7.976150
    ## 3        1.954143         29.88705        1.2824596       14.157414
    ## 4        2.385578         41.18020        1.4258402       21.558798
    ## 5        3.950584         34.05890        2.5410225       23.162095
    ## 6        1.742656         74.77930        0.4779353       12.060020
    ##   Pressures.Succ Pressures.. Pressures.Def.3rd Pressures.Mid.3rd
    ## 1       2.931585    34.29563          7.778336          1.570610
    ## 2       2.714430    34.33543          5.529045          2.394222
    ## 3       4.086451    28.75381          8.897441          3.658372
    ## 4       6.257996    28.96275         13.650868          7.383782
    ## 5       6.761399    30.25617         17.938925          5.736557
    ## 6       1.905037    16.64537          7.024879          3.122066
    ##   Pressures.Att.3rd Blocks.Blocks Blocks.Sh Blocks.ShSv Blocks.Pass       Int
    ## 1         0.1671108      1.951328 0.6465002  0.00000000   1.3340310 1.6713662
    ## 2         0.3243916      1.461140 0.7802588  0.00000000   0.8406223 1.7020335
    ## 3         2.1527809      2.281263 0.4904484  0.06305471   1.8091653 1.9780389
    ## 4         1.2680764      2.705465 0.2452242  0.00000000   2.2020649 1.8093689
    ## 5         0.7864040      2.149289 0.1783449  0.03152736   1.8548513 0.7053472
    ## 6         2.3395518      2.582918 1.1480951  0.07881839   1.5533238 1.9473716
    ##    Tkl.Int      Clr        Err League Year   Salary X2020.League.Indicator
    ## 1 2.826895 6.657582 0.03496592      A 2020 25480000                      1
    ## 2 2.316945 5.731632 0.13986366      A 2020  9950000                      1
    ## 3 3.946567 2.731553 0.01748296      A 2020 26500000                      1
    ## 4 5.964194 2.620439 0.01748296      A 2020 25530000                      1
    ## 5 4.656062 3.203788 0.00000000      A 2020  9510000                      1
    ## 6 4.401087 4.870498 0.00000000      A 2020 18120000                      1
    ##   X2021.League.Indicator Nation.Rank df_l_Pred
    ## 1                      1           0 17.432655
    ## 2                      1           0  8.849696
    ## 3                      0           0 20.509872
    ## 4                      1           0 22.047665
    ## 5                      0           0 18.079144
    ## 6                      1           0 15.700797

### Same code for predicting League midfielders

``` r
mf_l_y_test <- as.integer(mf_l_data$Nation.Rank) 
mf_l_X_test <- mf_l_data %>% select(-Nation.Rank, -Squad, -Salary)
#remove characters for Xgb
mf_l_X_test=mf_l_X_test[,-which(sapply(mf_l_X_test, class)=="character")]
mf_l_xgb_test <- xgb.DMatrix(data = as.matrix(mf_l_X_test), label = mf_l_y_test)
colnames(mf_l_xgb_test) <- NULL 

mf_y_both <- as.integer(mf_data$Nation.Rank) - 1
mf_X_both <- mf_data %>% select(-Nation.Rank)
mf_X_both=mf_X_both[,-which(sapply(mf_X_both, class)=="character")]
mf_xgb_both <- xgb.DMatrix(data = as.matrix(mf_X_both), label = mf_y_both)

mf_xgb_model <- xgb.train(data = mf_xgb_both, nrounds = 53, verbose = 1) #set parameter of 53 boosts
mf_l_xgb_preds <- predict(mf_xgb_model, mf_l_xgb_test, reshape = TRUE)

mf_l_data$mf_l_Pred <-  mf_l_xgb_preds
```

### Predict League forwards

``` r
fw_l_y_test <- as.integer(fw_l_data$Nation.Rank) 
fw_l_X_test <- fw_l_data %>% select(-Nation.Rank, -Squad, -Salary)
#remove characters for Xgb
fw_l_X_test=fw_l_X_test[,-which(sapply(fw_l_X_test, class)=="character")]
fw_l_xgb_test <- xgb.DMatrix(data = as.matrix(fw_l_X_test), label = fw_l_y_test)
colnames(fw_l_xgb_test) <- NULL 

fw_y_both <- as.integer(fw_data$Nation.Rank) - 1
fw_X_both <- fw_data %>% select(-Nation.Rank)
fw_X_both=fw_X_both[,-which(sapply(fw_X_both, class)=="character")]
fw_xgb_both <- xgb.DMatrix(data = as.matrix(fw_X_both), label = fw_y_both)

fw_xgb_model <- xgb.train(data = fw_xgb_both, nrounds = 23, verbose = 1) 
fw_l_xgb_preds <- predict(fw_xgb_model, fw_l_xgb_test, reshape = TRUE)

fw_l_data$fw_l_Pred <-  fw_l_xgb_preds
```

### Predict League Goalkeepers

``` r
gk_l_y_test <- as.integer(gk_league_data$Nation.Rank) 
gk_l_X_test <- gk_league_data %>% select(-Nation.Rank, -Squad, -Salary)
#remove characters for Xgb
gk_l_X_test=gk_l_X_test[,-which(sapply(gk_l_X_test, class)=="character")]
extra_columns_start=match("X90s",names(gk_l_X_test))
extra_columns_end=match("Err",names(gk_l_X_test))
gk_l_X_test=gk_l_X_test[,-c(extra_columns_start:extra_columns_end)]
gk_l_xgb_test <- xgb.DMatrix(data = as.matrix(gk_l_X_test), label = gk_l_y_test)
colnames(gk_l_xgb_test) <- NULL 

gk_y_both <- as.integer(gk_data$Nation.Rank) - 1
gk_X_both <- gk_data %>% select(-Nation.Rank)
gk_X_both=gk_X_both[,-which(sapply(gk_X_both, class)=="character")]
gk_xgb_both <- xgb.DMatrix(data = as.matrix(gk_X_both), label = gk_y_both)

gk_xgb_model <- xgb.train(data = gk_xgb_both, nrounds = 14, verbose = 1) 
gk_l_xgb_preds <- predict(gk_xgb_model, gk_l_xgb_test, reshape = TRUE)

gk_league_data$gk_l_Pred <-  gk_l_xgb_preds
```

### Conclusion

Now that we have the predicted quality of every player in the league
dataset, we can analyse the output of other nations as well as pick the
best Rarita team. Further steps were done in Excel.
