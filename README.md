# Actuarial Theory and Practice A

_"Tell me and I forget. Teach me and I remember. Involve me and I learn" - Benjamin Franklin_

---

### Congrats on completing the [2022 SOA Research Challenge](https://www.soa.org/research/opportunities/2022-student-research-case-study-challenge/)!

>Now it's time to build your own website to showcase your work.  
>To create a website on GitHub Pages to showcase your work is very easy.

This is written in markdown language. 
>
* Click [4001 link](https://classroom.github.com/a/ggiq0YzO) to accept your group assignment.
* Click [5100 link](https://classroom.github.com/a/uVytCqDv) to accept your group assignment 

#### Follow the [guide doc](Doc1.pdf) to submit your work. 
---
>Be creative! Feel free to link to embed your [data](player_data_salaries_2020.csv), [code](sample-data-clean.ipynb), [image](ACC.png) here

More information on GitHub Pages can be found [here](https://pages.github.com/)
![](Actuarial.gif)

# Team "Park the Bus" SOA Challenge Showcase 
>By: Nicholas Ngyugen, Nathan Ng, Sean Stephen, William Li and Brittanie Hsu
---

## Executive Summary 

![](highlights.gif)

Rarita’s football players has found moderate success in the international scene of football however not when
representing Rarita as a nation. After recognising competitive sports team generally have benefits for the country’s
economy and global visibility, Rarita had explored ideas for creating a national team to participate in the
international Football and Sporting Association (FSA) League.

The Rarita national team had been developed with the objective for a high probability for winning the Football and
Sporting Association (FSA) League and to have a positive impact on Rarita’s economy. The gradient boosting
model was used for identifying player effectiveness after comparing several after models. A team of the best 24
players were selected using the model, which can be accessed [here](player_data_salaries_2020.csv) (wrong file, update later)

This combination of players had a high probability of 89.33% to be in the top 10 football nations in the next 5 years
and 41.65% for winning the championship within 10 years. Moreover, the team was found to a positive impact on
Rarita’s economy with an expected NPV of $18.742 billion and statistical evidence of increased GDP growth with a
national football team.

## Data Cleaning 
The data provided were useful for preliminary analysis of player statistics, but were limited by size and quality. There were only 488 players in the tournament dataset where most of the analysis was conducted. When analysing the effects of different attributes on player effectiveness, nation rank was used as the proxy for the objective measure of a players’ effectiveness. It was found that whilst tournament data contained national ranks, the larger pool of teams in the league did not have a similar measure for performance. Therefore, the analysis was conducted using the smaller league dataset which resulted in a low sample count that reduced the effectiveness of the model. Moreover, the data provided only had two years’ worth. Increasing the number of players or increasing the number of years in the dataset could be useful for improving the validity of the model. 

Missing data was prevalent throughout the given dataset with that reduced the quality of the data. The players were analysed separately by position (DF, MF, FW or GK) and any player with multiple player positions were analysed in both respective datasets. The missing data was imputed by substituting the average value of the specific position. However, the majority of goalkeepers were missing shooting, passing and defence data in the tournament dataset, making imputation ineffective. Hence the GK model was limited on just goalkeeping attributes to determine player effectiveness.

Dirty data, such as values below zero or above 100 as a percentage, were bounded between sensible minimum and maximum values. Values below zero were set as exactly zero while values above 100 were set as 100. 

A major assumption when selecting players is that the players are independent from one another. Teamwork is very important in reality, but it was not considered in our model because there is insufficient data to separate the effects of good team chemistry and good individual attributes on successful team performance. Moreover, Rarita can build teamwork over time when the players train and play together in the national team, so individual attributes were deemed more important.

It must also be noted that team selection is a dynamic process and the players selected in this report are only for the current year. In the future, the model must be updated with the latest data and players selected accordingly. As Rarita plays more games in the tournament, performance in the tournament itself should be a major factor in determining if a player should be retained, probably more so than implied by league data.

## Modelling

To analyse the effectiveness of different players, we implemented a gradient boosting model using the tournament dataset with player stats as features and the rank of their nation in the 2021 tournament as the target.  

Gradient boosting was chosen for several reasons: 

* High predictive power for most tasks 

* Handles datasets with many features and relatively few training examples well, as 

* It does feature selection automatically, as features with little predictive power will simply be chosen less often as the splitting criterion. 

* Deals with multicollinearity better than most models, as many features provided by data are interaction terms. 

The package XGBoost was applied in R, automatically regularising trees built in the later iterations in order to stop overfitting.

Nation rank was chosen as the target variable as it is most directly related to our objective. 4 separate models were trained for defenders, midfielders, forwards, and goalkeepers, with players who can play multiple positions being considered for each separately. The model could be used to pick the strongest players as well as make predictions for nations. To predict the rank of a nation, we computed the average model output amongst their players in each of the 4 positions respectively, then took the overall average of those 4 position averages.

Code for modelling can be accessed [here](SOA-GBM-Model.md).

## Team Selection 

## Economic Impact

To determine the net present value, revenues, expenses and profits of the firm, interest and inflation rates, extensive modelling was undertaken using time series models for interest and inflation rates (ARIMA models), forecasting of different groups of revenues/expenses individually and through the creation of an exponential externality factor.

EconSlide1.JPG


An example of 1000 simulations of net present value, reveneues, expenses, profits and financial rates [here](ThousandSimulations.csv)
## Implementation Plan
### Key Plan Components
<img width="500" alt="components" src="https://user-images.githubusercontent.com/103094467/162359679-9f84cc73-04bd-48f7-afcd-c3773b51a216.png">

### Implementation Timeline
<img width="750" alt="timeline" src="https://user-images.githubusercontent.com/103094467/162359744-f80414e4-a78f-4904-b9f2-6dceb945c86e.png">

### Revenue, Expense and Profit Projections
<img width="500" alt="table" src="https://user-images.githubusercontent.com/103094467/162359764-f27abe66-a7a8-4b65-b4a5-6ce4f93d333c.png">
<img width="500" alt="graph" src="https://user-images.githubusercontent.com/103094467/162359758-9eb05b3f-4a62-442e-8446-431bedd5727f.png">

* Average revenues and expenses were determined with 10 000 simulations of Rarita rankings and their projected financial impacts 

## Risks and Mitigation
### Financial Risks 
![](risks1.png)

### Qualitative Risks 
![](risks2.png)
