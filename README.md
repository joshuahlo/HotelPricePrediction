# Hotel Price Prediction

## Description
This project predicts hotel reservation cancellations using a dataset containing various booking and customer features. Multiple machine learning models, including logistic regression, decision trees, random forests, and Lasso regression, were implemented and compared for their performance. Feature engineering, clustering, and hyperparameter tuning were applied to optimize the models. The project also incorporates a cost-benefit analysis to assess the profitability of cancellations.

## Languages and Libraries Used
- **R**
  - `tree`
  - `partykit`
  - `randomForest`
  - `readr`
  - `mclust`
  - `glmnet`
  - `dplyr`
  - `ggplot2`
  - `GGally`

## Environments Used
- **RStudio**

## Program Walk-Through

### 1. Dataset Preparation
- Imported data from `hotels-4.csv`.
- Cleaned the dataset by:
  - Removing duplicates.
  - Handling missing values.
  - Converting categorical variables to numerical formats.
  - Selecting relevant features based on business understanding.

### 2. Data Exploration
#### Visualized relationships between cancellation rates and key features (e.g., lead time).
<img width="611" alt="image" src="https://github.com/user-attachments/assets/4cda42ba-54b0-49db-8a6e-450774863e24" />

#### Generated correlation plots to identify significant variable interactions.
Example: 

<img width="756" alt="image" src="https://github.com/user-attachments/assets/4d8fd9c1-b229-4ce2-9b9b-e7ae63cadf92" />


#### Analyzed cancellation patterns over time and by hotel type (e.g., Resort Hotel vs. City Hotel).
<img width="856" alt="image" src="https://github.com/user-attachments/assets/bbe37567-f3a3-4ad2-9793-7ec4f322572b" />

### 3. Unsupervised Learning
#### Applied K-means clustering to identify patterns in customer behavior.
#### Determined the optimal number of clusters using AIC and HDIC. 
<img width="618" alt="image" src="https://github.com/user-attachments/assets/4fe3e068-a7e3-432f-8153-a1b59eb525e1" />

We did k-means clustering to find the optimal number of clusters in our analysis. AIC measures the goodness of fit but penalizes the number of parameters. BIC does something similar to AIC but will additionally prevent overfitting. However, this applies more of a penalty when we want more clusters; hence we do not use BIC in this case. Finally, HDIC applies a penalty that grows logarithmically with the number of parameters. We want to minimize IC but also not have too many clusters as that may overfit the data. We use HDIC in this case as a balance between BIC and AIC. 

TLDR; optimal number of clusters is 23


<img width="1283" alt="image" src="https://github.com/user-attachments/assets/b2eb17c3-0040-4062-9e40-4e5a1c618c55" />
We find that cluster 16 has a mean cancellation time of 0.94, and by far the highest mean lead time compared to every other cluster. This could indicate that almost all the reservations were cancelled in this cluster. This could indicate a correlation between the two, and we keep this fact in mind when creating our models.


### 4. Model Development
- **Logistic Regression**:
  - Built a logistic regression model with interaction terms.
- **Lasso Regression**:
  - Applied Lasso regression with cross-validation to select the best features.
- **Decision Tree**:
  - Developed a classification tree to predict cancellations.
- **Random Forest**:
  - Trained a random forest model with optimized hyperparameters (e.g., number of trees, node size).
 

### Tested Out of Sample Accuracy
<img width="417" alt="image" src="https://github.com/user-attachments/assets/ffa52268-3a95-446c-8f10-797e910c9892" />

### Tested MAE
<img width="417" alt="image" src="https://github.com/user-attachments/assets/eca7bcbf-d0db-4b6a-a128-9962b30b3615" />

#### We then choose random forest. After choosing random forest, conduct feature importance.
<img width="599" alt="image" src="https://github.com/user-attachments/assets/4675e410-9308-4d76-af7e-ded9b4780a37" />

##### Prediction for random forest: 
- This graph shows the probability for each prediction and its counts. There are many values that are predicted to be 0, the rest are somewhere in the middle, with about a handful of predictions around towards the end predicted to be 1.

<img width="1162" alt="image" src="https://github.com/user-attachments/assets/983c2b2c-9e75-4c83-aa33-c25f4a96817a" />



### 5. Model Evaluation
- Used 10-fold cross-validation to evaluate model performance.
- Metrics:
  - Mean Absolute Error (MAE)
  - Accuracy
- Compared model predictions using ROC and cumulative gain curves.

### 6. Cost-Benefit Analysis
- Conducted a profit analysis based on:
  - Average revenue per reservation.
  - Costs associated with cancellations.
- Developed a cost-benefit matrix to optimize decision-making.
- Identified a profit of **$78.17 per reservation** with the best model.

## Results
- Logistic regression with Post-Lasso demonstrated the best trade-off between accuracy and interpretability.
- Random forest provided superior predictive performance but required more computational resources.
- The model effectively identified patterns in cancellations, aiding in business decision-making.

## Future Work
- Enhance model performance by exploring additional features and larger datasets.
- Incorporate advanced techniques like ensemble learning and neural networks.
- Automate feature engineering and model selection for scalability.
