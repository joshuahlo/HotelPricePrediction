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
- Visualized relationships between cancellation rates and key features (e.g., lead time).
- Generated correlation plots to identify significant variable interactions.
- Analyzed cancellation patterns over time and by hotel type (e.g., Resort Hotel vs. City Hotel).

### 3. Unsupervised Learning
- Applied K-means clustering to identify patterns in customer behavior.
- Determined the optimal number of clusters using AIC and HDIC.

### 4. Model Development
- **Logistic Regression**:
  - Built a logistic regression model with interaction terms.
- **Lasso Regression**:
  - Applied Lasso regression with cross-validation to select the best features.
- **Decision Tree**:
  - Developed a classification tree to predict cancellations.
- **Random Forest**:
  - Trained a random forest model with optimized hyperparameters (e.g., number of trees, node size).

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
