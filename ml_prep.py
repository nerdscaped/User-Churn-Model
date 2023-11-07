import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score, brier_score_loss, recall_score
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss, EditedNearestNeighbours, AllKNN # near miss can have different versions
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier
import math  
from xgboost import XGBClassifier
from scipy import stats

# 1 - Data Prep
churn_prediction = pd.read_csv('/Users/mattcadel/Documents/Python/DSML/churn_prediction_hour.csv')

# print(churn_prediction.groupby(['churn_status']).count())
# 54369 vs 3541 non-churn vs churn user (this is an issue) - this is solved using Over-Sampling

# Processing Time/Date Fields for better Insight
churn_prediction['avg_hour_L60D'] = (math.pi * (churn_prediction['avg_hour_L60D'] + 1) / 24).apply(math.cos)
churn_prediction['month_access_date'] = (math.pi * churn_prediction['month_access_date'] / 12).apply(math.cos)

X = churn_prediction[['days_last_access', 'month_access_date','new_user_stat', 'first_brand_ranking_indicator',
       'plays_L60D', 'recs_L60D', 'unq_recs_L60D', 't20_plays_delta',
       'actions_L60D', 'brands_played_L60D', 'subcats_played_L60D', 'avg_hour_L60D',
       'platforms_L60D', 'weeks_accessed_L60D', 'plays_L7D', 'recs_L7D',
       'day_bounce_rate_L7D', 'brands_played_L7D', 'actions_L7D',
       'days_accessed_L7D', 'plays_delta', 'recs_delta', 'actions_delta',
       'day_bounce_rate_delta', 'brands_played_delta', 'subcats_played_delta',
       't20_plays_L60D', 't20_plays_L7D']]
y = churn_prediction['churn_status']

# SMOTE to equalise the split of churn to non-churn users\

functions = [SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE, ClusterCentroids,
              RandomUnderSampler, NearMiss, EditedNearestNeighbours, AllKNN,
              SMOTEENN, SMOTETomek, BalancedBaggingClassifier, RUSBoostClassifier]

sampler = []
algorithm = []

mse_output = []
roc_auc_output = []
brier_output = []
b_accuracy_output = []
recall_output = []

for f in functions:
    print(f"Running Models Using '{f}' resampling\n")
    # if f == SMOTENC:
    #     resampler = f(random_state = 0, categorical_features = 'auto')
    # else:
    #     resampler = f(random_state = 0)

    resampler = f(random_state = 0)
    X_resampled, y_resampled = resampler.fit_resample(X, y)

    # Creating Test/Train Splits
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

    # Standardising Data
    # scaler = StandardScaler()
    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    binary_X_train = X_train[['new_user_stat', 'first_brand_ranking_indicator']]
    binary_X_test = X_test[['new_user_stat', 'first_brand_ranking_indicator']]

    continuous_X_train = X_train[['days_last_access', 'month_access_date',
        'plays_L60D', 'recs_L60D', 'unq_recs_L60D', 't20_plays_delta',
        'actions_L60D', 'brands_played_L60D', 'subcats_played_L60D', 'avg_hour_L60D',
        'platforms_L60D', 'weeks_accessed_L60D', 'plays_L7D', 'recs_L7D',
        'day_bounce_rate_L7D', 'brands_played_L7D', 'actions_L7D',
        'days_accessed_L7D', 'plays_delta', 'recs_delta', 'actions_delta',
        'day_bounce_rate_delta', 'brands_played_delta', 'subcats_played_delta',
        't20_plays_L60D', 't20_plays_L7D']]

    continuous_X_test = X_test[['days_last_access', 'month_access_date',
        'plays_L60D', 'recs_L60D', 'unq_recs_L60D', 't20_plays_delta',
        'actions_L60D', 'brands_played_L60D', 'subcats_played_L60D', 'avg_hour_L60D',
        'platforms_L60D', 'weeks_accessed_L60D', 'plays_L7D', 'recs_L7D',
        'day_bounce_rate_L7D', 'brands_played_L7D', 'actions_L7D',
        'days_accessed_L7D', 'plays_delta', 'recs_delta', 'actions_delta',
        'day_bounce_rate_delta', 'brands_played_delta', 'subcats_played_delta',
        't20_plays_L60D', 't20_plays_L7D']]

    gaussian_features = []
    non_gaussian_features = []

    # Testing for Normality for Standisation
    for i in continuous_X_train.columns:
        data = list(continuous_X_train[i][:5000])
        normality_tests = {'Shapiro-Wilk Test': stats.shapiro(data)}
        
        for test_name, test_result in normality_tests.items():
            p_value = test_result[1]
            alpha = 0.05  # Significance level
            if p_value < alpha:
                non_gaussian_features.append(i)
            else:
                gaussian_features.append(i)

    continuous_X_train_for_transformer = continuous_X_train[non_gaussian_features] # For Non-Gaussian Features
    continuous_X_train_for_scaler = continuous_X_train[gaussian_features] # For Gaussian Features

    continuous_X_test_for_transformer = continuous_X_test[non_gaussian_features] # For Non-Gaussian Features
    continuous_X_test_for_scaler = continuous_X_test[gaussian_features] # For Gaussian Features


    quantile_transformer = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
    quantile_transformer.fit(continuous_X_train_for_transformer)
    continuous_X_train_for_transformer = quantile_transformer.transform(continuous_X_train_for_transformer)
    continuous_X_test_for_transformer = quantile_transformer.transform(continuous_X_test_for_transformer)

    continuous_X_train_for_transformer_df = pd.DataFrame(continuous_X_train_for_transformer, index=continuous_X_train.index, columns=non_gaussian_features)
    continuous_X_test_for_transformer_df = pd.DataFrame(continuous_X_test_for_transformer, index=continuous_X_test.index, columns=non_gaussian_features)

    binary_X_train['id'] = range(len(binary_X_train))
    binary_X_test['id'] = range(len(binary_X_test))

    continuous_X_train_for_transformer_df['id'] = binary_X_train['id']
    continuous_X_test_for_transformer_df['id'] = binary_X_test['id']

    X_train = binary_X_train.merge(continuous_X_train_for_transformer_df, on='id', how='inner')
    X_test = binary_X_test.merge(continuous_X_test_for_transformer_df, on='id', how='inner')

    scaler = StandardScaler()
    try:
        scaler.fit(continuous_X_train_for_scaler)
        continuous_X_train_for_scaler = scaler.transform(continuous_X_train_for_scaler)
        continuous_X_test_for_scaler = scaler.transform(continuous_X_test_for_scaler)

        X_scaled_train_df = pd.DataFrame(continuous_X_train_for_scaler, index=continuous_X_train.index, columns=gaussian_features)
        X_scaled_test_df = pd.DataFrame(continuous_X_test_for_scaler, index=continuous_X_train.index, columns=gaussian_features)

        X_scaled_train_df['id'] = binary_X_train['id']
        X_scaled_test_df['id'] = binary_X_test['id']

        X_train = X_train.merge(X_scaled_train_df, on='id', how='inner')
        X_test = X_test.merge(X_scaled_test_df, on='id', how='inner')

    except:
        print("No Gaussian Features")

    X_train = X_train.drop('id', axis = 1)
    X_test = X_test.drop('id', axis = 1)

    # 5 - LogisticRegression
    print('\nLogisticRegression\n')

    sampler.append(str(f))
    algorithm.append('log')

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions_train = model.predict(X_train)
    roc_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    mse_output.append(mean_squared_error(y_train, predictions_train))
    roc_auc_output.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    brier_output.append(brier_score_loss(y_train, predictions_train)) 
    b_accuracy_output.append(balanced_accuracy_score(y_train, predictions_train)) 
    recall_output.append(recall_score(y_train, predictions_train))

    # coef = model.coef_[0]
    # for n, i in enumerate(coef):
    #     print('Coefficient for feature - ', X.columns[n], '\n =', i)

    # 6 - MLPClassifier
    print('\nMLPClassifier\n')

    sampler.append(str(f))
    algorithm.append('mlp')

    model = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes= (10,5), activation = 'relu', solver = 'adam')
    model.fit(X_train, y_train)

    predictions_train = model.predict(X_train) #  can use predict_proba to output probabilities
    roc_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

    mse_output.append(mean_squared_error(y_train, predictions_train))
    roc_auc_output.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    brier_output.append(brier_score_loss(y_train, predictions_train)) 
    b_accuracy_output.append(balanced_accuracy_score(y_train, predictions_train)) 
    recall_output.append(recall_score(y_train, predictions_train))

    # 7 - XGBoost 
    print('\nXGBoost\n')

    sampler.append(str(f))
    algorithm.append('xgb')

    model = XGBClassifier(objective='binary:logistic', max_depth=5, min_child_weight=1)
    model.fit(X_train, y_train)

    predictions_train = model.predict(X_train) #  can use predict_proba to output probabilities
    
    mse_output.append(mean_squared_error(y_train, predictions_train))
    roc_auc_output.append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    brier_output.append(brier_score_loss(y_train, predictions_train)) 
    b_accuracy_output.append(balanced_accuracy_score(y_train, predictions_train)) 
    recall_output.append(recall_score(y_train, predictions_train))

dict = {'sampler': sampler, 'algorithm': algorithm, 'mse_output': mse_output,
        'roc_auc_output': roc_auc_output, 'brier_output': brier_output, 
        'b_accuracy_output': b_accuracy_output, 'recall_output': recall_output}

dataframe = pd.DataFrame(data=dict)
dataframe.to_csv('/Users/mattcadel/Documents/Python/ml_testing_117.csv')
          
