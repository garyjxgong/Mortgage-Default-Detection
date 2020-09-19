import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, plot_confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, learning_curve


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
def read_file(file):
    return pd.read_csv(f'data/{file}')

def load_application():
    application = read_file('application.csv')
    application_label = application['TARGET']
    application_features = application.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    application_features['DAYS_EMPLOYED_ANOM'] = application_features["DAYS_EMPLOYED"] == 365243
    application_features['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    return application, application_features, application_label

def load_application_new():
    application = read_file('application.csv')
    application_label = application['TARGET']
    application_features = application.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    application_features['DAYS_EMPLOYED_ANOM'] = application_features["DAYS_EMPLOYED"] == 365243
    application_features['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)
    application_features['CREDIT_INCOME_PERCENT'] = application_features['AMT_CREDIT'] / application_features['AMT_INCOME_TOTAL']
    application_features['ANNUITY_INCOME_PERCENT'] = application_features['AMT_ANNUITY'] / application_features['AMT_INCOME_TOTAL']
    application_features['CREDIT_TERM'] = application_features['AMT_ANNUITY'] / application_features['AMT_CREDIT']
    application_features['PRICE_CREDIT_PERCENT'] = application_features['AMT_GOODS_PRICE'] / application_features['AMT_CREDIT']
    application_features['DAYS_EMPLOYED_PERCENT'] = application_features['DAYS_EMPLOYED'] / application_features['DAYS_BIRTH']
    application_features['INCOME_PER_PERSON'] = application_features['AMT_INCOME_TOTAL'] / application_features['CNT_FAM_MEMBERS']
    return application, application_features, application_label

def count_plot(df, col):
    _, axes = plt.subplots(1, 2, figsize=(10, 8))
    sns.countplot(df.loc[df['TARGET'] == 0, col], ax=axes[0])
    sns.countplot(df.loc[df['TARGET'] == 1, col], ax=axes[1])
    plt.xlabel(col)
    plt.ylabel(f'Count')
    plt.title(f'Count of {col}')
    plt.show()
    
def count_plot(df, col):
    sns.countplot(x='TARGET',hue=col, data=df)
    plt.xlabel('Target')
    plt.ylabel(f'Count')
    plt.title(f'Count of {col} based on Target')
    plt.show()

def df_train_test_split(feature, label, train_size=0.8, random_state=55):
    train_features = feature.sample(frac=train_size, random_state=random_state)
    train_label = label[train_features.index]
    test_features = feature.iloc[~feature.index.isin(train_features.index), :]
    test_label = label[~label.index.isin(train_features.index)]
    return train_features, train_label, test_features, test_label

def prepare_training(feature, label, encoding='one_hot', train_size=0.8, random_state=55):
    print('Spliting Train and Test...')
    train_features, train_label, test_features, test_label = df_train_test_split(feature, label)
    print('\tTraining Features shape: ', train_features.shape)
    print('\tTesting Features shape: ', test_features.shape)
    
    if encoding == 'one_hot':
        print('One Hot Encoding...')
        train_features = pd.get_dummies(train_features, drop_first=True)
        test_features = pd.get_dummies(test_features, drop_first=True)
    
    
    print('\tTraining Features shape: ', train_features.shape)
    print('\tTesting Features shape: ', test_features.shape)
    
    print('Aligning Train and Test...')
    train_features, test_features = train_features.align(test_features, join='inner', axis=1)
    print('\tTraining Features shape: ', train_features.shape)
    print('\tTesting Features shape: ', test_features.shape)
    
    print('\nTraining Prepared.')
    return train_features.to_numpy(), train_label.to_numpy(), test_features.to_numpy(), test_label.to_numpy(), train_features.columns
    
def evaluate_on_test(model, X_test, y_test):
    print(f"[{model['model']}]\n")
    y_predict = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_predict)
    cr = classification_report(y_test, y_predict, target_names=['Repaid', 'Default'], digits=3)
    print('Out-of-Sample Performance:\n')
    print(cr)
    print(f'roc_auc: {roc_auc}\n')
    print(f'Confusion Matrix:\n')
    plot_confusion_matrix(model, X_test, y_test)
    plt.show()
    
def display_learning_curve(model, X, y, title, axes=None, ylim=None, cv=10, scoring='roc_auc', 
                           train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")
    train_sizes, train_scores, valid_scores, fit_times, _  = \
        learning_curve(model, X, y, cv=cv, train_sizes=train_sizes, 
                       scoring=scoring, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                         valid_scores_mean + valid_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label=f"Training {scoring}")
    axes[0].plot(train_sizes, valid_scores_mean, 'o-', color="g",
                 label=f"Cross-validation {scoring}")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, valid_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, valid_scores_mean - valid_scores_std,
                         valid_scores_mean + valid_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel(scoring)
    axes[2].set_title("Performance of the model")
    
    print(f'Validation ROC: {np.max(valid_scores_mean)}')

    return plt