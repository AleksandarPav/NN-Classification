import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import random


def main():
    # the goal is to predict whether the borrower will default, or return the borrowed money
    data_info = pd.read_csv('lending_club_info.csv', index_col = 'LoanStatNew')
    df = pd.read_csv('lending_club_loan_two.csv')
    print('Info about dataframe:\n', df.info(), '\n')
    print('Head of the dataframe:\n', df.head(), '\n')

    plt.figure(figsize = (10, 6))
    sns.countplot(x = 'loan_status', data = df) # how many defaults vs. how many paid off; unbalanced data

    plt.figure(figsize = (10, 6))
    df['loan_amnt'].hist(bins = 50) # histogram of loan amount
    plt.xlabel('loan_amnt')

    print('Correlation for the dataframe:\n', df.corr(), '\n')
    plt.figure(figsize = (15, 9))
    sns.heatmap(df.corr(), cmap = 'coolwarm', annot = True) # correlation between numeric features; strong correlation
    # between installment and loan amount

    print('Loan amount description: ', data_info.loc['loan_amnt']['Description'], '\n')
    print('Installment description: ', data_info.loc['installment']['Description'], '\n')
    plt.figure(figsize = (10, 6))
    plt.scatter(x = 'loan_amnt', y = 'installment', data = df)
    plt.xlabel('loan_amnt')
    plt.ylabel('installment')

    plt.figure(figsize = (10, 6))
    sns.boxplot(x = 'loan_status', y = 'loan_amnt', data = df) # distribution of loan amount based on whether the
    # borrower returned the money or not

    print('Loan amount info, separated by the loan status:\n',
          df.groupby('loan_status')['loan_amnt'].describe().transpose(), '\n') # statistic descriptions of loan amount,
    # separated by the loan status

    print('Unique values of grade feature:\n', df['grade'].unique(), '\n') # A B C D E F G
    print('Unique values of sub_grade feature:\n', df['sub_grade'].unique(), '\n') # all the letters in combination with
    # numbers 1-5 (A1, B3...)

    plt.figure(figsize = (10, 6))
    sns.countplot(x = 'grade', data = df, hue = 'loan_status') # as the letter goes away from 'A', charged_off/paid
    # ratio becomes bigger and bigger => (sub)grade is probably very important feature

    plt.figure(figsize = (10, 6))
    subgrad_sorted = sorted(df['sub_grade'].unique())
    sns.countplot(x = 'sub_grade', data = df, order = subgrad_sorted)
    plt.figure(figsize = (15, 9))
    sns.countplot(x = 'sub_grade', data = df, order = subgrad_sorted, hue = 'loan_status') # the worse the grade
    # (subgrade), the bigger default/paid ratio

    plt.figure(figsize = (10, 6))
    FandG = df[(df['grade'] == 'F') | (df['grade'] == 'G')]
    sns.countplot(x = 'sub_grade', data = FandG, order = sorted(FandG['sub_grade'].unique()), hue = 'loan_status') # F
    # and G grades isolated

    df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0}) # mapping 'Fully Paid' to 1 and
    # 'Charged Off' to 0
    print("'loan_repaid' and 'loan_status' columns:\n", df[['loan_repaid', 'loan_status']], '\n')

    plt.figure(figsize = (15, 9))
    df.corr()['loan_repaid'][:-1].sort_values().plot(kind = 'bar') # correlation of numeric variables with loan_repaid;
    # interest rate has highest correlation

    print('Length of the dataframe: ', len(df), '\n') # length of the dataframe

    print('Number of missing values per column:\n', df.isnull().sum(), '\n') # number of missing values per column

    print('Percentage of missing values per column:\n', df.isnull().sum()/len(df)*100.0, '\n') # emp_title, emp_length,
    # title, revol_util, mort_acc, pub_rec_bankruptcies have missing data

    feat_info(data_info, 'emp_title') # info on employment title
    feat_info(data_info, 'emp_length') # info on length of the employment

    print('\nNumber of unique job titles: ', df['emp_title'].nunique(), '\n')
    print('Number of each unique job title:\n', df['emp_title'].value_counts(), '\n')
    df = df.drop('emp_title', axis = 1) # too many unique job titles to be converted into dummy variables

    plt.figure(figsize = (15, 9))
    emp_length_order = sorted(df['emp_length'].dropna().unique()) # sorted order of employment length
    sns.countplot(x = 'emp_length', data = df, order = emp_length_order) # this order sorts like: 1, 10+, 2, 3, ... , <1
    emp_length_order = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years',
                        '8 years', '9 years', '10+ years'] # if you want something done right...
    plt.figure(figsize = (15, 9))
    sns.countplot(x = 'emp_length', data = df, order = emp_length_order) # counts how many borrowers per each employment
    # length
    plt.figure(figsize = (15, 9))
    sns.countplot(x = 'emp_length', data = df, order = emp_length_order, hue = 'loan_status') # counts how many
    # borrowers per each employment length, separated by whether they defaulted or not

    # what is the percentage of people that charged off per each emp_length category?
    num_co = df[df['loan_status'] == 'Charged Off'].groupby('emp_length')['loan_status'].count()
    num_fp = df[df['loan_status'] == 'Fully Paid'].groupby('emp_length')['loan_status'].count()
    print('Percentage of charged off borrowers per employment length:\n', 100.0 * num_co / (num_co + num_fp), '\n')
    plt.figure(figsize = (15, 9))
    (num_co / num_fp).plot(kind = 'bar') # rates of charge offs are very similar in each category, so:
    df = df.drop('emp_length', axis = 1)

    print('Amount of missing data:\n', df.isnull().sum(), '\n') # title, revol_util, mort_acc, pub_rec_bankruptcies have
    # missing data
    print("'title' feature info: ")
    feat_info(data_info, 'title')
    print('\nFirst ten rows of "title" and "purpose" columns:\n', df[['title', 'purpose']].head(10), '\n') # same
    # information in both columns, so:
    df = df.drop('title', axis = 1)

    print("'mort_acc' feature info: ")
    feat_info(data_info, 'mort_acc') # info on 'mort_acc' feature
    print('\nValue counts of "mort_acc" column:\n', df['mort_acc'].value_counts(), '\n')
    # checking which column mostly correlates with 'mort_acc':
    print('Correlations with "mort_acc" column:\n', df.corr()['mort_acc'].sort_values(ascending = False), '\n') # mostly
    # correlated with 'mort_acc' is the 'total_acc' column
    print('Number of each unique value in "total_acc":\n', df['total_acc'].value_counts(), '\n')
    mort_avg_by_total_acc = df.groupby('total_acc')['mort_acc'].mean()
    print('Average "mort_acc" value for people having a common value of "total_acc":\n', mort_avg_by_total_acc, '\n')
    # filling missing 'mort_acc' values with means of 'mort_acc' values that share the same 'total_acc' value:
    df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['mort_acc'], mort_avg_by_total_acc, x['total_acc']), axis = 1)

    print('Number of missing values per column:\n', df.isnull().sum(), '\n') # no missing values in 'mort_acc' column;
    # 'revol_util' and 'pub_rec_bankruptcies' have 276 + 535 = 811 missing values, which is ~0.2 % of all the data, so
    # it is safe to remove those rows with NaN values
    df = df.dropna()
    print('Number of missing values per column:\n', df.isnull().sum(), '\n') # no missing values

    print('Columns that are not numerical:\n', df.select_dtypes(include = ['object']).columns, '\n') # term, grade,
    # sub_grade, home_ownership, verification_status, issue_d, loan_status, purpose, earliest_cr_line,
    # initial_list_status, application_type, address

    print('Number of each unique value of the "term" column:\n', df['term'].value_counts(), '\n')
    df['term'] = df['term'].map({' 36 months': 36, ' 60 months': 60}) # converting object type to integer
    print('First 10 values in the "term" column:\n', df['term'].head(10), '\n') # now 36 months are replaced with int
    # 36, and 60 months are replaced with 60

    # since 'grade' information column is already contained in the 'sub_grade' column, it can be dropped:
    df = df.drop('grade', axis = 1)

    # converting 'sub_grade' categorical data to dummy variables
    sub_grade_dummies = pd.get_dummies(df['sub_grade'], drop_first = True)
    df = pd.concat([df.drop('sub_grade', axis = 1), sub_grade_dummies], axis = 1)
    print('Columns of the new dataframe are:\n', df.columns, '\n')

    print('Columns that are not numerical:\n', df.select_dtypes(include = ['object']).columns, '\n') # home_ownership,
    # verification_status, issue_d, loan_status, purpose, earliest_cr_line, initial_list_status, application_type,
    # address

    # converting 'verification_status', 'application_type', 'initial_list_status' and 'purpose' to dummy variables
    dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']],
                             drop_first = True)
    df = pd.concat([df.drop(['verification_status', 'application_type', 'initial_list_status', 'purpose'], axis = 1),
                    dummies], axis = 1)

    print('Columns that are not numerical:\n', df.select_dtypes(include = ['object']).columns, '\n') # home_ownership,
    # issue_d, loan_status, earliest_cr_line, address

    print('Number of each unique value of "home_ownership" column:\n', df['home_ownership'].value_counts(), '\n')

    df['home_ownership'] = df['home_ownership'].replace(to_replace = ['ANY', 'NONE'], value = 'OTHER') # replacing
    # values 'ANY' and 'NONE' with value 'OTHER'
    home_dummies = pd.get_dummies(df['home_ownership'], drop_first = True)
    df = pd.concat([df.drop('home_ownership', axis = 1), home_dummies], axis = 1)

    print('Columns that are not numerical:\n', df.select_dtypes(include = ['object']).columns, '\n') # issue_d,
    # loan_status, earliest_cr_line, address

    print('First 10 values of the "address" column:\n', df['address'].head(10), '\n')
    df['zip_code'] = df['address'].apply(lambda x: x[-5:]) # extracting the zip code from the addresses
    zip_dummies = pd.get_dummies(df['zip_code'], drop_first = True)
    df = pd.concat([df.drop(['zip_code', 'address'], axis = 1), zip_dummies], axis = 1)

    print('Columns that are not numerical:\n', df.select_dtypes(include = ['object']).columns, '\n') # issue_d,
    # loan_status, earliest_cr_line

    df = df.drop('issue_d', axis = 1) # dropping because the model should predict (not) giving a loan, and 'issue_d'
    # feature tells the issue date of the loan; the model shouldn't know that

    print('First 10 values of the "earliest_cr_line" column:\n', df['earliest_cr_line'].head(10), '\n')
    df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:])) # extracting the year
    df = df.drop('earliest_cr_line', axis = 1)
    print('First 10 values of the "earliest_cr_year" column:\n', df['earliest_cr_year'].head(10), '\n')

    print('Columns that are not numerical:\n', df.select_dtypes(include = ['object']).columns, '\n') # only loan_status

    df = df.drop('loan_status', axis = 1)
    print('Columns in the dataframe are:\n', df.columns, '\n')

    X = df.drop('loan_repaid', axis = 1).values
    print('Shape of the training set: ', X.shape, '\n') # 78 features
    y = df['loan_repaid'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train) # fitting parameters and transforming the training data
    X_test = scaler.transform(X_test) # not fitting the test data, just transforming based on parameters fitted on train
    # set

    model = Sequential()
    model.add(Dense(units = 78, activation = 'relu'))
    model.add(Dropout(rate = 0.5)) # 50 % chance of dropping out a neuron
    model.add(Dense(units = 39, activation = 'relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units = 19, activation = 'relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units = 9, activation = 'relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units = 1, activation = 'sigmoid')) # the last layer
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 25) # stop if validation loss
    # starts increasing (overfitting); because of potential noisy shape of the loss, 25 epochs is duration of waiting
    # for the function to stabilize
    model.fit(x = X_train, y = y_train, batch_size = 256, epochs = 500, validation_data = (X_test, y_test),
              callbacks = [early_stop])

    model_loss = pd.DataFrame(model.history.history) # dataframe with train and validation losses
    model_loss.plot()

    model.save('model_78_39_19_9_1.h5')

    predictions = (model.predict(X_test) > 0.5).astype("int32")

    print('Classification report:\n', classification_report(y_test, predictions), '\n')
    print('Confusion matrix:\n', confusion_matrix(y_test, predictions), '\n')

    # predicting random customer
    random.seed(101)
    random_idx = random.randint(0, len(df))
    new_customer = (df.drop('loan_repaid', axis = 1).iloc[random_idx])
    print('New customer is:\n', new_customer, '\n')
    new_customer = scaler.transform(new_customer.values.reshape(1, 78)) # training data is scaled and has 78 columns
    new_prediction = (model.predict(new_customer) > 0.5).astype("int32")

    print('Predicted: ', new_prediction, '\n', 'Issued: ', df.iloc[random_idx]['loan_repaid'])

    plt.show()


def feat_info(data_info, col_name):
    print(data_info.loc[col_name]['Description'])


def fill_mort_acc(mort_acc, mort_avg_by_total_acc, total_acc):
    """
    Receives:
        'mort_acc' value,
        series with averages of 'mort_acc' values per 'total_acc',
        'total_acc' value.
    If the 'mort_acc' is NaN, it is filled with the mean of 'mort_acc' values with the same 'total_acc' values.
    """
    if np.isnan(mort_acc):
        return mort_avg_by_total_acc[total_acc]
    else:
        return mort_acc


if __name__ == '__main__':
    main()