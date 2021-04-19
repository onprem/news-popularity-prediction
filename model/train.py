import csv
import pickle
import pandas as pd

# Importing the dataset
uci_dataset = pd.read_csv('dataset.csv', quoting = 3, index_col = False)

# Cleaning the columns headers of whitespaces
arr = list(uci_dataset)
cleaned_columns = {x:x.lower().strip() for x in arr}
new_dataset = uci_dataset.rename(columns=cleaned_columns)

# We are removing features which are not most relevant for our model
x = new_dataset.drop(['url','shares', 'timedelta', 'lda_00','lda_01',
                  'lda_02','lda_03','lda_04','num_self_hrefs', 
                  'kw_min_min', 'kw_max_min', 'kw_avg_min',
                  'kw_min_max','kw_max_max','kw_avg_max',
                  'kw_min_avg','kw_max_avg','kw_avg_avg',
                  'self_reference_min_shares','self_reference_max_shares',
                  'self_reference_avg_sharess','rate_positive_words',
                  'rate_negative_words','abs_title_subjectivity',
                  'abs_title_sentiment_polarity'], axis = 1)
y = new_dataset['shares']

# Splitting the new_dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0) 

# Fitting the random forest regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

# Comparison of y_test and y_pred
pred_result = pd.DataFrame(list(y_test), y_pred)
pred_result.reset_index(0, inplace=True)
pred_result.columns = ['Predicted share','Actual shares']

# Save the model
pickle.dump(regressor, open('model.sav', 'wb'))
