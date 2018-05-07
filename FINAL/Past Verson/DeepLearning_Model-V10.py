
# coding: utf-8

# In[3]:


"""
@Project: Connexin Group 

@FileName: DeepLearning_Model_V10

@Author：Zhejian Peng

@Create date: Mar. 25th, 2018

@description：reduce dimension of our dataset using pca, without normalizing categorical data.

@Update date：Mar. 25th, 2018
            Try to split train and test before normalization.
            1. Need to update normalization for zipcode on V3
            2. Update drop2 to drop more features that might leak information
            
            Update_deep learing model-V and make comparison with logistic Regression
            
            April. 6th, 2018 V3
            1. Deploy Model for validation
            2. Visualize our Deep Learning Model
           i: 
            April. 21, 2018 V5
            1. Use finalized datasets.
            2. Try Overfit the model and finalize the model
            
            April. 22, 2018 V5
            1. Use tuned paramters for final model
@Vindicator：  

"""  


# # I. Select all categorical Variables

# In[4]:


import pandas as pd 
import numpy as np
import math
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import LogisticRegression
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# Read in our data Frame
def readcsv(file_path):
    LARGE_FILE = file_path
    CHUNKSIZE = 100000 # processing 100,000 rows at a time
    # Add encoding encoding = "ISO-8859-1", why?
    reader = pd.read_csv(LARGE_FILE, chunksize=CHUNKSIZE, low_memory=False, encoding = "ISO-8859-1")
    frames = []
    for df in reader:
        frames.append(df)
    loan_data = pd.concat(frames)
    return loan_data   


# In[6]:

print('Start Reading')
FILE_PATH = "loan_data_no_current_converted.csv"
loan_data = readcsv(FILE_PATH)
print('Data Imported')

# In[7]:


df = loan_data.copy()


# In[8]:


# Convert Verification_status_joint, add this categorical data to the categorical list
for idx, i in df["verification_status_joint"].iteritems():
    if i == "Verified":
        df.at[idx, "verification_status_joint"] = 1
    elif i == "Source Verified":
        df.at[idx, "verification_status_joint"] = 2
    elif i == "Not Verified":
        df.at[idx, "verification_status_joint"] = 3


# In[9]:


categorical = ['grade', 'sub_grade', 'emp_length', 'purpose', 'title', 'application_type', 'hardship_flag', 'hardship_type', 'hardship_reason', 
              'hardship_status', 'hardship_loan_status', 'settlement_status', 'disbursement_method', 'home_ownership',
              'pymnt_plan', 'debt_settlement_flag', 'title', 'initial_list_status', 'loan_status', 'verification_status',
              'term', 'verification_status_joint']


# In[10]:


print("There are %d categorical data in our dataset." % len(categorical))


# # Feature Engineering

# In[11]:


# Read in engineered features.
FE_PATH = "This_week_FE.csv"
df_FE = readcsv(FE_PATH)
# temp.replace(float('nan'), -9999999, inplace =True)
# temp = df_FE[df_FE['ratio_rev_acct']!='#DIV/0!']


# In[12]:


df_FE['ratio_rev_acct'].replace('#DIV/0!', float(-np.inf), inplace = True)
df_FE.loc[:,'ratio_rev_acct'] = [float(x) for x in df_FE['ratio_rev_acct']]

df_FE['ratio_rev_acct'].replace(float(-np.inf), np.max(df_FE['ratio_rev_acct']), inplace = True)


# In[13]:


type(df_FE['loan_amt_to_avg_inc'][0])


# In[14]:


# Replace '#DIV/0!' with the max of each col
for i in df_FE.columns[0:3]:
    df_FE[i].replace('#DIV/0!', np.max(df_FE[i]), inplace = True)
    df_FE.loc[:,i] = [float(x) for x in df_FE[i]]

    print(i)

df = pd.concat([df,df_FE],axis=1)
assert(df_FE.shape[0] == df.shape[0])


# In[15]:


# 我写了一堆code 然后发现其实简单一点就能弄出来，所以大家忽略后面的code！！！
# I have wrote a lot of code for this only to find out that I only need this simple function!!!
def norm_inc_by_zip(zipcode, income):
    '''
        @description: Use on a column of data; output a dictionary that returns mean and average in each zipcode area
        @zipcode： zipcode dataframe column
        @income: income df column 
        @return:      return a dictionary
    '''  
    # I try to replace nan with 0 for income, and nan in zipcode for "000xx"
    df["annual_inc"].fillna(0)
    df["zip_code"].fillna("000xx")
    
    mean_var = {}
    for idx, value in zipcode.iteritems():
        # calculate total income
        if value in mean_var:
            mean_var[value].append(income[idx])
        else:
            mean_var[value] = [income[idx]]

    
    #assert(len(zip_code) == len(mean_var))
    # compute the average income in each zip_code area
    for key, value in mean_var.items():
        # if there only one element, we set their variance to 1. This way when normalize, it will have a 0 z-score.
        if len(value) == 1:
            #print(value[0])
            mean_var[key] = [value[0], 1]
        else:
            mean_var[key] = [np.mean(value), np.std(value)]
        
    # first loop through every annual income by calculate its z score. (Income - mean_by_zipcode) / variance_by_zipcode
    for idx, value in df["zip_code"].iteritems():
        #inc_colnum = df.columns.get_loc("annual_inc")
        col_num_inc = df["annual_inc"]
        mean, std = mean_var[value]
        df.at[idx, "annual_inc"] = (df.at[idx, "annual_inc"] - mean) / std
    print("Income is successfually normalized")
    return mean_var


# In[16]:


dic = norm_inc_by_zip(df["zip_code"], df["annual_inc"])
df.drop('zip_code', axis=1,inplace= True)


# # III. Set, X, Y, Train/Test Sets And normalize it accordingly

# In[17]:


def percentage(x):
    x = np.str(x)
    if x[-1] == '%':
        x = x[0:len(x)-1]
    return float(x) / 100


df['revol_util'] = [percentage(x) for x in df['revol_util']]


# In[18]:


# drop the observation that was missing for any field
df.dropna(axis=0, how='all', inplace=True)
df.dropna(axis=1, how='all', inplace=True)
# Use Finalized droplist provide by Yufei Gao
drop_list=['installment','term','settlement_date','pymnt_plan','hardship_length', 'settlement_percentage', 'settlement_term', 'sec_app_earliest_cr_line','policy_code','hardship_end_date','settlement_amount',
           'payment_plan_start_date','hardship_start_date','out_prncp','emp_title','title','earliest_cr_line','desc','issue_d','id','member_id','url','grade','sub_grade',
                   'int_rate','avg_cur_bal','out_prncp_inv','debt_settlement_flag_date','hardship_amount','hardship_reason','addr_state','funded_amnt','funded_amnt_inv','collection_recovery_fee',
                   'collections_12_mths_ex_med','mths_since_last_major_derog','next_pymnt_d','recoveries','total_pymnt',
                   'total_pymnt_inv','total_rec_int','last_pymnt_d','last_credit_pull_d',
                  'total_rec_prncp','settlement_status','hardship_loan_status','hardship_status','debt_settlement_flag',
                   'verification_status','total_rec_late_fee','verification_status_joint','hardship_flag', 'hardship_type', 'hardship_reason'
                    'hardship_status','hardship_loan_status','acc_now_delinq','delinq_amnt','deferral_term','hardship_amount'
                    'hardship_length','hardship_dpd','hardship_payoff_balance_amount','hardship_last_payment_amount']


# Drop drop_updated
df.drop(drop_list, inplace=True, axis=1, errors='ignore')

# Drop all colums where value missed more than 20%
num_rows=df.count(axis=0)
df=df.iloc[:,(num_rows>=0.8*len(df)).tolist()]

# Then fill rest of missing value with mean
df.fillna(df.mean(), inplace=True)
# Drop all rows with 4,5,6
'''for idx, i in loan_data["loan_status"].iteritems():
    if i == "Fully Paid":
        loan_data.at[idx, "loan_status"] = 1
    elif i == "Does not meet the credit policy. Status:Fully Paid":
        loan_data.at[idx, "loan_status"] = 2
    elif i == "Does not meet the credit policy. Status:Charged Off":
        loan_data.at[idx, "loan_status"] = 3
    elif i == "In Grace Period":
        loan_data.at[idx, "loan_status"] = 4
    elif i == "Late (16-30 days)":
        loan_data.at[idx, "loan_status"] = 5
    elif i == "Late (31-120 days)":
        loan_data.at[idx, "loan_status"] = 6
    elif i == "Default":
        loan_data.at[idx, "loan_status"] = 7
    elif i == "Charged Off":
        loan_data.at[idx, "loan_status"] = 8'''


# Let's test our result previous result, Previous result set Y to binary number 1,2.
# 4,5,6 are dropped, and we dont care about ‘Credit policy'
df = df[(df['loan_status']!=4) & (df['loan_status']!=5) & (df['loan_status']!=6)]
print("Input Dataset size is : ",df.shape)


# In[19]:


for i in df['revol_util']:
    if math.isnan(i):
        print(i)


# In[20]:


# Select features that is not in categorical data to normalize: categorial[], verification_status_joint, and annual_inc
# There categorical features are features need to include in X
features= list(df.columns)
features_need_norm = []
categorical_features = []
for i in features:
    if i not in categorical and i != "verification_status_joint":
        features_need_norm.append(i)
    else:
        categorical_features.append(i)



# In[21]:


print("There are %d numerical features need normalization" %len(features_need_norm))


# In[22]:


Y = df.loc[:,['loan_status']].values
features.remove("loan_status")
categorical_features.remove('loan_status')


# In[23]:


# We can not have loan_status in X, we want to check this, if it prints "Warning", we have a problem!!!
for i in categorical_features:
    if i == "loan_status":
        print("Warning!")
for i in features_need_norm:
    if i == "loan_status":
        print("Warning!")


# In[24]:


# Let's test our result previous result, Previous result set Y to binary number 1,2.

Y = list(Y.reshape(len(Y)))
for i in range(len(Y)):
    if Y[i]==7 or Y[i]==8 or Y[i] == 3:
        Y[i] = 1 # Default
    else:
        Y[i] = 0 # Fully Paid
        


# In[25]:


for i in Y:
    if i != 0 and i!=1:
        print(i)
        break


# In[26]:


# I want to find the starting index and ending index of categorical data in X. 
# starting_col_index_of_categorical_data: starting index of categorical data in X
# last_col_index_in_X: ending index of categorical data in X
X = df.loc[:,features_need_norm].values
starting_col_index_of_categorical_data = X.shape[1] 


# In[27]:


# df.loc[:,categorical_features].values.shape = (891823, 16)
X = np.concatenate((X, df.loc[:,categorical_features].values), axis=1)
last_col_index_in_X = X.shape[1]-1
# X.shape = (891823, 109)


# In[28]:


print("Number of categorical featurs in dataset:", len(categorical_features))
print("Number of numerical features in dataset:", len(features_need_norm))
# From column index 93 to column index 107 are 15 categorical data in X
# len(categorical_features) = 15
last_col_index_in_X-starting_col_index_of_categorical_data+1 == len(categorical_features)


# In[29]:


# split train test set 
Y = np.ravel(Y)

x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=1, test_size=0.25)
# x_train, x_test, y_train, y_test = log_reg.split(X,Y,rand=None)


# In[30]:


print("shape of x_train: ", x_train.shape)
print("shape of x_test: ", x_test.shape)
print("shape of y_train:", y_train.shape)
print("shape of y_test:", y_test.shape)


# # III. Normalize Train set and set mean variance in a dictionary:

# In[31]:


def normalize(numerical_features):
    '''
        @description: normalize all numerical features in X, and return a dictionary with mean variance of each features
        @numerical_features： a numpy array of numerical features of shape (668867, 79)
        @return: a dictionary contain mean variance of each features
    '''  
        # Step1: calculate mean variance of each columns in numerical features
    epsilon = 10**-8
    dic = {}
    counter = 0
    for columns in numerical_features.T:
        mean  = np.mean(columns)
        std = np.std(columns)
        dic[counter] = [mean, std]
        counter +=1

    assert(counter == numerical_features.shape[1])

    # Step2: Normalize numerical_features
    for key, val in dic.items():
        try :
        #numerical_features[:,key] = (numerical_features[:,key] - val[0]) / val[1]
            numerical_features[:, key] = (numerical_features[:,key] - val[0]) / val[1]
            assert(np.mean(numerical_features[:, key])- 0.0 < epsilon)
            assert(np.std(numerical_features[:, key])- 1.0 < epsilon)
        except AssertionError:
            print(np.mean(numerical_features[:, key])," | ", np.std(numerical_features[:, key]))
            print(numerical_features[:,key], "Key: ", key)
            print("SUM: ", np.sum(numerical_features[:, key]))
    return dic

            


# In[32]:


a = np.array([[1,2,3,4,5],[6,7,8,9,10]])
normalize(a)


# In[33]:


mv_dic = normalize(x_train[:,0: starting_col_index_of_categorical_data])


# In[34]:


# Normalize test set according to train set's mean variance
for key,val in mv_dic.items():
    x_test[:, key] = (x_test[:,key] - val[0] ) / val[1]
    # print(np.mean(x_test[:, key]))
    # print(np.std(x_test[:, key]))


# In[35]:


counter = 0
for i in y_test:
    if i==1:
        counter +=1
    
print('Train set ratio \n', counter/len(y_test), " | ", (len(y_test)-counter) / len(y_test))
    
counter = 0
for i in y_train:
    if i==1:
        counter +=1
    
print('Train set ratio \n', counter/len(y_train), " | ", (len(y_train)-counter) / len(y_train))
    


# In[36]:


df["loan_status"].value_counts()/ len(df)


# # IV. Train a Deep Learning Model

# # with tuned paramters
# 

# In[38]:


from keras.utils.np_utils import to_categorical
# Use small data set to test for overfitting
np.random.seed(1)
# rand_numbers = np.random.randint(0, len(x_train), int(len(x_train)*0.1))
# X_train = x_train[rand_numbers]
# X_test = x_test
# Y_train = y_train[rand_numbers]
# Y_test = y_test

X_train = x_train
X_test = x_test

Y_train = y_train
Y_test = y_test

print("shape of X_train: ", X_train.shape)
print("shape of X_test: ", X_test.shape)
print("shape of Y_train:", Y_train.shape)
print("shape of Y_test:", Y_test.shape)


# In[39]:


# Use Keras to construct a sequential model and visualize it 
from keras import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
import keras
import keras.backend as tensorflow


# In[43]:


# Best: 0.867080 using {'batch_size': 8, 'dropout_rate': 0.0, 
# 'epochs': 10, 'init_mode': 'normal', 'learn_rate': 0.001}

def create_model(dropout_rate=0.1, init_mode = 'uniform', learn_rate = 0.00001):
    #model_dnn.reset_states()
    model_dnn = Sequential()
    dim = X_train.shape[1]
    model_dnn.add(Dense(57, kernel_initializer=init_mode, activation='relu',   input_dim=dim))
    model_dnn.add(Dropout(dropout_rate))

    model_dnn.add(Dense(64, kernel_initializer=init_mode, activation='relu'))
    model_dnn.add(Dropout(dropout_rate))

    model_dnn.add(Dense(64, kernel_initializer=init_mode, activation='relu'))
    model_dnn.add(Dropout(dropout_rate))
    
    model_dnn.add(Dense(64, kernel_initializer=init_mode, activation='relu'))
    model_dnn.add(Dropout(dropout_rate))
    
    model_dnn.add(Dense(32, kernel_initializer=init_mode, activation='relu'))
    model_dnn.add(Dropout(dropout_rate))
    
    model_dnn.add(Dense(16, kernel_initializer=init_mode, activation='relu'))
    model_dnn.add(Dropout(dropout_rate))
    
    model_dnn.add(Dense(16, kernel_initializer=init_mode, activation='relu'))
    model_dnn.add(Dropout(dropout_rate))
    # model.add(Dense(1, activation='relu'))
    # model.add(Flatten())
    model_dnn.add(Dense(2, activation='softmax'))
    # adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
    # model.compile(optimizer=adam,loss='binary_crossentropy', metrics=['accuracy'])
    optimizer = keras.optimizers.Adam(lr=learn_rate)
    model_dnn.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model_dnn.fit(X_train, y_train, epochs=100, batch_size=64)
    return model_dnn


# In[44]:

print('START TRAINING')
print('Dropout_rate = 0.0')
print('Batch_Size = 64')
print('init_model = normal')
print('learning rate = 0.00001')

# Load model V9 to continue training with same paramters.
from keras.models import load_model
model_dnn = load_model('binary_classification_V10.h5')
model_dnn.fit(X_train, Y_train, epochs=1000, batch_size=64, verbose = 2)
model_dnn.save('binary_classification_V10.h5')

# In[143]:


# Print Test Accuracy: 
preds = model_dnn.evaluate(x = X_test, y = Y_test)
### END CODE HERE ###
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# In[103]:


# model_dnn.predict(X_test, batch_size=64)


# In[144]:


preds = model_dnn.evaluate(x = x_train, y = y_train)
### END CODE HERE ###
print ("Loss = " + str(preds[0]))
print ("Train Accuracy = " + str(preds[1]))

