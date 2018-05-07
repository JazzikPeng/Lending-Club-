
# coding: utf-8

# In[1]:


"""
@Project: Connexin Group 

@FileName: DeepLearning_Model_V3

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
            
@Vindicator：  

"""  


# # I. Select all categorical Variables

# In[2]:


import pandas as pd 
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import LogisticRegression



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


# In[4]:

print("Start Reading")
FILE_PATH = "~/Practicum/loan_data_no_current_converted.csv"
loan_data = readcsv(FILE_PATH)
print("Finished Reading Data")

# In[5]:


df = loan_data.copy()


# In[6]:


# Convert Verification_status_joint, add this categorical data to the categorical list
for idx, i in df["verification_status_joint"].iteritems():
    if i == "Verified":
        df.at[idx, "verification_status_joint"] = 1
    elif i == "Source Verified":
        df.at[idx, "verification_status_joint"] = 2
    elif i == "Not Verified":
        df.at[idx, "verification_status_joint"] = 3


# In[7]:


categorical = ['grade', 'sub_grade', 'emp_length', 'purpose', 'title', 'application_type', 'hardship_flag', 'hardship_type', 'hardship_reason', 
              'hardship_status', 'hardship_loan_status', 'settlement_status', 'disbursement_method', 'home_ownership',
              'pymnt_plan', 'debt_settlement_flag', 'title', 'initial_list_status', 'loan_status', 'verification_status',
              'term', 'verification_status_joint']


# In[8]:


print("There are %d categorical data in our dataset." % len(categorical))


# # II. Import log_reg class 

# In[9]:


class log_reg():
    # Evaluate the model by splitting into train and test sets
    def split(x,y,rand=0, test_size=0.25):
        
        y = np.ravel(y)
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_size,random_state=rand)
        
        return x_train, x_test, y_train, y_test 
    #we need to add validation dataset here
    
    # Find binary column method one
    def bool_cols(df,isbool=True):
        bool_cols=[]
        for col in df:
            if isbool==True:
                if df[col].dropna().value_counts().index.isin([0,1]).all():
                    bool_cols.append(col)
            else:
                if not df[col].dropna().value_counts().index.isin([0,1]).all():
                    bool_cols.append(col)
        return bool_cols
    # this above step is to facilitate normalization later
    # method two
    def not_bi(x):
        not_bi=[]
        for i in list(x):
            u=x[i].unique()
            if not (0 in u and 1 in u and len(u)==2): #if not binary
                not_bi.append(i)
        return not_bi
    
    def reg(x_train, y_train):
           
        #  Update for multiclass classification
        #model = LogisticRegression(penalty='l2',class_weight='balanced',solver='sag',n_jobs=-1)
        model = LogisticRegression(penalty='l2',class_weight='balanced',solver='saga',n_jobs=-1,)
        
#         weight = {1: 1.0, 2: 55.72585567010309, 3: 24.082261111309123, 4: 3.8398317847299177}
#         model = LogisticRegression(penalty='l2',class_weight=weight,solver='saga',n_jobs=-1,multi_class="ovr")


        
        """
        Why we need standardize?
        
        Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features 
        with approximately the same scale. You can preprocess the data with 
        a scaler from sklearn.preprocessing.
        """
        
        model = model.fit(x_train, y_train)
        
        return model
    
    def ModelValuation(x_test,y_test,model):
        
        probs = model.predict_proba(x_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probs[:, 1])
        
        plt.figure(1)
        plt.plot(fpr, tpr, label='LogisticRegression')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
        
        print("Area Under the Curve (AUC) from prediction score is %f" % metrics.roc_auc_score(y_test, probs[:, 1]))
    
        return None  
    
    def y_pred(x_test,threshold=0.5):
        
        if threshold == 0.5:
            y_predicted = model.predict(x_test)
        else:
            probs = model.predict_proba(x_test)
            y_predicted = np.array(probs[:,1] >= threshold).astype(int)
        
        return y_predicted    
    
    def GetScores(y_test,y_predicted):
        #G means score 
        CM = metrics.confusion_matrix(y_test, y_predicted)
        TN = CM[0,0]
        FN = CM[1,0]
        TP = CM[1,1]
        FP = CM[0,1]
        
        sensitivity = float(TP)/float(TP+FN)
        specificity = float(TN)/float(TN+FP)
        G = np.sqrt(sensitivity*specificity)
        print("G score is %f" % G)
        print("Specificity is %f" % specificity)
        
        # Generate and display different evaluation metrics
        print("Mean accuracy score is %f" % metrics.accuracy_score(y_test, y_predicted))
          
        print("Confusion Marix")
        print(CM)
        
        return specificity , G
        
    # Convenience function to plot confusion matrix
    def confusion(y_test,y_predicted,title):
        
        # Define names for the three Iris types
        names = ['Not Default', 'Default']
    
        # Make a 2D histogram from the test and result arrays
        pts, xe, ye = np.histogram2d(y_test, y_predicted, bins=2)
    
        # For simplicity we create a new DataFrame
        pd_pts = pd.DataFrame(pts.astype(int), index=names, columns=names )
        
        # Display heatmap and add decorations
        hm = sns.heatmap(pd_pts, annot=True, fmt="d")
        hm.axes.set_title(title)
        
        return None
            
    def find_threshold(x_test,y_test):
    
        probs = model.predict_proba(x_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, probs[:, 1])
        
        sensitivity = tpr
        specificity = 1 - fpr
        G = np.sqrt(sensitivity*specificity)
        
        plt.figure(2)
        plt.plot(thresholds,G)
        plt.xlabel('Thresholds')
        plt.ylabel('G-Scores')
        plt.title('G-Scores with different thresholds')
        plt.show()
        
        
        print("The highest G score is %f with threshold at %f" % (np.amax(G),thresholds[np.argmax(G)]) )
        
        return thresholds[np.argmax(G)]


# # III. Set, X, Y, Train/Test Sets And normalize it accordingly

# In[10]:


# Value in policy_code contains only 1!?
for i in df['policy_code']:
    if i!=1:
        print(i)


# In[11]:


# Adjustment to Used Features:
drop_col = ["id", "member_id", "funded_amnt","installment", "url", 'int_rate', "grade", "sub_grade", "addr_state","avg_cur_bal", "funded_amnt_inv", "title", 
            "collection_recovery_fee", "collections_12_mths_ex_med", "next_pymnt_d", "recoveries", "total_pymnt",  
            "total_pymnt_inv", "desc", "pymnt_plan",'term']
drop_col2= ["emp_title", "issue_d","earliest_cr_line", "last_pymnt_d", "last_credit_pull_d", "sec_app_earliest_cr_line ", "debt_settlement_flag_date",
            "hardship_start_date", "payment_plan_start_date", "hardship_end_date", "settlement_date", "zip_code",
            "revol_util", "sec_app_earliest_cr_line","settlement_status", "out_prncp_inv", "total_rec_late_fee", "total_bal_ex_mort", "policy_code", "total_rec_prncp"]

drop_sum = list(set(drop_col + drop_col2))

# Update to my Drop Columns
drop_update = ["verification_status", "verification_status_joint", "hardship_flag", "hardship_type", "hardship_reason", 
              "hardship_loan_status", "debt_settlement_flag", "acc_now_delinq", "delinq_amnt", "deferral_term",
              "hardship_amount", "hardship_length", "hardship_dpd", "hardship_payoff_balance_amount", "hardship_last_payment_amount", 
               "out_prncp"]
for i in drop_update:
    drop_sum.append(i)
len(drop_sum)==len(drop_col) + len(drop_col2) + len(drop_update)
# drop the observation that was missing for any field
df.dropna(axis=0, how='all', inplace=True)
df.dropna(axis=1, how='all', inplace=True)

# Drop drop_updated
df.drop(drop_sum, inplace=True, axis=1, errors='ignore')
df.fillna(0, inplace=True)

df.shape


# In[12]:


len(drop_sum)


# In[13]:


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



# In[14]:


print("There are %d numerical features need normalization" %len(features_need_norm))


# In[15]:


Y = df.loc[:,['loan_status']].values
features.remove("loan_status")
categorical_features.remove('loan_status')


# In[16]:


# We can not have loan_status in X, we want to check this, if it prints "Warning", we have a problem!!!
for i in categorical_features:
    if i == "loan_status":
        print("Warning!")
for i in features_need_norm:
    if i == "loan_status":
        print("Warning!")


# In[17]:


# Let's test our result previous result, Previous result set Y to binary number 1,2.
'''
    elif i == "Default":
        loan_data.at[idx, "loan_status"] = 7
    elif i == "Charged Off":
        loan_data.at[idx, "loan_status"] = 8
'''
'''
    def simplify_status(loan_data):
        #does not meet policy: fully paid, can be also considered as fully paid
        loan_data['loan_status'].replace(2,1,inplace = True)
        #does not meet policy: charged off, can be also considered as charged off
        loan_data['loan_status'].replace(3,8,inplace = True)
        #Default merged to charged off
        loan_data['loan_status'].replace(7,8,inplace = True)
        #merge those "late" status
        loan_data['loan_status'].replace(6,5,inplace = True)
        
        #By doing these, we have 1-fully paid, 4-grace period
        # 5-late, and 8-charged off
        #Then we can renumber these categories, as:
        #1-fully paid, 2-grace period, 3-late, 4-charged off
        loan_data['loan_status'].replace(4,2,inplace = True)
        loan_data['loan_status'].replace(8,4,inplace = True)
        loan_data['loan_status'].replace(5,3,inplace = True)
        return loan_data
'''
Y = list(Y.reshape(len(Y)))
for i in range(len(Y)):
    if Y[i]==7 or Y[i]==8:
        Y[i] = 1
    else:
        Y[i] = 0
        


# In[18]:


for i in Y:
    if i != 0 and i!=1:
        print(i)
        break


# In[19]:


# I want to find the starting index and ending index of categorical data in X. 
# starting_col_index_of_categorical_data: starting index of categorical data in X
# last_col_index_in_X: ending index of categorical data in X
# df.loc[:,features_need_norm].values.shape = (891823, 93)
X = df.loc[:,features_need_norm].values
starting_col_index_of_categorical_data = X.shape[1] 


# In[20]:


# df.loc[:,categorical_features].values.shape = (891823, 16)
X = np.concatenate((X, df.loc[:,categorical_features].values), axis=1)
last_col_index_in_X = X.shape[1]-1
# X.shape = (891823, 109)


# In[21]:


len(categorical_features)


# In[22]:


len(features_need_norm)


# In[23]:


# From column index 93 to column index 107 are 15 categorical data in X
# len(categorical_features) = 15
last_col_index_in_X-starting_col_index_of_categorical_data+1 == len(categorical_features)


# In[24]:


# split train test set uing X_transformed_pca
x_train, x_test, y_train, y_test = log_reg.split(X,Y,rand=1, test_size=0.25)
# x_train, x_test, y_train, y_test = log_reg.split(X,Y,rand=None)


# In[25]:


print("shape of x_train: ", x_train.shape)
print("shape of x_test: ", x_test.shape)
print("shape of y_train:", y_train.shape)
print("shape of y_test:", y_test.shape)


# # IV. Normalize Train set and set mean variance in a dictionary:

# In[26]:


def normalize(numerical_features):
    '''
        @description: normalize all numerical features in X, and return a dictionary with mean variance of each features
        @numerical_features： a numpy array of numerical features of shape (668867, 93)
        @return: a dictionary contain mean variance of each features
    '''  
        # Step1: calculate mean variance of each columns in numerical features
    epsilon = 10**-8
    dic = {}
    counter = 0
    for columns in numerical_features.T:
        mean = np.mean(columns)
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

            


# In[27]:


a = np.array([[1,2,3,4,5],[6,7,8,9,10]])
normalize(a)


# In[28]:


# len(mv_dic) = 92
mv_dic = normalize(x_train[:,0: starting_col_index_of_categorical_data])


# In[29]:


# Normalize test set according to train set's mean variance
for key,val in mv_dic.items():
    x_test[:, key] = (x_test[:,key] - val[0] ) / val[1]
    # print(np.mean(x_test[:, key]))
    # print(np.std(x_test[:, key]))


# In[30]:


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
    


# In[31]:


df["loan_status"].value_counts()/ len(df)


# # V. Train a Deep Learning Model

# # with ADAM and Dropout Layer

# In[32]:


from keras.utils.np_utils import to_categorical

X_train = x_train
X_test = x_test

Y_train = y_train
Y_test = y_test

print("shape of X_train: ", X_train.shape)
print("shape of X_test: ", X_test.shape)
print("shape of Y_train:", y_train.shape)
print("shape of Y_test:", y_test.shape)


# In[33]:


# Use Keras to construct a sequential model and visualize it 
from keras import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
import keras
import keras.backend as K

import tensorflow as tf
# Sequential?


# In[34]:

# # Parameter Tuning

print("START PARAMETER TUNING")
# In[38]:


# Step 1. Defind a creat_model() function that returns a model according to different parameters


# In[39]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


# In[40]:


# model = KerasClassifier(build_fn=create_model, epochs=1)


# In[122]:


'''hyper_parameters_1 = {"epochs": [1,2,3], "batch_size" : [4,8,16,32,64,128], 
                    'learn_rate': [0.001, 0.005, 0.01, 0.02, 0.1],
                    'init_mode': ['uniform', 'lecun_uniform', 'normal', 'zero'],
                    'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                    'neurons': [1, 5, 10, 15, 20, 25, 30],
                     'decay':[0.0001,0.001,0.01,0.1]}'''


# In[124]:


hyper_parameters_1 = {"epochs": [5], "batch_size" : [4,8,16,32,64],
                    'learn_rate': [0.001, 0.005, 0.01],
                    'init_mode':  ['uniform','normal'],
                    'dropout_rate': [0.0, 0.1, 0.3, 0.5, 0.7], 
                    'neurons': [1, 10, 30]}


# In[125]:


def create_model(dropout_rate=0.0, init_mode = 'uniform', learn_rate = 0.001, neurons = 4):
    #model_dnn.reset_states()
    model_dnn = Sequential()
    dim = X_train.shape[1]
    model_dnn.add(Dense(neurons, kernel_initializer=init_mode, activation='relu',   input_dim=dim))
    model_dnn.add(Dropout(dropout_rate))
    model_dnn.add(Dense(neurons, kernel_initializer=init_mode, activation='relu'))
    model_dnn.add(Dropout(dropout_rate))
    model_dnn.add(Dense(neurons, kernel_initializer=init_mode, activation='relu'))
    model_dnn.add(Dropout(dropout_rate))
    model_dnn.add(Dense(neurons, kernel_initializer=init_mode, activation='relu'))
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

'''def create_model():
    #model_dnn.reset_states()
    model_dnn = Sequential()
    dim = X_train.shape[1]
    model_dnn.add(Dense(64, activation='relu', input_dim=dim))
    model_dnn.add(Dropout(0.5))
    model_dnn.add(Dense(64, activation='relu'))
    model_dnn.add(Dropout(0.5))
    # model.add(Dense(1, activation='relu'))
    # model.add(Flatten())
    model_dnn.add(Dense(2, activation='softmax'))
    # adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
    # model.compile(optimizer=adam,loss='binary_crossentropy', metrics=['accuracy'])
    model_dnn.compile(optimizer="Adam",loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model_dnn'''


# In[126]:


# sample a small portion of my dataset to accelrate thie process.
np.random.seed(1)
rand_numbers = np.random.randint(0, 668867, int(668867*0.05))
X_para_tuning = X_train[rand_numbers]
Y_para_tuning = Y_train[rand_numbers]


# In[127]:


print("shape of X_para_tuning: ", X_para_tuning.shape)
print("shape of Y_para_tuning:", Y_para_tuning.shape)


# In[128]:


model = KerasClassifier(build_fn=create_model, verbose = 2)
# model.fit(X_para_tuning,Y_para_tuning)


# In[129]:


# hyper_parameters_1 = {"batch_size": [10, 20, 40],
# "epochs" [1,2,3]}


# In[ ]:


param_grid = hyper_parameters_1
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_para_tuning, Y_para_tuning)


# In[ ]:


best_epoches = grid_result.best_params_["epochs"]
best_batch_size = grid_result.best_params_["batch_size"]


# In[ ]:


# print("################################# best_epoches ###############################\n", best_epoches)
# print("################################# best_batch_size ###############################\n", best_batch_size)


# In[ ]:


# Then use best epoches and batch size to tune optimizer


# In[ ]:


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


print("DONE!!!!!!!!!!!!!!!!!!!")
