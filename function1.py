import numpy
import numpy as np
from sklearn import datasets
import warnings
warnings.filterwarnings("ignore") #Ignore Deprecation warnings of the paramter min_impurity_split in DecisionTrees
import random
import math
from xgboost.sklearn import XGBClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
# Ignore Warnings

def generate_svm(N_i):
    '''Generate sets of random values of Hyperparameters for SVM Algorithm'''
    params_list=[]
    C = range(1, 10) #Define range of values for C
    Gamma = range(2, 5) #Define range of values for Gamma
    Max_iter = range(1,2) #Define range of values for Max_iter
    Kernel = ['linear', 'poly', 'rbf'] #Define range of values for Kernel
    Decision_function_shape = ['ovr', 'ovo'] #Define range of values for Decision_function_shape
    Tol = [1e-7, 1e-4, 1e-6, 1e-8, 1e-5] #Define range of values for Tolerance
    for i in range(1, N_i+1):
    #For the given length of N_i, generate N_i sets of Hyperparameters with values of each parameter picked at random
        dict = {'C': float(C[random.randint(0, len(C)-1)]),
         'gamma': float(Gamma[random.randint(0, len(Gamma)-1)]),
         'max_iter': float(Max_iter[random.randint(0, len(Max_iter)-1)]),
         'kernel': Kernel[random.randint(0, len(Kernel)-1)],
         'decision_function_shape': Decision_function_shape[random.randint(0, len(Decision_function_shape)-1)],
         'tol': Tol[random.randint(0, len(Tol)-1)],
         'probability': True}
        params_list.append(dict)
    return params_list #Return the array of dictionaries where each dictionary is a set of Hyperparameters

def generate_DT(N_i):
    '''Generate sets of random values of Hyperparameters for DT Algorithm'''
    params_list=[]
    Criterion = ['gini', 'entropy']#also 'chi_square' #Define range of values for Criterion
    Minimum_samples_split = range(2, 11) #Define range of values for Minimum_samples_split
    Max_depth = range(1, 11) #Define range of values for Max_depth
    Min_samples_leaf = range(1, 11) #Define range of values for Min_samples_leaf
    Max_leaf_nodes = range(2, 20) #Define range of values for Max_leaf_nodes
    Max_features = ['auto', 'sqrt', 'log2'] #Define range of values for Max_features
    Min_impurity_split = [1e-7, 1e-4, 1e-6, 1e-8, 1e-5] #Define range of values for Min_impurity_split
    for i in range(1, N_i+1):
    #For the given length of N_i, generate N_i sets of Hyperparameters with values of each parameter picked at random
        dict = {'criterion': Criterion[random.randint(0, len(Criterion)-1)],
         'min_samples_split': Minimum_samples_split[random.randint(0, len(Minimum_samples_split)-1)],
         'max_depth': Max_depth[random.randint(0, len(Max_depth)-1)],
         'min_samples_leaf': Min_samples_leaf[random.randint(0, len(Min_samples_leaf)-1)],
         'max_leaf_nodes': Max_leaf_nodes[random.randint(0, len(Max_leaf_nodes)-1)],
         'max_features': Max_features[random.randint(0, len(Max_features)-1)],
         'min_impurity_split': Min_impurity_split[random.randint(0, len(Min_impurity_split)-1)]}
        params_list.append(dict)
    return params_list #Return the array of dictionaries where each dictionary is a set of Hyperparameters

def generate_knn(N_i):
    '''Generate sets of random values of Hyperparameters for KNN Algorithm'''
    params_list=[]
    N_neighbors = range(3, 10) #Define range of values for N_neighbors
    Weights = ['distance', 'uniform'] #Define range of values for Weights
    P = range(1, 5) #Define range of values for P
    Algorithm = ['brute', 'auto', 'ball_tree', 'kd_tree'] #Define range of values for Algorithm
    for i in range(1, N_i+1):
    #For the given length of N_i, generate N_i sets of Hyperparameters with values of each parameter picked at random
        dict = {'n_neighbors': N_neighbors[random.randint(0, len(N_neighbors)-1)],
         'weights': Weights[random.randint(0, len(Weights)-1)],
         'p': P[random.randint(0, len(P)-1)],
         'algorithm': Algorithm[random.randint(0, len(Algorithm)-1)]
         # , 'n_jobs': -1
         # 'leaf_size': Leaf_size[random.randint(0, len(Leaf_size)-1)]
         }
        params_list.append(dict)
    return params_list #Return the array of dictionaries where each dictionary is a set of Hyperparameters

def generate_xgboost():
    '''Generate Hyperparameters for XGBoost Algorithm'''
    params_list=[]
    Max_depth = range(3, 10 + 1) #Define range of values for Max_depth
    Booster = ['gbtree', 'gblinear' ] #Define range of values for Booster
    Objective = ['reg:logistic', 'reg:linear', 'binary:logistic'] #Define range of values for Objective
    Min_child_weight = range(1, 6) #Define range of values for Min_child_weight
    Learning_rate = numpy.arange(0, 0.5, 0.001) #Define range of values for learning_rate
    Gamma = [gm/10 for gm in range(0, 5)] #Define range of values for Gamma
    '''Generate 1 set of random values of Hyperparameters'''
    dict = {'params': {'max_depth': Max_depth[random.randint(0, len(Max_depth)-1)],
          'min_child_weight': Min_child_weight[random.randint(0, len(Min_child_weight)-1)],
          'gamma': Gamma[random.randint(0, len(Gamma)-1)],
          # 'booster': Booster[random.randint(0, len(Booster)-1)],
          'objective': Objective[random.randint(0, len(Objective)-1)],
          'learning_rate': Learning_rate[random.randint(0, len(Learning_rate)-1)]}}
    return dict #Return the dictionary of the value of Hyperparameters

def GenParams(P, N):
    '''Generate a set of parameters Phi(For all the base algorithms) and paramters for Chi(For the blender algorithm) and array of count of each base algorithm'''
    distribution = numpy.random.dirichlet(P) #Generate Dirchlet distribution for input P
    N_i = distribution * N #Generate a new set N_i by multiplying N with each value of the distribution
    for iter in range(0, len(N_i)): #Convert each value to it's ceil value
         N_i[iter] = math.ceil(N_i[iter])
    N_i = np.array(N_i, dtype=np.int64) #Convert the array into numpy array
    Chi = [] #Array to contain sets of paramters for each algo
    Chi.append(generate_svm(N_i[0])) #Generate parameter sets for base algorithm SVM
    Chi.append(generate_knn(N_i[1])) #Generate parameter sets for base algorithm KNN
    Chi.append(generate_DT(N_i[2])) #Generate parameter sets for base algorithm DT
    Phi = generate_xgboost() #Generate parameters for blend algorithm XGBoost
    return Phi, N_i, Chi #Return Phi, N_i & Chi

def one_hot_encode(A, Dim):
    '''Do hot encoding on given input matrix A with the given dimensions Dim'''
    b = np.zeros((len(A),Dim))
    b[np.arange(len(A)),A] = 1
    return b #Return the encoded matrix

def G(M, Data, Models_count, Number_of_algo, Number_of_classes):
    '''Generate matrix G'''
    G = np.empty([len(Data),Models_count*Number_of_classes]) #Size of the G matrix would number of rows = rows in Data, number of columns = Total number of models * number of classes for given dataset
    k = 0
    #For each model and for each row predict the probability and horizontally append them for each model and vertically for each row
    for D in Data:
        G_each_datarow = []
        for i in M:
            temp = i.predict_proba(D.reshape(1,-1))
            for i in temp[0]:
                G_each_datarow.append(i)
        G[k] = G_each_datarow
        k = k+1
    return G

def F(G, Data, Models_count, Number_of_algo):
    '''Generate matrix F'''
    F = np.empty([len(Data),G.shape[1]*Data.shape[1]]) #Size of the matrix F would be number of rows = rows in Data, number of columns = number of columns in G * Number of columns in Data
    #For each element in G and for each element in Data is mutiplied and appended horizontally for each row
    for i in range(0,Data.shape[1]):
        F_each_datarow = []
        for G_each_datapoint in G[i]:
            for D in Data[i]:
                F_each_datarow.append(D*G_each_datapoint)
        F[i] = F_each_datarow
    return F

def Build_Models(N, Chi, D, L):
    '''Build models for the given parameters and given base algorithms on a data'''
    M = []
    # SVM
    for j in range(0,N[0]):
        Object = svm.SVC()
        Object.set_params(**Chi[0][j])
        M.append(Object.fit(D,L))
    # KNN
    for j in range(0,N[1]):
        Object = KNeighborsClassifier()
        Object.set_params(**Chi[1][j])
        M.append(Object.fit(D,L))
    # DecisionTrees
    for j in range(0,N[2]):
        Object = DecisionTreeClassifier()
        Object.set_params(**Chi[2][j])
        M.append(Object.fit(D,L))
    return M

def Blend(Number_of_algo, L, Phi, Chi, N, X, y):
    '''Build a new data set with the created F and G matrixes for L levels. Train the Belended algorithm with the new data and it's corresponding parameters'''
    rho = 0.7
    Levels_D_fw = []
    for i in range(0,L):
        sample_size = int(rho * len(X)) #Get sample size for this level
        indices = range(0,len(X)) #Get all the indices of the data to sample
        new_indices = random.sample(indices, sample_size) #sample the indices according to the sample size
        D_dash = X[new_indices] #Retrieve sampled indices data from X
        Labels_dash = y[new_indices] #Retrieve sampled indices data from y
        remaining_indices = list(set(indices) - set(new_indices)) #Get those indices which are not sampled
        D_complement = X[remaining_indices] #Retrieve remaining indices data from X
        Labels_complement = y[remaining_indices] #Retrieve remaining indices data from y
        Labels_complement = Labels_complement.reshape(-1,1) #Convert a one dimensional single array into a two dimensional single array
        M = Build_Models(N, Chi, D_dash, Labels_dash) #Generate models for the sampled indices data
        Models_count = len(M) #Total number of Models
        G_Matrix = G(M, D_complement, Models_count, Number_of_algo, 3) #Generate G
        F_Matrix = F(G_Matrix, D_complement, Models_count, Number_of_algo) #Generate F
        D_fw_temp = np.concatenate((D_complement,G_Matrix,F_Matrix,Labels_complement), axis=1) #Generate new dataset i.e add the elements of reamining data, F & G row wise
        Levels_D_fw.append(D_fw_temp) #Add each such new dataset for the present Level

    '''Number of columns of D_fw would be d(columns of the data set) + Nc(Number of models * Number of classes)
     + dNc(Number of columns * Number of models * Number of classes) + 1(Labels columns which is 1)'''

    D_fw = np.concatenate(Levels_D_fw, axis=0) #Column wise append all the new datasets generated for each Level
    Object = XGBClassifier() #Create a base algorithm classifier
    Object.set_params(**Phi['params']) #Set the randomly generated params for the base algorithm classifier
    Object.fit(D_fw[:,:(D_fw.shape[1]-1)],D_fw[:,[D_fw.shape[1]-1]]) #Train the base algorithm with the new Data
    return Object #Return the trained Model

def cv_split(X, y, k):
    '''Create data for cross validation'''
    Combined_Data = zip(X,y)
    np.random.shuffle(Combined_Data) #Shuffle the data
    X,y = zip(*Combined_Data)
    X = np.array(X)
    y = np.array(y)
    A = np.split(X,k)
    B = np.split(y,k)
    return A, B #Return array of data split into the given k times

def BlendingEnsemble(Number_of_algo, L, X, y, k, P, N_Bold):
    '''Split the dataset using k-fold cross validation.
    Use k-1 for training the blended model and the other for testing the blended model.
    And then find the model that gives the minimum error or maximum accuracy using cross validation.
    Now build on the blended algorithm using the entire dataset & the best Hyperparameters obtained from cross validation'''
    r_list = [] #Blended model for each r
    R = 3 #Number of times the loop has to be ran
    List_of_inputs = []
    for r in range(0,R):
        Phi, N, Chi = GenParams(P, N_Bold) #Get Phi, N, Chi
        List_of_inputs.append([Phi,N,Chi]) #Store the Hyperparameters that are used in every iteration
        X_Array, y_Array = cv_split(X, y, k) #Split the data for corss validation
        Models_list = []
        accuracies = []
        for i in range(0,k):
            X_Array_Temp = []
            y_Array_Temp = []
            for j in range(0,k):
                if(j!=i):
                    X_Array_Temp.append(X_Array[j])
                    y_Array_Temp.append(y_Array[j])
            # concatenate the data that has to be used for training as a part of cross validation
            data_Training_X = np.concatenate((X_Array_Temp))
            data_Training_y = np.concatenate((y_Array_Temp))
            #build model using the current paramters and cross validated train data
            Model = Blend(Number_of_algo, L, Phi, Chi, N, data_Training_X, data_Training_y)
            Models_list.append(Model) #Append all the models created
            '''Build the training data'''
            M = Build_Models(N, Chi, X_Array[i], y_Array[i]) #Build the models for the test data which has been split as a part of the cross validation
            G_Matrix = G(M, X_Array[i], len(M), Number_of_algo, 3) #Generate G to buid the testing data
            F_Matrix = F(G_Matrix, X_Array[i], len(M), Number_of_algo) #Generate F to build the testing data
            test_data_new = np.concatenate((X_Array[i],G_Matrix,F_Matrix),axis=1) #Generate test dataset
            q = Models_list[i].predict(test_data_new) #Predict using new test data on the current model
            accuracies.append(1 - accuracy_score(y_Array[i],q)) #Calculate error
        r_list.append(np.mean(accuracies)) #Calculate mean of the error found using Cross Validation
        print r_list
    r_star = r_list.index(np.min(r_list)) #Find the parameter set where the error is minimum
    print r_star
    '''Train the model on the blended algorithm using the min error paramters set'''
    Output_Model = Blend(Number_of_algo, L, List_of_inputs[r_star][0], List_of_inputs[r_star][2],List_of_inputs[r_star][1], X, y)
    return Output_Model #Output the best model

if __name__ == "__main__":
    Number_of_algo = 3
    P = [0.4, 0.4, 0.2]
    N = 10
    L = 2
    cross_validation_times = 3
    iris = datasets.load_iris() #Using the iris dataset
    BlendingEnsemble(Number_of_algo, L, iris.data, iris.target, cross_validation_times, P, N)
