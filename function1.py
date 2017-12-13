import numpy
import numpy as np
from sklearn import datasets
import random
import math
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
# Ignoring Warnings
import warnings
warnings.filterwarnings("ignore")

# print iris
# X = iris.data
# print X
# y = iris.target
# y = np.array(y, dtype=np.float32)
# print y
def generate_abc(N_i):
    params_list=[]
    weight = range(1, 10)
    height = range(2, 5)
    for i in range(1, N_i):
        dict = {'weight': weight[random.randint(0, len(weight)-1)],
         'height':weight[random.randint(0, len(weight)-1)]}
        params_list.append(dict)
    return params_list

def generate_svm(N_i):
    params_list=[]
    C = range(1, 10)
    Gamma = range(2, 5)
    Max_iter = range(1,2)
    Kernel = ['linear', 'poly', 'rbf']
    Decision_function_shape = ['ovr', 'ovo']
    Tol = [1e-7, 1e-4, 1e-6, 1e-8, 1e-5]
    for i in range(1, N_i+1):
        dict = {'C': float(C[random.randint(0, len(C)-1)]),
         'gamma': float(Gamma[random.randint(0, len(Gamma)-1)]),
         'max_iter': float(Max_iter[random.randint(0, len(Max_iter)-1)]),
         'kernel': Kernel[random.randint(0, len(Kernel)-1)],
         'decision_function_shape': Decision_function_shape[random.randint(0, len(Decision_function_shape)-1)],
         'tol': Tol[random.randint(0, len(Tol)-1)],
         'probability': True}
        params_list.append(dict)
    return params_list

def generate_DT(N_i):
    params_list=[]
    Criterion = ['gini', 'entropy']#also 'chi_square'
    Minimum_samples_split = range(2, 11)
    Max_depth = range(1, 11)
    Min_samples_leaf = range(1, 11)
    Max_leaf_nodes = range(2, 20)
    Max_features = ['auto', 'sqrt', 'log2']
    Min_impurity_split = [1e-7, 1e-4, 1e-6, 1e-8, 1e-5]
    for i in range(1, N_i+1):
        dict = {'criterion': Criterion[random.randint(0, len(Criterion)-1)],
         'min_samples_split': Minimum_samples_split[random.randint(0, len(Minimum_samples_split)-1)],
         'max_depth': Max_depth[random.randint(0, len(Max_depth)-1)],
         'min_samples_leaf': Min_samples_leaf[random.randint(0, len(Min_samples_leaf)-1)],
         'max_leaf_nodes': Max_leaf_nodes[random.randint(0, len(Max_leaf_nodes)-1)],
         'max_features': Max_features[random.randint(0, len(Max_features)-1)],
         'min_impurity_split': Min_impurity_split[random.randint(0, len(Min_impurity_split)-1)]}
        params_list.append(dict)
    return params_list

def generate_knn(N_i):
    params_list=[]
    N_neighbours = range(3, 10)
    Weights = ['distance', 'uniform']
    P = range(1, 5)
    Algorithm = ['brute', 'auto', 'ball_tree', 'kd_tree']
    for i in range(1, N_i+1):
        dict = {'n_neighbors': N_neighbours[random.randint(0, len(N_neighbours)-1)],
         'weights': Weights[random.randint(0, len(Weights)-1)],
         'p': P[random.randint(0, len(P)-1)],
         'algorithm': Algorithm[random.randint(0, len(Algorithm)-1)]
         # , 'n_jobs': -1
         # 'leaf_size': Leaf_size[random.randint(0, len(Leaf_size)-1)]
         }
        params_list.append(dict)
    return params_list

def generate_xgboost():
    params_list=[]
    Max_depth = range(3, 10 + 1)#in case doesn't work inc step size to 2
    Booster = ['gbtree', 'gblinear' ]
    Objective = ['reg:logistic', 'reg:linear', 'binary:logistic']
    Min_child_weight = range(1, 6) #in case doesn't work inc step size to 2
    Learning_rate = numpy.arange(0, 0.5, 0.001)
    # print Learning_rate[3]
    Gamma = [gm/10 for gm in range(0, 5)]
    # for i in range(1, N_i):
    dict = {'params': {'max_depth': Max_depth[random.randint(0, len(Max_depth)-1)],
          'min_child_weight': Min_child_weight[random.randint(0, len(Min_child_weight)-1)],
          'gamma': Gamma[random.randint(0, len(Gamma)-1)],
          'booster': Booster[random.randint(0, len(Booster)-1)],
          'objective': Objective[random.randint(0, len(Objective)-1)],
          'learning_rate': Learning_rate[random.randint(0, len(Learning_rate)-1)]}}
    # params_list.append(dict)
    return dict

P = [0.4, 0.4, 0.2]
N = 10
# def function1(A1, A2, A3, B, P, N):
def GenParams(P, N):
        distribution = numpy.random.dirichlet(P)
        N_i = distribution * N
        for iter in range(0, len(N_i)):
             N_i[iter] = math.ceil(N_i[iter])
        N_i = np.array(N_i, dtype=np.int64)
        Chi = []
        print N_i
        Chi.append(generate_svm(N_i[0]))
        Chi.append(generate_knn(N_i[1]))
        Chi.append(generate_DT(N_i[2]))
        print Chi[0][0]
        Phi = generate_xgboost()
        # print Phi
        return Phi, N_i, Chi

# Phi, N_Bold, Chi = GenParams(P, N)

# print generate_DT(10)
# print generate_xgboost(10)

def Blend(L, Chi, N, X, y):
    print 'N'
    print N
    rho = 0.7
    #Dfw=NUll
    # for i in range(1,l+1):
    sample_size = int(rho * len(X))
    indices = range(0,len(X))
    new_indices = random.sample(indices, sample_size)
    # print new_indices
    D_dash = X[new_indices]
    Lables_dash = y[new_indices]
    type(Lables_dash)
    remaining_indices = list(set(indices) - set(new_indices))
    D_complement = X[remaining_indices]
    Labels_complement = y[remaining_indices]
    # for i in range(0,3):
    M=[]
    ## SVM
    temp = []
    # print D_dash[0  ]
    for j in range(0,N[0]):
    # print(Chi[0][0])
        Object = svm.SVC()
        Object.set_params(**Chi[0][j])
        temp.append(Object.fit(D_dash,Lables_dash))
    # print(Object)
    M.append(temp)
    ## KNN
    temp = []
    for j in range(0,N[1]):
        Object = KNeighborsClassifier()
        Object.set_params(**Chi[1][j])
        temp.append(Object.fit(D_dash,Lables_dash))
    M.append(temp)
    ## DecisionTrees
    temp = []
    for j in range(0,N[2]):
        Object = DecisionTreeClassifier()
        Object.set_params(**Chi[2][j])
        temp.append(Object.fit(D_dash,Lables_dash))
    M.append(temp)
    # print D_dash[0]
    # return M
    Dash_Data = zip(D_dash, Lables_dash)
    # print len(Dash_Data)
    # print Dash_Data
    # print len(M)
    # for i in M:
        # print len(i)
    # for i in D_dash:
    Data = D_complement
    var = Data[0]
    # print D_dash[0]
    F_for_all_algo = []
    for i in M:
        # print i
        # F_for_each_algo = []
        F_for_each_algo = np.empty([len(i),3])
        for j in range(0,len(i)):
            np.vstack((F_for_each_algo, i[j].predict_proba(var.reshape(1,-1))))
            # F_for_each_algo.append(i[j].predict_proba(var.reshape(1,-1)))
            # print i
        # print 'Here'
        # print F_for_each_algo
        F_for_all_algo.append(F_for_each_algo)
        # np.append(F_for_all_algo, F_for_each_algo)
    print "F"
    print F_for_all_algo

    G_for_all_algo = []
    for i in M:
        G_for_each_algo = np.empty([1,len(i)])
        for j in range(0,len(i)):
            # print i
            # print i[j].predict(var.reshape(1,-1))
            G_for_each_algo[0][j] = i[j].predict(var.reshape(1,-1))
        # print G_for_each_algo
        G_for_all_algo.append(G_for_each_algo)
    print "G"
    print G_for_all_algo

    cross_product_of_G_F = []
    for i in range(0,len(M)):
        cross_product_of_G_F.append(np.matmul(G_for_all_algo[i],F_for_all_algo[i]))
    print "Cross Product"
    print cross_product_of_G_F
    print "KNN Model Row"
    # print M[1]
    # print cross_product_of_G_F

# Blend(3, Chi, N_Bold)

def cv_split(X, y, k):
    # X = iris.data
    # y = iris.target
    Combined_Data = zip(X,y)
    np.random.shuffle(Combined_Data)
    X,y = zip(*Combined_Data)
    X = np.array(X)
    y = np.array(y)
    A = np.split(X,k)
    B = np.split(y,k)
    return A, B

# A,B = cv_split(3)
# print B.pop(2)
def BlendingEnsemble(X, y, k, Chi, N_Bold):

    X_Array, y_Array = cv_split(X, y, k)
    for i in range(0,k):
        X_Array_Temp = []
        y_Array_Temp = []
        for j in range(0,k):
            if(j!=i):
                X_Array_Temp.append(X_Array[j])
                y_Array_Temp.append(y_Array[j])
        # del X_Array_Temp[i]
        data_Training_X = np.concatenate((X_Array_Temp))
        data_Training_y = np.concatenate((y_Array_Temp))
        # print type(data_Training_X)
        # Phi, N_Bold, Chi = GenParams(P, N)
        Models_list = Blend(2, Chi, N_Bold, data_Training_X, data_Training_y)
        # print data_Training.shape
        print Models_list
    # q = Models_list[1][0].predict(X_Array[k-1])
    # print accuracy_score(y_Array[k-1],q) * 100
    q = Models_list[2][0].predict(X_Array[k-1])
    print accuracy_score(y_Array[k-1],q) * 100
X = iris.data
y = iris.target
Phi, N_Bold, Chi = GenParams(P, N)
Blend(2, Chi, N_Bold, X, y)
# BlendingEnsemble(iris.data, iris.target, 5, Chi, N_Bold)

# from sklearn.metric import accuracy
# q=svm.predict(X)
# print accuracy(q, y)
