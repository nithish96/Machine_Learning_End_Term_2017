import numpy
import numpy as np
from sklearn import datasets
import random
import math
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
# print iris
X = iris.data
# print X
y = iris.target
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
         'tol': Tol[random.randint(0, len(Tol)-1)]}
        params_list.append(dict)
    return params_list

def generate_DT(N_i):
    params_list=[]
    Criterion = ['gini', 'entropy']#also 'chi_square'
    Minimum_samples_split = range(1, 11)
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
         'algorithm': Algorithm[random.randint(0, len(Algorithm)-1)], 'n_jobs': -1
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

def Blend(L, Chi, N):
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

    print M

# Blend(3, Chi, N_Bold)

def cv_split(k):
    X = iris.data
    # print X
    y = iris.target
    Combined_Data = zip(X,y)
    np.random.shuffle(Combined_Data)
    X,y = zip(*Combined_Data)
    X = np.array(X)
    y =np.array(y)
    A = np.split(X,k)
    B = np.split(y,k)
    return [A,B]

A = cv_split(3)
# print A[0]
# print A[1]
print A[0][0].shape
print A[1][0].shape
print A[0][1].shape
print A[1][1].shape
print A[0][2].shape
print A[1][2].shape
# print A[1][0]
# print A[1][1]
# print A[1][2]
print np.count_nonzero(A[1][0]==0)
print np.count_nonzero(A[1][0]==1)
print np.count_nonzero(A[1][0]==2)
print 'fcdsfvd'
print np.count_nonzero(A[1][1]==0)
print np.count_nonzero(A[1][1]==1)
print np.count_nonzero(A[1][1]==2)
