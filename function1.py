import numpy
from sklearn import datasets
import random

iris = datasets.load_iris()
# print iris
X = iris.data[:, :2]
# print X
y = iris.target
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
    Max_iter = range()
    Kernel = ['linear', 'poly', 'rbf']
    Decision_function_shape = ['ovr', 'ovo']
    Tol = [1e-7, 1e-4, 1e-6, 1e-8, 1e-5]
    for i in range(1, N_i):
        dict = {'C': C[random.randint(0, len(weight)-1)],
         'gamma': Gamma[random.randint(0, len(gamma)-1)],
         'max_iter': Max_iter[random.randint(0, len(Max_iter)-1)],
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
    Max_features = ['auto', 'sqrt', 'log2', 'none']
    Min_impurity_split = [1e-7, 1e-4, 1e-6, 1e-8, 1e-5]
    for i in range(1, N_i):
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
    for i in range(1, N_i):
        dict = {'n_neighbors': N_neighbours[random.randint(0, len(N_neighbours)-1)],
         'weights': Weights[random.randint(0, len(Weights)-1)],
         'p': P[random.ran2dint(0, len(P)-1)],
         'algorithm': Algorithm[random.randint(0, len(Algorithm)-1)], 'njobs': -1
         # 'leaf_size': Leaf_size[random.randint(0, len(Leaf_size)-1)]
         }
        params_list.append(dict)
    return params_list

def generate_xgboost(N_i):
    params_list=[]
    Max_depth = range(3, 10 + 1)#in case doesn't work inc step size to 2
    Booster = ['gbtree', 'gblinear' ]
    Objective = ['reg:logistic', 'reg:linear', 'binary:logistic']
    Min_child_weight = range(1, 6) #in case doesn't work inc step size to 2
    Learning_rate = numpy.arange(0, 0.5, 0.001)
    print Learning_rate[3]
    Gamma = [gm/10 for gm in range(0, 5)]
    for i in range(1, N_i):
        dict = {'params': {'max_depth': Max_depth[random.randint(0, len(Max_depth)-1)],
          'min_child_weight': Min_child_weight[random.randint(0, len(Min_child_weight)-1)],
          'gamma': Gamma[random.randint(0, len(Gamma)-1)],
          'booster': Booster[random.randint(0, len(Booster)-1)],
          'objective': Objective[random.randint(0, len(Objective)-1)],
          'learning_rate': Learning_rate[random.randint(0, len(Learning_rate)-1)]}}
        params_list.append(dict)
    return params_list

P=[0.4, 0.4, 0.2]
N = 10
# print numpy.random.dirichlet(P)
def function1(A1, A2, A3, B, P, N):
        distribution = numpy.random.dirichlet(P)
        N_i = distribution * N
        Chi = []
        Chi.append(generate_svm(N_i))
        Chi.append(generate_knn(N_i))
        Chi.append(generate_DT(N_i))
        Phi = generate_xgboost(N_i)
print generate_DT(10)
print generate_xgboost(10)
