import numpy
from sklearn import datasets
import random

iris = datasets.load_iris()
# print iris
X = iris.data[:, :2]
# print X
y = iris.target
# print y
# weight =
def generate_abc(N_i):
    params_list=[]
    weight = range(1,10)
    height = range(2,5)
    for i in range(1,N_i):
        dict = {'weight': weight[random.randint(0,len(weight)-1)],'height':weight[random.randint(0,len(weight)-1)]}
        params_list.append(dict)
    return params_list

def generate_svm(N_i):
    params_list=[]
    C = range(1,10)
    Gamma = range(2,5)
    # Kernel =
    for i in range(1,N_i):
        dict = {'c': C[random.randint(0,len(weight)-1)],'gamma': Gamma[random.randint(0,len(gamma)-1)],'kernel': Kernel[random.randint(0,len(Kernel)-1)]}
        params_list.append(dict)
    return params_list

def generate_DT(N_i):
    params_list=[]
    Criterion = ['gini', 'entropy']#also 'chi_square'
    Minimum_samples_split = range(1,11)
    Max_depth = range(1,11)
    Min_samples_leaf = range(1,11)
    Max_leaf_nodes = range(2,20)
    Max_features = ['auto', 'sqrt', 'log2', 'none']
    Min_impurity_split = [1e-7, 1e-4, 1e-6, 1e-8, 1e-5]
    for i in range(1,N_i):
        dict = {'criterion': Criterion[random.randint(0,len(Criterion)-1)],
         'minimum_samples_split': Minimum_samples_split[random.randint(0,len(Minimum_samples_split)-1)],
         'max_depth': Max_depth[random.randint(0,len(Max_depth)-1)],
         'min_samples_leaf': Min_samples_leaf[random.randint(0,len(Min_samples_leaf)-1)],
         'max_leaf_nodes': Max_leaf_nodes[random.randint(0,len(Max_leaf_nodes)-1)],
         'max_features': Max_features[random.randint(0,len(Max_features)-1)],
         'min_impurity_split': Min_impurity_split[random.randint(0,len(Min_impurity_split)-1)]
        params_list.append(dict)
    return params_list

def generate_knn(N_i):
    params_list=[]
    ks = range(1,10)
    distance_metric = range(2,5)
    # Kernel =
    for i in range(1,N_i):
        dict = {'c': C[random.randint(0,len(weight)-1)], 'gamma': Gamma[random.randint(0,len(gamma)-1)],
        'kernel': Kernel[random.randint(0,len(Kernel)-1)]}
        params_list.append(dict)
    return params_list

def generate_xgboost(N_i):
    params_list=[]
    Max_depth = range(3,10 + 1)
    Min_child_weight = range()
    # Gamma =
    # Objective =
    # Reg_alpha =
    # Subsample =
    # Colsample_bytree =
    for i in range(1,N_i):
        dict = {'max_depth': Max_depth[random.randint(0,len(Max_depth)-1)],
         'min_child_weight': Min_child_weight[random.randint(0,len(Min_child_weight)-1)],
         'gamma': Gamma[random.randint(0,len(Gamma)-1)],
         'reg_alpha': Reg_alpha[random.randint(0,len(Reg_alpha)-1)],
         'subsample': Subsample[random.randint(0,len(Subsample)-1)],
         'colsample_bytree': Colsample_bytree[random.randint(0,len(Colsample_bytree)-1)]}
        params_list.append(dict)
    return params_list

P=[0.4,0.4,0.2]
N = 10
# print numpy.random.dirichlet(P)
def function1(A1,A2,B,P,N):
        distribution = numpy.random.dirichlet(P)
        N_i = distribution * N

print generate_DT(10)
