import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import scipy.special as sp
import sklearn
from sklearn.metrics import r2_score
np.random.seed(0)

n_samples_array=[20,30,50,80,95,120]
qf_array=[3,18,30,43,88,92]
var_array = [0,0.3,0.5,1.2,1.7,2]


degrees = [2, 10]
matrix = np.zeros((216,6))


def mylegendre(Qf,X,coef) :
    res = 0
    for m in range(Qf+1):
        leg = sp.legendre(m)
        res+=leg(X)*coef[m]
    return res

def fillMatrix():
    bigcounter=0
    min_o_mes = 0
    max_o_mes = 0
    average_H10=0
    average_H2=0
    n_c_min=0
    qf_c_min=0
    v_c_max=0
    n_c_max=0
    qf_c_max=0
    v_c_max=0
    for i in range(len(n_samples_array)):
        for j in range(len(qf_array)):
            for k in range(len(var_array)):
                n_samples = n_samples_array[i]
                qf=qf_array[j]
                var=var_array[k]
                
                X = np.sort(np.random.uniform(-1.0,1.0,n_samples))
                
                coef = np.random.normal(0,1,qf+1)
                
                y=np.zeros(n_samples)
                
                for i in range(n_samples):
                    y[i]=mylegendre(qf, X[i],coef)
                
                y=y.reshape(1,-1)
                y = sklearn.preprocessing.normalize(y,norm="l2")
                
                y = y[0]
                y = y+var
                
                #plt.figure(figsize=(14, 5))
                counter=0
                resh2=0
                resh10=0
                for i in range(len(degrees)):
                    #ax = plt.subplot(1, len(degrees), i + 1)
                    #plt.setp(ax, xticks=(), yticks=())
                
                    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                             include_bias=False)
                    linear_regression = LinearRegression()
                    pipeline = Pipeline([("polynomial_features", polynomial_features),
                                         ("linear_regression", linear_regression)])
                    
                    pipeline.fit(X[:, np.newaxis], y)
                
                    # Evaluate the models using crossvalidation, use of 10 subsets
                    scores = cross_val_score(pipeline, X[:, np.newaxis], y, scoring="r2",cv=10)
                    if(counter == 0):
                        resh2=-scores.mean()
                    else:
                        resh10=-scores.mean()
                    counter=1
                    X_test = np.linspace(0, 1, 150)
                    
                    #plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
                    #plt.plot(X_test, mylegendre(X_test), label="True function")
                    #plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
                    #plt.xlabel("x")
                    #plt.ylabel("y")
                    #plt.xlim((0, 1))
                    #plt.ylim((-2, 2))
                    #plt.legend(loc="best")
                    #plt.title("Degree {}".format(degrees[i]))
                #plt.show()
                o_mes=resh10-resh2
                average_H2+=resh2
                average_H10+=resh10
                if(o_mes<min_o_mes):
                    min_o_mes = o_mes
                    n_c_min=n_samples
                    qf_c_min=qf
                    v_c_min=var
                if(o_mes>max_o_mes):
                    max_o_mes = o_mes
                    n_c_max=n_samples
                    qf_c_max=qf
                    v_c_max=var
                matrix[bigcounter]=[n_samples,qf,var,resh2,resh10,o_mes]
                bigcounter+=1
    return matrix,average_H2,average_H10,min_o_mes,max_o_mes,n_c_min,qf_c_min,v_c_min,n_c_max,qf_c_max,v_c_max

matrix,average_H2,average_H10,min_o_mes,max_o_mes,n_c_min,qf_c_min,v_c_min,n_c_max,qf_c_max,v_c_max = fillMatrix()

print("Average out of error for H10 ",average_H10/216)
print("Average out of error for H2 ",average_H2/216)
print("Minimum overfitting measure ",min_o_mes)
print("Maximum overfitting measure ",max_o_mes)
print("N,QF,var values that making overfitting measure minimum ", n_c_min," ",qf_c_min," ",v_c_min)
print("N,QF,var values that making overfitting measure maximum ", n_c_max," ",qf_c_max," ",v_c_max)