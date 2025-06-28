import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def sklearn_result():
    df = pd.read_csv('test_scores.csv')
    reg = linear_model.LinearRegression()
    reg.fit(df[['math']], df.cs)
    return reg.coef_, reg.intercept_

def gradient_descent(x, y):
    m_curr = b_curr = 0
    n = len(x)
    learning_rate = 0.0002
    epochs = 500000
    prev_cost = 0
    for i in range(epochs):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum(val **2 for val in (y - y_predicted))
        md = -(2/n) * sum(x * (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        
        if math.isclose(prev_cost, cost, rel_tol=1e-20):
            print("Converged after {} iterations".format(i))
            print("Cost: {}".format(cost))
            break
        
        prev_cost = cost
        

    return m_curr, b_curr




if __name__ == "__main__":
    df = pd.read_csv("test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x,y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = sklearn_result()
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn, b_sklearn))
        
    