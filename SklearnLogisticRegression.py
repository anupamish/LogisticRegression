#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568

import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as Accuracy

def main():
    data=np.loadtxt(sys.argv[1],dtype='float',delimiter=',',usecols=(0,1,2,4))
    input_data = data[:,0:3]
    labels =data[:,3] 
    predictor = LogisticRegression(fit_intercept = True, max_iter = 7000)
    predictor.fit(input_data, labels)
    accuracy = Accuracy(predictor.predict(input_data),labels)  
    print ("Weights: ", predictor.coef_)
    print("Accuracy:", accuracy)

if __name__ == '__main__':
    main()