import sys
import arff
import copy
import scipy
import sklearn
import argparse
import warnings
import numpy as np
from sklearn import *
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import *

##############################################################################################################
# Function to convert strings in raw data to int and return a list and its element are 
# [processed data matrix, training data, label, distigunish value in label,
# number of #distigunish value in label, the original dataset]
def process_data(filename):
	raw_data = arff.load(open(filename, 'rb'))
	#print raw_data
        raw_train_original = raw_data[u'data']
        raw_train = copy.deepcopy(raw_train_original)
	raw_train_y = copy.deepcopy(raw_train_original)
      	train_rows = len(raw_train)
	train_cols = len(raw_train[0])
	#print train_cols
	train = np.zeros((train_rows, train_cols))
	L = []    
	for j in range (train_cols):
		if type(raw_train[0][j]) is unicode:
			temp = ["" for x in range(train_rows)]
			for i in range(train_rows):
				temp[i] = raw_train[i][j]
			unique_item = np.unique(temp).tolist()
			L.append(unique_item)		
		else:
			L.append(0)
	#print temp
	for j in range (train_cols):
		for i in range (train_rows):		
			if type(raw_train[i][j]) is unicode:
				raw_train[i][j] = float(L[j].index(str(raw_train[i][j])))
				#print(raw_train[i][j], type(raw_train[i][j]))	
			else:
				float(raw_train[i][j])			
	for j in range (train_cols):
		for i in range (train_rows):		
			train[i, j] = raw_train[i][j]

	#print L
        x = np.delete(train, train_cols-1, 1)
        #print x
	       

        if type(raw_train_original[0][train_cols-1]) is unicode:
                y = ["" for i in range(train_rows)]
                for j in range(train_rows):
                        y[j] = raw_train_original[j][train_cols-1] 

        else:
                y = np.zeros((train_rows, 1))
                for j in range(train_rows):
                        y[j, 0] = raw_train_original[j][train_cols-1]
        #print type(y)
        #print y
	return [train, x, y]
#####################################################################################################################
#command line argument configuration
#warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument("-T", type=str, nargs= 1, dest = 'train', required = True,
                   help="training data set")
parser.add_argument("-t", type=str, nargs= 1, dest = 'test', required = True,
                    help="testing data set")
parser.add_argument("-O", type=str, nargs= 1, dest = 'out_file',required = True,
                    help="output predictions file name")
parser.add_argument("-w", type=str, choices=['svm', 'rfc', 'nb'],
                   nargs = 1, required = True, dest ='classifiers', help="classifier to use.")
parser.add_argument("-o", nargs = '*', type = str, dest = 'para',
                    help="classifier options")
args = parser.parse_args()
######################################################################################################################
# obtain train and test data
train = process_data(args.train[0])
test = process_data(args.test[0])
x_train = train[1]
y_train = train[2]

#print y_train
x_test = test[1]
y_test = test[2]

#print y_test

######################################################################################################################
#classifiers
if args.classifiers[0] == 'svm':
    if args.para == None:
        print 'Warning: Option -o is required'
        exit()
    if args.para == []:
        print 'Warning: Option -o must have two arguments: kernal and Penalty parameter C of the error term'
        exit()
    try:
        float(args.para[1])
    except ValueError:
        print 'Warning: The Penalty parameter C of the error term (seconed argument of -o) must be a number!'
    if args.para[0] not in ['linear', 'rbf']:
        print 'kernal must be linear or rbf'
    else:
        clf = svm.SVC(C=float(args.para[1]), cache_size=40, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel=args.para[0], max_iter=-1, probability=True, random_state=1, shrinking=True, tol=0.001, verbose=False)



clf.fit(x_train, y_train)
y_label = clf.predict(x_test)
#print type(y_label)
#print type(y_test)
#print y_label
################################################################################################################################
# write results to file

p = np.zeros((len(y_label), len(clf.classes_)), dtype = '|S8')


for i in range(len(x_test)):
	for j in range(len(clf.classes_)):
		prob = clf.predict_proba(x_test[i])
		p[i][j] = str("{:.5f}".format(prob[0][j]))


with open(args.out_file[0],"w") as f:
	f.write('---\n')
	for i in range(len(clf.classes_)):
		f.write(str(clf.classes_[i])+ ":\n")
		for j in range(len(x_test)):			
			f.write('     - '+p[j][i]+ "\n")
f.close()


'''
test whether the sum of the prob of each class in ith entry is equal to 1
for i in range(len(x_test)):

		a= float(p[i][0])+ float(p[i][1])+ float(p[i][2])+ float(p[i][3])+ float(p[i][4])+ float(p[i][5])+ float(p[i][6])+ float(p[i][7])+ float(p[i][8])+ float(p[i][9])
		print a

'''

