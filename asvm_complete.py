import re
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
parser = argparse.ArgumentParser(prog='SVM classifier', usage='%(prog)s [options]')

parser.add_argument("-T", type=str, dest = 'train', required = True,
                   help="training data set")
parser.add_argument("-t", type=str, dest = 'test', required = True,
                    help="testing data set")
parser.add_argument("-O", type=str, dest = 'out_file',required = True,
                    help="output predictions file name")

parser.add_argument("-k", type=str, dest = 'kernel',default ='linear', choices = ['linear', 'rbf', 'poly', 'sigmoid'], help="Specifies the kernel type to be used in the algorithm.default= rbf")

parser.add_argument("-c", type=float, dest = 'C',default =1.0, help="Penalty parameter C of the error term.default = 1.0")

parser.add_argument("-d", type=int, dest = 'degree',default =3, help="Degree of the polynomial kernel function poly. Ignored by all other kernels.default = 3")

parser.add_argument("-g", type=float, dest = 'gamma',default =0.0, help="Kernel coefficient for rbf, poly and sigmoid. If gamma is 0.0 then 1/n_features will be used instead.default = 0.0")

parser.add_argument("-C", type=float, dest = 'coef',default =0.0, help="Independent term in kernel function. It is only significant in poly and sigmoid.default = 0.0")

parser.add_argument("-o", type=float, dest = 'tol',default =1e-3, help="Tolerance for stopping criterion.default = 1e-3")

parser.add_argument("-s", type=float, dest = 'cache_size',default =40, help="Specify the size of the kernel cache (in MB).default=40")

parser.add_argument("-m", type=int, dest = 'max_iter',default =-1, help="Hard limit on iterations within solver, or -1 for no limit.default=-1")

parser.add_argument("-r", type=int, dest = 'random_state',default =None, help="The seed of the pseudo random number generator to use when shuffling the data for probability estimation.default=None")

parser.add_argument("-w", dest = 'class_weight',default =None, choices=[None, 'auto'],help="Set the parameter C of class i to class_weight[i]*C for SVC.default=None")

parser.add_argument('--shrinking', dest='shrinking', action='store_true', help='use the shrinking heuristic.default=True')
parser.add_argument('--no-shrinking', dest='shrinking', action='store_false', help='not to use the shrinking heuristic.default=True')
parser.set_defaults(shrinking=True)

parser.add_argument('--verbose', dest='verbose', action='store_true', help='Enable verbose output.default=false')
parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='Disable verbose output.default=false')
parser.set_defaults(verbose=False)

args = parser.parse_args()
'''
print args.kernel
print type(args.kernel)
print args.C
print type(args.C)
print args.degree
print type(args.degree)
print args.gamma
print type(args.gamma)
print args.coef
print type(args.coef)
print args.shrinking
print type(args.shrinking)
print args.tol
print type(args.tol)
print args.cache_size
print type(args.cache_size)
print args.verbose
print type(args.verbose)
print args.max_iter
print type(args.max_iter)
print args.random_state
print type(args.random_state)
print args.class_weight
print type(args.class_weight)
'''
if args.C <=0:
	parser.error('-c: C must be float greater or equal to 0')
if args.degree < 0:
	parser.error('-d: degree must be float greater than 0')
if args.gamma < 0:
	parser.error('-g: gamma must be float greater than 0')
if args.tol <= 0:
	parser.error('-o: tol must be float greater than 0')
if args.cache_size <= 0:
	parser.error('-o: cache_size must be float greater than 0')
if args.random_state!= None:
	 if args.random_state<= 0:
		parser.error('-r: The random state must be integer greater than zero')
######################################################################################################################
# obtain train and test data
train = process_data(args.train)
test = process_data(args.test)
x_train = train[1]
y_train = train[2]

#print y_train
x_test = test[1]
y_test = test[2]

#print y_test

######################################################################################################################
#SVM classifiers

clf = svm.SVC(C=args.C, cache_size=args.cache_size, class_weight=args.class_weight, coef0=args.coef, degree=args.degree, gamma=args.gamma, kernel=args.kernel, max_iter=args.max_iter, probability=True, random_state=args.random_state, shrinking=args.shrinking, tol=args.tol, verbose=args.verbose)

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

with open(args.out_file,"w") as f:
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

c = float(count)/float(len(y_label))
print "\n"
print "###########################################################"
print "Correctly Classified Instances :\t%d\t%0.4f%%" %(count,c*100)
print "Incorrectly Classified Instances :\t%d\t%0.4f%%" %(len(y_label) - count,(1-c)*100)
print "Total number of Instances :\t\t%d" %(len(y_label))
print "###########################################################"
print "\n
'''

