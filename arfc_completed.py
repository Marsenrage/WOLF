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
from sklearn import metrics
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier

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
parser = argparse.ArgumentParser(prog='Random forest classifier', usage='%(prog)s [options]')
parser.add_argument("-T", type=str, dest = 'train', required = True,
                   help="training data set")
parser.add_argument("-t", type=str, dest = 'test', required = True,
                    help="testing data set")
parser.add_argument("-O", type=str, dest = 'out_file',required = True,
                    help="output predictions file name")

parser.add_argument("-n", type=int, dest = 'numOfTrees',
                    help="The number of trees in the forest.default= 10", default= 10 )

parser.add_argument("-d", type=int, dest = 'maxDepth',
                    help="The maximum depth of the tree.default= None ", default= None )

parser.add_argument("-r", type=int, dest = 'randomState',
                    help="The random number seed to be used.default= None", default= None )

parser.add_argument("-j", type=int, dest = 'njobs',
                    help="The number of jobs to run in parallel for both fit and predict.default= 1 ", default= 1 )

parser.add_argument("-f", dest = 'features', help="the number of features to consider when looking for the best split.default= auto.value could be int, float and ['auto', 'log2', 'sqrt']", default= 'auto', type = str )

args = parser.parse_args()

# validation -f
if re.match("^\d+?\.\d+$", args.features)or re.match("^\.\d+$", args.features):
	features = float(args.features)
	if not 0<features<1:
		parser.error('floats must between (0,1)')
elif re.match("^\d+?$", args.features):

	features = int(args.features)
elif re.match("^\-\d+?$", args.features):
	parser.error('value should not be less than 0')
elif args.features not in ['auto', 'log2', 'sqrt']: 
	parser.error('value should be select from \'auto\', \'log2\' or \'sqrt\'')
else:
	features = args.features

# validation -n
if args.numOfTrees<= 0:
	parser.error('-n: The number of trees in the forest must be integer greater than zero')

# validation -d
if args.maxDepth!= None:
	 if args.maxDepth<= 0:
		parser.error('-d: The maximum depth of the tree must be integer greater than zero')

# validation -r 
if args.randomState!= None:
	 if args.randomState<= 0:
		parser.error('-r: The random state must be integer greater than zero')

# validation -j 
if args.njobs< -1:
	parser.error('-j: The number of jobs to run must be -1 or integer greater than 0')
if args.njobs==0:
	parser.error('-j: The number of jobs to run must be -1 or integer greater than 0')
'''
print args.numOfTrees
print type(args.numOfTrees)
print args.maxDepth
print type(args.maxDepth)
print args.randomState
print type(args.randomState)
print args.njobs
print type(args.njobs)
print args.features
print type(args.features)
'''

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
#classifiers

clf = RandomForestClassifier(n_estimators=args.numOfTrees, max_depth=args.maxDepth,	min_samples_split=1, random_state=args.randomState, max_features = features, n_jobs = args.njobs )

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
#test whether the sum of the prob of each class in ith entry is equal to 1
for i in range(len(x_test)):

		a= float(p[i][0])+ float(p[i][1])+ float(p[i][2])+ float(p[i][3])+ float(p[i][4])+ float(p[i][5])+ float(p[i][6])+ float(p[i][7])+ float(p[i][8])+ float(p[i][9])
		print a

#test results
count = 0 
for i in range (len(y_label)):
	if y_label[i] == y_test[i]:
		count += 1

c = float(count)/float(len(y_label))
print "\n"
print "###########################################################"
print "Correctly Classified Instances :\t%d\t%0.4f%%" %(count,c*100)
print "Incorrectly Classified Instances :\t%d\t%0.4f%%" %(len(y_label) - count,(1-c)*100)
print "Total number of Instances :\t\t%d" %(len(y_label))
print "###########################################################"
print "\n"
'''
