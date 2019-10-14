import argparse
import numpy as np
import os
from build_model import fit_model
from sklearn.preprocessing import normalize


def main():

	NUM_FREQ_SAMPLES = 512

	parser = argparse.ArgumentParser()
	parser.add_argument('-tr', '--train_data', help='Path to the train data', required=True)
	parser.add_argument('-trl', '--train_label', help='Path to the train labels', required=True)
	parser.add_argument('-te', '--test_data', help='Path to test data', required=True)
	parser.add_argument('-tel', '--test_label', help='Path to the train labels', required=True)
	parser.add_argument('-v', '--val_data', help='Path to validation data')
	parser.add_argument('-vl', '--val_label', help='Path to the train labels')
	parser.add_argument('-n', '--epochs', help='Number of epochs', required=True)

	args = parser.parse_args()
	global_path = r'D:/DLSP/Project/spa_data/'

	trainDataFolder = os.fsencode(global_path + args.train_data)
	testDataFolder = os.fsencode(global_path + args.test_data)
	trainLabFolder = os.fsencode(global_path + args.train_label)
	testLabFolder = os.fsencode(global_path + args.test_label)
	
	print("Reading stfts....")
	stfts_train = np.array([np.load(global_path + args.train_data + os.fsdecode(x)) for x in os.listdir(trainDataFolder)])
	stfts_test = np.array([np.load(global_path + args.test_data + os.fsdecode(x)) for x in os.listdir(testDataFolder)])

	print("Reading labels....")
	train_labels = np.array([np.load(global_path + args.train_label + os.fsdecode(x)) for x in os.listdir(trainLabFolder)])
	test_labels = np.array([np.load(global_path + args.test_label + os.fsdecode(x)) for x in os.listdir(testLabFolder)])
	
	#test = np.array([x[:359,] for x in stfts_test]) 
	

	X_train = np.array([normalize(x) for x in stfts_train])
	X_test = np.array([normalize(x) for x in stfts_test])


	#print(test.shape, test[0].shape)

	model, model_hist = fit_model(X_train, train_labels, X_test, test_labels,\
															 NUM_FREQ_SAMPLES, int(args.epochs))
	model.save(global_path + 'base_model.h5')
	

if __name__=='__main__':   
	main()