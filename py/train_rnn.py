import argparse
<<<<<<< HEAD
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
=======
import librosa
import numpy as np
from speech_preproc import STFT
import pandas as pd
import os
from build_model import fit_model


## DEFINE PARAMS ##
WIN_TYPE = 'hamming'
NUM_CLASSES = 2
NUM_FREQ_SAMPLES = 512
F_S = 16000


def load_data(path):
    return librosa.load(path, sr=16000)


def main():
    if __name__=="__main__":

        parser = argparse.ArgumentParser()
        parser.add_argument('-h', '--hop_size', help='Hop size', required=True)
        parser.add_argument('-w', '--win_length', help='Window length', required=True)
        parser.add_argument('-t', '--train_data', help='Path to the train data', required=True)
        parser.add_argument('-p', '--test_data', help='Path to test data', required=True)
        parser.add_argument('-v', '--val_data', help='Path to validation data')

        args = parser.parse_args()

        # Do stft preproc
        HOP_SIZE = args.hop_size
        WIN_LENGTH = args.win_length

        train_df = pd.read_csv(args.train_data, sep='\t')
        train_paths = train_df['path'].values

        global_path = '/N/u/anakuzne/Carbonate/accent/spa_data/'
        
        train_speech = [load_data(os.path.join(global_path, p)) for p in train_paths]

        #Convert true values to 0 and 1
        train_labels = [[1, 0] if val=='la' else [0, 1] for val in train_df['binary'].values]
        train_labels =  np.array(train_labels)

        # Stfts of train data
        stfts_train = np.array([STFT(i, WIN_TYPE, WIN_LENGTH, HOPSIZE, NUM_FREQ_SAMPLES, FS) for i train_speech])

        # Preproc test data
        test_df = pd.read_csv(args.test_data, sep='\t')
        test_paths = test_df['path'].values

        test_speech = [load_data(os.path.join(global_path, p)) for p in test]

        test_labels = [[1, 0] if val=='la' else [0, 1] for val in test_df['binary'].values]
        test_labels =  np.array(test_labels)

        stfts_test = np.array([STFT(i, WIN_TYPE, WIN_LENGTH, HOPSIZE, NUM_FREQ_SAMPLES, FS) for i test_speech])







main()
>>>>>>> d8d5afb97a6fe775b058c6c43bd65f72a897e949
