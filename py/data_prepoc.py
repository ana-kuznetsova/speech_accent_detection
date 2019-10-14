import argparse
import librosa
import numpy as np
from speech_preproc import STFT
import pandas as pd
import os
#from build_model import fit_model
import matplotlib.pyplot as plt


## DEFINE PARAMS ##
WIN_TYPE = 'hamming'
NUM_CLASSES = 2
NUM_FREQ_SAMPLES = 512
FS = 16000


def load_data(path):
	sig, fs = librosa.load(path, sr=16000, dtype="float32", mono=True)
	return sig

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-j', '--hop_size', help='Hop size', required=True)
	parser.add_argument('-w', '--win_length', help='Window length', required=True)
	parser.add_argument('-t', '--train_data', help='Path to the train data', required=True)
	parser.add_argument('-p', '--test_data', help='Path to test data', required=True)

	args = parser.parse_args()

	# Do stft preproc
	HOP_SIZE = int(args.hop_size)
	WIN_LENGTH = int(args.win_length)

	global_path = 'D:/DLSP/Project/spa_data/'

	train_py_path = r"D:/DLSP/Project/spa_data/train_stfts/"
	test_py_path = r"D:/DLSP/Project/spa_data/test_stfts/"

	train_lab_path = r"D:/DLSP/Project/spa_data/train_labels/"
	test_lab_path = r"D:/DLSP/Project/spa_data/test_labels/"


	# Preproc test data
	test_df = pd.read_csv(global_path + args.test_data, sep='\t')
	test_filenames = test_df['path'].values
	
	print("Reading test files...")
	test_speech = [load_data(os.path.join(global_path,"clips",p)) for p in test_filenames]

	max_test_len = 0
	for x in test_speech:
		if len(x) > max_test_len:
			max_test_len = len(x)
	
	test_speech = [np.array(list(x) + [0 for i in range(max_test_len - len(x))]) for x in test_speech]
	
	print("one hot encoding the train labels...")
	test_labels = [[1, 0] if val=='la' else [0, 1] for val in test_df['binary'].values]
	test_labels =  np.array(test_labels)
	
	print("Converting test files to STFTs...")
	stfts_test = np.array([STFT(i, WIN_TYPE, WIN_LENGTH, HOP_SIZE, NUM_FREQ_SAMPLES, FS) for i in test_speech])
	print("Conversion finished...starting RNN")
	
	del(test_speech)

	for i, x in enumerate(stfts_test):
		np.save(test_py_path + str(i) + ".npy", x)
		np.save(test_lab_path + str(i) + ".npy", test_labels[i])

	train_df = pd.read_csv(global_path + args.train_data, sep='\t')
	train_filenames = train_df['path'].values
	
	print("Reading train files...")
	train_speech = [load_data(os.path.join(global_path, "clips", p)) for p in train_filenames]
	
	max_train_len = 0
	for x in train_speech:
		if len(x) > max_train_len:
			max_train_len = len(x)
	
	train_speech = [np.array(list(x) + [0 for i in range(max_train_len - len(x))]) for x in train_speech]
	
	print("one hot encoding the train labels...")
	#Convert true values to 0 and 1
	train_labels = [[1, 0] if val=='la' else [0, 1] for val in train_df['binary'].values]
	train_labels =  np.array(train_labels)
	
	print("Converting train files to STFTs...")
	# Stfts of train data
	stfts_train = np.array([STFT(i, WIN_TYPE, WIN_LENGTH, HOP_SIZE, NUM_FREQ_SAMPLES, FS) for i in train_speech])
	print("Conversion finished...")
	
	del(train_speech)

	for i, x in enumerate(stfts_train):
		np.save(train_py_path + str(i) + ".npy", x)
		np.save(train_lab_path + str(i) + ".npy", train_labels[i])


	#model, model_hist = fit_model(stfts_train, train_labels, NUM_FREQ_SAMPLES, args.epochs)
	#model.save(global_path + 'base_model.h5')


if __name__=='__main__':   
	main()