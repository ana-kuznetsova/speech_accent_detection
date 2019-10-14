import librosa
import numpy as np

def expdft(win_length, hopsize, N):
<<<<<<< HEAD
	pow_ = []
	for i in range(N):
		temp = []
		for m in range(i*hopsize, i*hopsize + win_length):
			temp.append(-3.14j*i*m/ N)
		pow_.append(temp)
	return np.exp(pow_)

def STFT(x, win_type, win_length, hopsize, num_freq_samples, fs):
	win_length = fs*win_length//1000
	hopsize = fs*hopsize//1000
	if win_type == 'hamming':
		w = np.hamming(win_length)
	else:
		w = [0*win_length]
	
	num_segments = ((len(x) - win_length)//hopsize) + 2

	#signal segments
	X = np.array([list(x[i*hopsize:i*hopsize+win_length]) for i in range(num_segments)\
					if i*hopsize+win_length < len(x)])
	
	#padding signals on both ends
	padding = np.zeros((win_length))
	X = np.row_stack((padding, X))
	X = np.row_stack((X, padding))
	
	#exp matrix powers
	E = expdft(win_length, hopsize, num_freq_samples)
 
	return np.abs(np.dot(X[1:]*w, E.T)).tolist()

def get_power_response(x):
	return 20*np.log10(np.abs(x)+0.00000001)
=======
    pow_ = []
    for i in range(N):
        temp = []
        for m in range(i*hopsize, i*hopsize + win_length):
            temp.append(-3.14j*i*m/ N)
        pow_.append(temp)
    return np.exp(pow_)

def STFT(x, win_type, win_length, hopsize, num_freq_samples, fs):
    win_length = fs*win_length//1000
    hopsize = fs*hopsize//1000
    if win_type == 'hamming':
        w = np.hamming(win_length)
    else:
        w = [0*win_length]
    
    num_segments = ((len(x) - win_length)//hopsize) + 2

    #signal segments
    X = np.array([list(x[i*hopsize:i*hopsize+win_length]) for i in range(num_segments)\
                    if i*hopsize+win_length < len(x)])

    #padding signals on both ends
    padding = np.zeros((win_length))
    X = np.row_stack((padding, X))
    X = np.row_stack((X ,padding))
    
    #exp matrix powers
    E = expdft(win_length, hopsize, num_freq_samples)
 
    return np.abs(np.dot(X[1:]*w, E.T)).tolist()

def get_power_response(x):
    return 20*np.log10(np.abs(x)+0.00000001)
>>>>>>> d8d5afb97a6fe775b058c6c43bd65f72a897e949
