import os 
import numpy as np 
import librosa 
import pandas as pd


'''Features to be extracted'''
'''
1.Audio Data Mean
2.Audio Data Variation
3.Zero Crossing Rate Mean
4.Zero Crossing Rate Variation
5.Spectral Centroid Mean
6.Spectral Centroid Variation
7.Spectral Rolloff Mean
8.Spectral Rolloff Variation
9.MFCC - Mel Frequency Cepstral Coefficients Mean
10.MFCC - Mel Frequency Cepstral Coefficients Variation  
11.Chroma STFT Mean
12.Chroma STFT Variation

'''


data = []
modes = ['AnchiHoye','Bati','Ambassel','Tizita']
for scales in modes:

	for audio_file in os.listdir(scales):
		if audio_file.lower().endswith((".wav")):
			x,sr = librosa.load(os.getcwd() + "/" + scales + "/" + audio_file)



			zero_crossing_rate = librosa.zero_crossings(x)
			spectral_centroid = librosa.feature.spectral_centroid(x, sr=sr)[0]
			spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)[0]
			mfccs = librosa.feature.mfcc(x, sr=sr)
			chroma_stft = librosa.feature.chroma_stft(x, sr=sr)


			ad_mean = np.mean(x)
			ad_var = np.var(x)
			zero_crossing_rate_mean = np.mean(zero_crossing_rate)
			zero_crossing_rate_var = np.var(zero_crossing_rate)
			spectral_centroid_mean = np.mean(spectral_centroid)
			spectral_centroid_var = np.var(spectral_centroid)
			spectral_rolloff_mean = np.mean(spectral_rolloff)
			spectral_rolloff_var = np.var(spectral_rolloff)
			mfccs_mean = np.mean(mfccs)
			mfccs_var = np.var(mfccs)
			chroma_stft_mean = np.mean(chroma_stft)
			chroma_stft_var = np.var(chroma_stft)
			data.append((audio_file,
				ad_mean,ad_var,
				zero_crossing_rate_mean,zero_crossing_rate_var,
				spectral_centroid_mean,spectral_centroid_var,
				spectral_rolloff_mean,spectral_rolloff_var,
				mfccs_mean,mfccs_var,chroma_stft_mean,chroma_stft_var,scales))


df = pd.DataFrame(data = data, columns = ['filename','audio_data_mean','audio_data_var','zcr_mean','zcr_var','sc_mean','sc_var','srolloff_mean','srolloff_var','mfcc_mean','mfcc_var','chroma_mean','chroma_var','scale'])
df.to_csv('features.csv',index=False)
