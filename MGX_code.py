def getmetadata(filename):
 import librosa
 import numpy as np
 y, sr = librosa.load(filename, sr=None)
 
 #fetching tempo
 onset_env = librosa.onset.onset_strength(y=y, sr=sr)
 # Extract features using librosa
 tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
 chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
 rmse = librosa.feature.rms(y=y)
 spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
 spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
 rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
 zcr = librosa.feature.zero_crossing_rate(y)
 mfcc = librosa.feature.mfcc(y=y, sr=sr)
 metadata_dict = 
{'tempo':tempo,'chroma_stft':np.mean(chroma_stft),'rmse':np.mean(rmse),
 
'spectral_centroid':np.mean(spec_cent),'spectral_bandwidth':np.mean(spec_bw), 
 'rolloff':np.mean(rolloff), 'zero_crossing_rates':np.mean(zcr)}
 for i in range(1,21):
 metadata_dict.update({'mfcc'+str(i):np.mean(mfcc[i-1])})
 return list(metadata_dict.values())