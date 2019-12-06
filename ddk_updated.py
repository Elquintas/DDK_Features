import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.io.wavfile import read


def log_energy(signal):
	new_signal = np.power(signal,2)
	sum_new_signal = np.sum(new_signal)
	logE = np.log10(np.abs(sum_new_signal))
	return logE



def multi_find(s, r):
    s_len = len(s)
    r_len = len(r)
    _complete = []
    if s_len < r_len:
        n = -1
    else:
        for i in range(s_len):
            # search for r in s until not enough characters are left
            if s[i:i + r_len] == r:
                _complete.append(i)
            else:
                i = i + 1
    return(_complete)




def praat_f0_decode(fileTxt):
    fid=open(fileTxt)
    datam=fid.read()
    end_line1=multi_find(datam, '\n')
    F0=[]
    ji=10
    while (ji<len(end_line1)-1):
        line1=datam[end_line1[ji]+1:end_line1[ji+1]]
        cond=(line1=='2' or line1=='3' or line1=='4' or line1=='5' or line1=='6' or line1=='7' or line1=='8' or line1=='9' or line1=='10' or line1=='11' or line1=='12' or line1=='13' or line1=='14' or line1=='15')
        if (cond): F0.append(float(datam[end_line1[ji+1]+1:end_line1[ji+2]]))
        ji=ji+1
    F0=np.asarray(F0)
    return F0





def decode_Texgrid(file_textgrid, file_audio, type_segm, win_trans=0.04):
    fid=open(file_textgrid)
    data=fid.read()
    fs, data_audio=read(file_audio)
    if (type_segm=='Unvoiced' or type_segm=='Onset'):
        pos=multi_find(data, '"U"')
    elif (type_segm=='Voiced' or type_segm=='Offset'):
        pos=multi_find(data, '"V"')
    segments=[]
    for j in range(len(pos)):
        pos2=multi_find(data[0:pos[j]], '\n')
        nP=len(pos2)
        inicio=data[pos2[nP-3]:pos2[nP-2]]
        fin=data[pos2[nP-2]:pos2[nP-1]]
        inicioVal=int(float(inicio)*fs)
        finVal=int(float(fin)*fs)-1
        if (type_segm=='Onset' or type_segm=='Offset'):
            segments.append(data_audio[int(finVal-win_trans*fs):int(finVal+win_trans*fs)])
        else:
            segments.append(data_audio[inicioVal:finVal])
    return segments, fs

def Power(sig):
    sig2=np.square(sig)
    sumsig2=np.sum(sig2)/len(sig)
    return sumsig2




def DDK_Features(fs, data, audio):
	
	#fs,data = read(audio)
	
	#Signal Normalization
	data = data-np.mean(data)
	data = data/float(np.max(np.abs(data)))
	
	frame_size = 0.02 * fs
	hop_size = 0.01 * fs
	overlap = hop_size/frame_size
	#print('Total number of frames: ', int(len(data)/hop_size))
	
	# F0 Extraction with Praat
	path = './'
	os.system('./Toolkits/praat F0_Praat.praat '+'./wavs/'+audio+' Toolkits/tempF0.txt 75 500 0.02')
	F0 = praat_f0_decode('Toolkits/tempF0.txt')

	
	frame_iterator = int((len(data)/frame_size/overlap))-1
	logE = []
	for i in range(frame_iterator):
		frame = data[int(i*hop_size):int(i*hop_size+frame_size)]
		logE.append(10.0**log_energy(frame))
	logE = np.asarray(logE)
	
	#Energy Features
	avg_energy = 10*np.log10(np.mean(logE))
	std_energy = 10*np.log10(np.std(logE))
	max_energy = 10*np.log10(np.max(logE))
	
	# Fundamental Frequency Features
	avg_F0 = np.mean(F0[F0!=0])
	std_F0 = np.std(F0[F0!=0])
	max_F0 = np.max(F0)
	

	time_step_F0 = 0
	F0_min = 75
	F0_max = 600
	max_vuv_period = 0.02
	average_vuv_period = 0.01
	
	# Extract Voiced and Unvoiced features with Praat
	os.system('./Toolkits/praat '+'vuv.praat'+' '+'./wavs/'+audio+' '+'Toolkits/vuv.txt '
				+str(F0_min)+' '+str(F0_max)+' '+str(time_step_F0)
				+' '+str(max_vuv_period)+' '+str(average_vuv_period))
				
	voiced_segments, fs = decode_Texgrid('Toolkits/vuv.txt', './wavs/'+audio, 'Voiced')
	unvoiced_segments, fs = decode_Texgrid('Toolkits/vuv.txt', './wavs/'+audio, 'Unvoiced')
	nr_voiced = len(voiced_segments)
	nr_unvoiced = len(unvoiced_segments)
	
	
	
	#DDK Features
	DDK_rate = fs*float(nr_voiced)/len(data)
	avg_dur_DDK = 1000*np.mean([len(voiced_segments[i]) 
								for i in range(nr_voiced)])/float(fs)
	regularity_DDK = 1000*np.std([len(voiced_segments[i])
								for i in range(nr_voiced)])/float(fs)							

	
	
	
	#Silence Features
	thr_len_pause = 0.14*float(fs)
	thr_en_pause = 0.2
	silence = []
	for i in range(nr_unvoiced):
		eu = log_energy(unvoiced_segments[i])
		if (eu<thr_en_pause or len(unvoiced_segments[i])>thr_len_pause):
			silence.append(unvoiced_segments[i])

	Sil_rate=fs*float(len(silence))/len(data)
	
	if (len(silence)>0):
		Sil_avg_dur=1000*np.mean([len(silence[k]) 
							for k in range(len(silence))])/float(fs)
		Sil_std_dur=1000*np.std([len(silence[k]) 
							for k in range(len(silence))])/float(fs)
	else:
		Sil_avg_dur=0.0
		Sil_std_dur=0.0
	
	
	
	#Encapsulate
	feature_array = np.array([filename,avg_F0,std_F0,max_F0,avg_energy,std_energy,max_energy,
			DDK_rate,avg_dur_DDK,regularity_DDK,
			Sil_rate,Sil_avg_dur,Sil_std_dur])
						
	feature_array = feature_array.reshape(1,len(feature_array))
	
	return feature_array
 	
	


if __name__=="__main__":
		
	path = "./"
	wav_dir = os.path.join(path,'wavs')
	
	titles = np.array(['Name','avg_F0','std_F0','max_F0','avg_E','std_E',
						'max_E','DDK_rate','avgdur_DDK','reg_DDK',
						'Sil_rate','Sil_avg_dur','Sil_std_dur'])
	titles = titles.reshape(1,len(titles))
	to_file = np.concatenate((titles,), axis=0)
	
	num_files = len([f for f in os.listdir(wav_dir)if os.path.isfile(os.path.join(wav_dir, f))])
	
	bar = tqdm.tqdm(total=num_files,desc='Feature Extraction')
	for filename in os.listdir(wav_dir):
		if filename.endswith('.wav'):
			fs,audio = read(os.path.join(wav_dir,filename))
			features = DDK_Features(fs, audio, filename)
			#print(features)
			to_file = np.concatenate((to_file,features), axis=0)
			bar.update(1)
	
	#Save as a .csv file
	np.savetxt("DDK_Features.csv", to_file, delimiter=",", fmt='%s')
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
