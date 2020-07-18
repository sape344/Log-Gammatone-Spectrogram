#librosa.version.version=0.7.2
#spafe.sys.version=3.7.6 
import librosa
import numpy as np
import spafe
from librosa import display


def Log_Gammatone_Spectrum(File_name,sr): 
    y,sr=librosa.load(File_name,sr)
    gammatone_filter_bank = spafe.fbanks.gammatone_fbanks.gammatone_filter_banks(nfilts=64, nfft=2048, fs=sr, low_freq=50, high_freq=None, scale='contsant', order=4)
    y=librosa.util.normalize(y)
    magnitude = np.abs(librosa.stft(y,win_length=2048))**2
    Gam=gammatone_filter_bank.dot(magnitude)
    LogGamSpec = librosa.power_to_db(Gam,ref=np.max)
    display.specshow(LogGamSpec,y_axis='log')
    return LogGamSpec
    
 #Example
    
  a=Log_Gammatone_Spectrum("Valse.mp3",44100) #Valse.mp3 is music name, 44100 is sample rate(sr)

