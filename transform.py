import scipy.io.wavfile
import pydub

def transform(path,filename):
    mp3 = pydub.AudioSegment.from_mp3(path+filename+".mp3")
    mp3.export(path+filename+".wav",format="wav")

if __name__ == "__main__":
    temp_folder = './static/'
    mp3 = pydub.AudioSegment.from_mp3(temp_folder+"tmp"+number+".mp3")
    #convert to wav
    mp3.export(temp_folder+"file"+number+".wav", format="wav")