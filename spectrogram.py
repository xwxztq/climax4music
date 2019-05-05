import numpy as np
# from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks



def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(
        samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(
                spec[:, int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs


""" plot spectrogram"""


def plotstft(audiopath, binsize=2**12, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    tracks = len(samples.shape)

    if(tracks == 2):
        samples = samples.sum(axis=1) / 2

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel

    
    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    # plt.figure(figsize=(15, 7.5))
    # plt.imshow(np.transpose(ims), origin="lower", aspect="auto",
    #             cmap=colormap, interpolation="none")
    # plt.colorbar()
    #
    # plt.xlabel("time (s)")
    # plt.ylabel("frequency (hz)")
    # plt.xlim([0, timebins-1])
    # plt.ylim([0, freqbins])
    #
    # xlocs = np.float32(np.linspace(0, timebins-1, 5))
    # plt.xticks(xlocs, ["%.02f" % l for l in (
    #     (xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    # ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    # plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    #
    # if plotpath:
    #     plt.savefig(plotpath, bbox_inches="tight")
    # else:
    #     plt.show()

    # plt.clf()

    return ims

def get_that(path,filename):
    outfile = open(path+filename+".txt","w",encoding="utf-8")
    filepath = path + filename +".wav"
    ims = plotstft(filepath,plotpath=path+filename)
    samplerate, samples = wav.read(filepath)
    timelength = samples.shape[0]/samplerate
    timebins,freqbins = np.shape(ims)
    blockPerSecond = 1.*timebins/timelength
    blocks = int(blockPerSecond * min(30,timelength) )

    print("timebins",timebins)
    print("freqbins",freqbins)
    print("timelength",timelength)
    print("block per second",blockPerSecond)
    valarray = []
    blockArray= []
    xValArray = []

    for i in range(0,timebins):
        val = 0
        for j in range(0,freqbins):
            val = val + (max(0.,ims[i][j])**2)*((j+1)**3)
        valarray.append(val)
        xValArray.append(i/blockPerSecond)
    
    # plt.plot(xValArray,valarray)

    mx = 0
    mxend = blocks
    for i in range(0,timebins):
        if i>0:
            valarray[i] = valarray[i-1]+valarray[i]
        #print("valarray[i]",valarray[i])
        if i>blocks :
            blockArray.append(valarray[i]-valarray[i-blocks])
            if blockArray[-1]>mx:
                mx = blockArray[-1]
                mxend = i
        else :
            blockArray.append(0)

    st = (mxend-blocks)/blockPerSecond 
    ed =  mxend/blockPerSecond
    '''
    plt.savefig(path+filename+"point.png")
    plt.clf()
    plt.plot(xValArray,blockArray)
    plt.savefig(path+filename+"sum.png")
    plt.clf()
    '''
    print(st,ed)
    return st,ed

def process_audio(filename,path="./static/"):
    from . import transform
    transform.transform(path,filename)
    return get_that(path,filename)

if __name__ == "__main__":
    process_audio("tmp")