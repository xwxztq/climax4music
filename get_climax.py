# -*-coding:utf-8-*-
import os
import scipy.io.wavfile as wav
import numpy as np
import audio_transfer
import spectrogram


def get_that(path, filename):
    # outfile = open(path+filename+".txt","w",encoding="utf-8")
    filepath = path + filename + ".wav"
    ims = spectrogram.plotstft(filepath, plotpath=path + filename)
    samplerate, samples = wav.read(filepath)
    timelength = samples.shape[0] / samplerate
    timebins, freqbins = np.shape(ims)
    blockPerSecond = 1. * timebins / timelength
    blocks = int(blockPerSecond * min(30, timelength))

    # print("timebins",timebins)
    # print("freqbins",freqbins)
    # print("timelength",timelength)
    # print("block per second",blockPerSecond)

    valarray = []
    blockArray = []
    xValArray = []

    for i in range(0, timebins):
        val = 0
        for j in range(0, freqbins):
            val = val + (max(0., ims[i][j]) ** 2) * ((j + 1) ** 3)
        valarray.append(val)
        xValArray.append(i / blockPerSecond)

    # plt.plot(xValArray,valarray)

    mx = 0
    mxend = blocks
    for i in range(0, timebins):
        if i > 0:
            valarray[i] = valarray[i - 1] + valarray[i]
        # print("valarray[i]",valarray[i])
        if i > blocks:
            blockArray.append(valarray[i] - valarray[i - blocks])
            if blockArray[-1] > mx:
                mx = blockArray[-1]
                mxend = i
        else:
            blockArray.append(0)

    st = (mxend - blocks) / blockPerSecond
    ed = mxend / blockPerSecond
    '''
    plt.savefig(path+filename+"point.png")
    plt.clf()
    plt.plot(xValArray,blockArray)
    plt.savefig(path+filename+"sum.png")
    plt.clf()
    '''
    print(st, ed)
    return st, ed


def process_audio(filepath, path=""):
    # process the audio with some pre-work and calculate the result

    prepath, filename = os.path.split(filepath)
    purename, the_format = filename.split('.')


    if path == "":
        path = prepath

    if the_format == "mp3":

        try:
            audio_transfer.mp3_transform(prepath, purename)
        except:
            raise Exception("Error:file transform failed using mp3_transform")

    elif the_format == "wav":

        pass

    else:

        try:
            audio_transfer.other_transform(prepath, prepath, the_format)
        except:
            raise Exception("Error:file transfor failed using other_transform")

    return get_that(prepath, purename)


if __name__ == '__main__':
    process_audio("D:\example\signal_process\static\Kelly Clarkson - Catch My Breath.mp3")