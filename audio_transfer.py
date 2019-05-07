# -*-coding:utf-8-*-
import scipy.io.wavfile
import pydub
import os


def mp3_transform(path, filename):
    # transfer mp3 file to wav

    path_file = os.path.join(path, filename)
    mp3 = pydub.AudioSegment.from_mp3(path_file + ".mp3")
    mp3.export(path_file + ".wav", format="wav")


def other_transform(path, filename, the_format):
    # transfer other format file to wav

    path_file = os.path.join(path, filename)
    try:
        file = pydub.AudioSegment.from_file(path_file + the_format, format=the_format)
    except:
        raise Exception("Error: File format transform failed")

    file.export(path_file + ".wav", format="wav")


if __name__ == "__main__":
    temp_folder = './static/'
    mp3 = pydub.AudioSegment.from_mp3(temp_folder + "tmp" + number + ".mp3")
    # convert to wav
    mp3.export(temp_folder + "file" + number + ".wav", format="wav")
