#!/usr/bin/python
import numpy as np
from noter import Noter 
# based on : www.daniweb.com/code/snippet263775.html
# https://stackoverflow.com/a/33913403
import math
import wave
import struct

class Waver:

    def __init__(self, init_samples=None, sample_rate=44100.0, default_ms=500, channels=None):
        self.channels = channels if channels is not None else 1
        if init_samples is None:
            self.samples = []
            for _ in range(self.channels):
                self.samples.append([])
        else:
            self.samples = init_samples.copy()
            self.channels = len(self.samples[0])
        self.sample_rate = sample_rate
        self.default_ms = default_ms

    def append_silence(self, ms=None, channel=0):
        """
        Adding silence is easy - we add zeros to the end of our array
        """
        if ms is None: ms = self.default_ms
        num_samples = ms * (self.sample_rate / 1000.0)

        for _ in range(int(num_samples)): 
            self.samples[channel].append(0.0)

    def append_sinewave(self, note=440.0, ms=None, volume=1.0, channel=0):
        """
        The sine wave generated here is the standard beep.  If you want something
        more aggresive you could try a square or saw tooth waveform.   Though there
        are some rather complicated issues with making high quality square and
        sawtooth waves... which we won't address here :) 
        """ 

        freq = Noter.to_freq(note)
        if freq is 0:
            return self.append_silence(ms, channel)

        if ms is None: ms = self.default_ms
        num_samples = ms * (self.sample_rate / 1000.0)

        for x in range(int(num_samples)):
            self.samples[channel].append(volume * math.sin(2 * math.pi * freq * ( x / self.sample_rate )))

    def append_sinewave_single_channel(self, note=440.0, ms=None, volume=1.0):
        self.append_sinewave(note, ms, volume, channel=0)
        for ch in range(1, self.channels):
            self.append_silence(ms, channel=ch)

    def save_wav(self,  file_name):
        # Open up a wav file
        with wave.open(file_name,"w") as wav_file:
            # wav params
            nchannels = self.channels
            sampwidth = 2

            # 44100 is the industry standard sample rate - CD quality.  If you need to
            # save on file size you can adjust it downwards. The stanard for low quality
            # is 8000 or 8kHz.
            nframes = len(self.samples)
            comptype = "NONE"
            compname = "not compressed"
            wav_file.setparams((nchannels, sampwidth, self.sample_rate, nframes, comptype, compname))

            # WAV files here are using short, 16 bit, signed integers for the 
            # sample size.  So we multiply the floating point data we have by 32767, the
            # maximum value for a short integer.  NOTE: It is theortically possible to
            # use the floating point -1.0 to 1.0 data directly in a WAV file but not
            # obvious how to do that using the wave module in python.
            samples = np.array(self.samples)
            samples = samples.copy()
            samples = np.transpose(samples)
            for channels_sample in samples:
                sample = np.array(list(map(lambda x: int(x*32767.0), channels_sample )))
                wav_file.writeframes(struct.pack('<'+'h'*nchannels, *sample))


class WaveDecoder:

    def __init__(self, chunk_ms=10):
        self.chunk_ms = chunk_ms
        self.sample_rate = None
        self.channels = None
        self.samples = None
        self.nframes = None

        self.decoded_freqs = None
        self.fftfreq = {}
        self.noter = Noter()

    def open_wav(self, filename):
        with wave.open(filename) as wf:
            self.sample_rate = wf.getframerate()
            self.channels = wf.getnchannels()
            nframes = wf.getnframes()
            self.nframes = nframes
            data = wf.readframes(nframes)
            print("WaveDecoder: '%s' cargado: leídos %d frames" % (filename, nframes))

            self.samples = np.frombuffer(data, dtype='<i2')
            if self.channels > 1:
                self.samples = self.samples.reshape(-1, self.channels)                

    def open_waver(self, wv):
        self.channels = wv.channels
        self.sample_rate = wv.sample_rate
        self.samples = wv.samples.copy()
        self.chunk_ms = wv.default_ms
        self.nframes = len(self.samples)

    def get_freq_chunk(self, chunk):
        transform = np.fft.fft(chunk)
        len_fft = len(transform)
        try:
            freqs = self.fftfreq[len_fft]
        except KeyError:
            freqs = np.fft.fftfreq(len_fft)
            self.fftfreq[len_fft] = freqs
        idx_freq = np.argmax(np.abs(transform))
        freq = freqs[idx_freq]
        freq_in_hertz = abs(freq * self.sample_rate)
        return freq_in_hertz

    def decode(self):
        n_samples = int(self.chunk_ms * (self.sample_rate / 1000.0))

        start = 0
        end = n_samples - 1

        self.decoded_freqs = []

        if self.channels > 1:
            for _ in range(self.channels):
                self.decoded_freqs.append([])

        n_chunks = 0
        while start < self.nframes:
            try:
                chunk = self.samples[start:end]
            except IndexError:
                chunk = self.samples[start:]

            if self.channels > 1:
                chunk = np.transpose(chunk)
                for ch in range(self.channels):
                    freq_raw = self.get_freq_chunk(chunk[ch])
                    note = self.noter.get_closest_freq(freq_raw)
                    self.decoded_freqs[ch].append(note)
            else:
                freq_raw = self.get_freq_chunk(chunk)
                note = self.noter.get_closest_freq(freq_raw)
                self.decoded_freqs.append(note)

            start += n_samples
            end += n_samples
            n_chunks += 1
            print("Procesado %d chunk%s" % (n_chunks, 's' if n_chunks > 1 else ''))

        return self.decoded_freqs

