# waver
Python simple WAV audio synthesizer and information encoder/decoder

Soon to be a ridiculous way to encode images.

## Dependencies
- Python 3.x
- numpy
- Pillow (yes, the Imaging library, but only for image encoding)

## Example
```
>>> from waver import *
>>> from noter import *
>>> wd = WaveDecoder(chunk_ms=100)
>>> wd.open_wav("test_encode100.wav")
WaveDecoder: 'test_encode100.wav' cargado: leídos 30870 frames
>>> dec = wd.decode()
Procesado 1 chunk
Procesado 2 chunk
Procesado 3 chunk
Procesado 4 chunk
Procesado 5 chunk
Procesado 6 chunk
Procesado 7 chunk
>>> noter = Noter()
>>> list(map(noter.to_note, dec[0]))
['B0', 'B0', 'B0', 'D#1', 'B0', 'F#6', 'F#6']
```
