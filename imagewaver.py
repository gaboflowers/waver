from waver import Waver, WaveDecoder
from noter import used_notes
from PIL import Image
import numpy as np
import itertools

class WaveImageEncoder:
    
    def __init__(self, default_ms=10):
        self.channels = None
        self.image = None
        self.default_ms = default_ms
        self.wv = None

    def open(self, filename):
        self.image = Image.open(filename)
        self.channels = list(self.image.split())

        for i in range(len(self.channels)):
            self.channels[i] = np.array(self.channels[i])

        self.wv = Waver(default_ms=self.default_ms, channels=2)

    def _encode_header(self):
        # Formato (1: grises, 2: rgb, 3: rgba)
        mode = self.image.mode
        print("Modo: %s" % mode)
        formato = 0
        if mode == 'L':
            formato = 1
        elif mode == 'RGB':
            formato = 2
        elif mode == 'RGBA':
            formato = 3
        else:
            print("No prometo nada")
        nota_formato = self.uint6_to_note(formato)
        self.wv.append_sinewave_single_channel(nota_formato)
        print("Canal: %d - Modo %s - Nota: %d (L), [mute] (R)" % (0, mode, nota_formato))

        # Tamaño
        for i in [0, 1]:
            a, b, c = self.uint16_to_note_triplet(self.image.size[i])
            self.wv.append_sinewave(a, channel=i)
            self.wv.append_sinewave(b, channel=i)
            self.wv.append_sinewave(c, channel=i)

    def encode(self):
        if self.image is None:
            raise Exception("No hay imagen para codificar")

        self._encode_header()

        for c in range(len(self.channels)):
            print("Codificando el canal %d" % c)
            matriz = self.channels[c]
            pixel_old = None

            width = self.image.size[0]
            height = self.image.size[1]
            print("height, width: %s, %s" % tuple((height, width)))
            for i, j in itertools.product(range(height), range(width)):
                pixel = matriz[i][j]
            
                if j == width and pixel_old is None: # fila de largo impar
                    pixel_old = pixel
                    # este pixel va a estar duplicado
                    
                if pixel_old is not None:
                    freq_l = self.color_to_note(pixel_old)
                    freq_r = self.color_to_note(pixel)
                    print("Canal: %d - Pixel %s,%s - Notas: %d (L), %d (R)" % (c, i, j, freq_l, freq_r))
                    self.wv.append_sinewave(freq_l, channel=0)
                    self.wv.append_sinewave(freq_r, channel=1)
                    pixel_old = None
                else:
                    pixel_old = pixel

        self.wv.append_silence(ms=500) # Por si se trunca

    def save_wav(self, filename):
        self.wv.save_wav(filename)

    @staticmethod
    def color_to_note(n):
        """
        n es un color entre 0 y 255
        n se divide en 3 (0 - 85) y se mapea en una frecuencia entre C1 y A#7

        retorna la frecuencia (0 = silencio)
        """

        n//=3 # dividir entre 3

        if n == 85: return 0
        if n > 83: n = 83 # justo me faltó 1 nota
        return used_notes[n][1]

    @staticmethod
    def uint6_to_note(n):
        """
        n es un numero entre 0 y 64 (bueno, hasta 83). Se codifica en una sola
        nota

        retorna la frecuencia
        """
        return used_notes[n][1]

    @staticmethod
    def uint16_to_note_triplet(n):
        """
        n es un entero entre 0 y 65536 (bueno, hasta 592703)
        n se codifica en 3 notas (12*7 = 84 notas distintas), dando 84**3 
        valores posibles

        retorna la frecuencia de las 3 notas que corresponden al numero
        """

        c = n % 84
        n = n // 84
        b = n % 84
        n = n // 84
        a = n % 84

        return used_notes[a][1], used_notes[b][1], used_notes[c][1]


class WaveImageDecoder:
    
    def __init__(self):
        pass
