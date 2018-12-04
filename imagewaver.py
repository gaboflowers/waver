"""
(c) G. Flores 2018, where applicable
"""
from waver import Waver, WaveDecoder
from noter import used_notes, used_notes_by_freq, used_freqs
from PIL import Image
import numpy as np
import itertools, collections

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
        try:
            formato = intmode_by_mode[mode]
        except KeyError:
            print("No prometo nada")
        nota_formato = self.uint6_to_note(formato)
        self.wv.append_sinewave_single_channel(nota_formato)
        #self.wv.append_sinewave_single_channel(nota_formato)
        print("Canal: %d - Modo %s - Nota: %.2f (L), [mute] (R)" % (0, mode, nota_formato))

        # Tamaño
        for ch in [0, 1]:
            a, b, c = self.uint16_to_note_triplet(self.image.size[ch])
            self.wv.append_sinewave(a, channel=ch)
            self.wv.append_sinewave(b, channel=ch)
            self.wv.append_sinewave(c, channel=ch)
            print("Canal: %d - Notas: %.2f, %.2f, %.2f" % (ch, a, b, c))

    def encode(self):
        if self.image is None:
            raise Exception("No hay imagen para codificar")

        self._encode_header()

        for c in range(len(self.channels)):
            print("Codificando el canal (de color) %d" % c)
            matriz = self.channels[c]
            pixel_old = None

            width = self.image.size[0]
            height = self.image.size[1]
            print("height, width: %s, %s" % tuple((height, width)))
            for i, j in itertools.product(range(height), range(width)):
                pixel = matriz[i][j]
            
                if j == width-1 and pixel_old is None: # fila de largo impar
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

        self.wv.append_silence(ms=500, channel=0) # Por si se trunca
        self.wv.append_silence(ms=500, channel=1)

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
        print("c2n - color %d - note %.2f" % (n, used_notes[n][1]))
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


class WaveImageDecoder(WaveDecoder):

    def __init__(self, chunk_ms=10):
        super().__init__(chunk_ms)
        self.image = None
        self.matrix = None
        self.width = None
        self.height = None
        self.mode = None
        self.decoded_pairs = None
        self.layers = 0
    
    def decode(self):
        decoded_freqs = super().decode()
        self.decoded_pairs = list(zip(*decoded_freqs))
        return decoded_freqs

    def decode_header(self):
        chunk_modo = self.decoded_freqs[0][0]
        chunk_size = self.decoded_pairs[1:4]

        modo = self.note_to_uint6(chunk_modo)
        self.mode = mode_by_intmode[modo]
        print("Modo: '%s'" % self.mode)
        self.layers = layers_by_mode[self.mode]

        width_triplet = [x[0] for x in chunk_size]
        height_triplet = [x[1] for x in chunk_size]

        self.width = self.note_triplet_to_uint16(*width_triplet)
        self.height = self.note_triplet_to_uint16(*height_triplet)
        print("Dimensiones: {}".format((self.width, self.height)))

        self.matrix = np.zeros((self.layers, self.height, self.width))

    def _valid_mode(self):
        return self.mode in layers_by_mode.keys()

    def to_image(self):
        if self.image is not None:
            return self.image

        if self.decoded_freqs is None:
            raise Exception("Samples not decoded. Try decode().")

        self.decode_header()
        if not self._valid_mode():
            raise Exception("Invalid mode: {}.".format(self.mode))

        data_pairs = collections.deque(self.decoded_pairs[4:].copy())
        for c in range(self.layers):
            this_layer = self.matrix[c]
            data_tuple = None
            for i, j in itertools.product(range(self.height), range(self.width)):
                parity = j%2
                if parity == 0:
                    data_tuple = data_pairs.popleft()
                    #print("({}) data_tuple: {}".format((i,j),data_tuple))

                freq_pixel = data_tuple[parity]
                pixel_value = self.note_to_color(freq_pixel)
                this_layer[i][j] = pixel_value
                #print("({}) | freq: {} | value: {}".format((i,j),freq_pixel, pixel_value))

        print("Quedaron %d elementos sin usar en la cola" % len(data_pairs))
        if self.layers == 1:
            self.matrix = self.matrix[0]

        self.matrix = self.matrix.astype(np.uint8)
        self.image = Image.fromarray(self.matrix, self.mode)
        return self.image

    def save_image(self, filename):
        if self.image is None:
            raise Exception("Image not created. Try to_image().")
        self.image.save(filename)

    @staticmethod
    def note_to_color(note):
        """
        note es una frecuencia entre C1 y A#7. Se toma el índice de las 84
        alternativas, y se multiplica por 3 para retornar un color
        (entre 0 y 249, con un punto mentiroso al 255)

        retorna la frecuencia (0 = silencio)
        """
        if note == 0: return 255

        n = used_freqs.index(note)
        print("n2c - note %.2f - color %d" % (note, n*3))

        return n*3

    @staticmethod
    def note_to_uint6(note):
        """
        note es una frecuencia entre C1 y A#7

        retorna un número entre 0 y 83
        """
        return used_freqs.index(note)

    @staticmethod
    def note_triplet_to_uint16(note_a, note_b, note_c):
        """
        n es un entero entre 0 y 65536 (bueno, hasta 592703)
        n se codifica en 3 notas (12*7 = 84 notas distintas), dando 84**3 
        valores posibles

        retorna la frecuencia de las 3 notas que corresponden al numero
        """

        a = used_freqs.index(note_a)
        b = used_freqs.index(note_b)
        c = used_freqs.index(note_c)

        n = a*84 + b
        n *= 84
        n += c

        return n

        

modos_ints = [('L', 1), ('RGB', 2), ('RGBA', 3)]
intmode_by_mode = dict(modos_ints)
mode_by_intmode = dict([(x[1],x[0]) for x in modos_ints])

layers_by_mode = {'L': 1, 'RGB': 3, 'RGBA': 4}
