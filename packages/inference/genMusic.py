from src.tokenizer.midi_util import VocabConfig, convert_str_to_midi
from pydub import AudioSegment
from pydub.effects import speedup
import math
import os
import random
import wave
import time
import tinysoundfont
import pyaudio
import numpy as np
import fluidsynth
import sys
sys.path.append("./../model")
# from midi2audio import FluidSynth

SECOND = 1000


def relpath(p): return os.path.normpath(
    os.path.join(os.path.dirname(__file__), p))


def midiToWav(font: str, midi: str, prS):
    synth = tinysoundfont.Synth()

    sfid = synth.sfload(font)
    synth.program_select(prS[0], sfid, prS[1], prS[2], prS[3])
    seq = tinysoundfont.Sequencer(synth)
    seq.midi_load(midi)

    s = []
    while not seq.is_empty():
        buffer = synth.generate(44100)
        s = np.append(s, np.frombuffer(bytes(buffer), dtype=np.float32))

    return (s * 32767).astype(np.int16).tobytes()


class GenMusic:
    def __init__(self, data, dataDrum):

        tempo = random.randint(14, 18) * 4 * 3

        self.midi = convert_str_to_midi(" ".join(map(str, data)), tempo, 0).save(
            relpath("./soundFont/mdiOut.mid")
        )

        self.pianoRoll = AudioSegment(
            midiToWav(relpath("./soundFont/OmegaGMGS2.sf2"),
                      relpath("./soundFont/mdiOut.mid"),
                      random.choice([[0, 0, i, False] for i in range(0, 127)])),
            frame_rate=44100,
            sample_width=2,
            channels=1)

        fillName = (
            random.choice(os.listdir(relpath("./loops/vinyl")))
        )
        self.fill = AudioSegment.from_file(
            relpath(f"./loops/vinyl/{fillName}"))

        # handel drum
        convert_str_to_midi(" ".join(map(str, dataDrum)), 130, 9).save(
            relpath("./soundFont/mdiOutDrum.mid")
        )

        self.drum = AudioSegment(
            midiToWav(relpath("./soundFont/FluidR3_GM.sf2"),
                      relpath("./soundFont/mdiOutDrum.mid"), [9, 0, 0, True]),
            frame_rate=44100,
            sample_width=2,
            channels=1
        )

        # self.drum = AudioSegment.from_file(f"./loops/drumloop{tempo}/{drum}")

    def mix_lines(self, music_len=60):
        self.pianoRoll = self.pianoRoll + 10

        self.fill = self.fill * \
            math.ceil(self.pianoRoll.duration_seconds /
                      self.fill.duration_seconds)
        self.drum = self.drum * \
            math.ceil(self.pianoRoll.duration_seconds /
                      self.drum.duration_seconds)
        music = self.pianoRoll.overlay(
            self.fill + 5,
        ).overlay(
            self.drum - 5,
        )
        music = music * math.ceil(music_len / music.duration_seconds)

        music = music.fade_out(2 * SECOND)

        return music
