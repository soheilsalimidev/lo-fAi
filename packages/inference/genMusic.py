
import os
import random
from midi2audio import FluidSynth
from src.tokenizer.midi_util import VocabConfig, convert_str_to_midi
from pydub import AudioSegment

soundfont_path = './soundFont/florestan-piano.sf2'


presets = {
    1: {

    },
    2: {

    },
    3: {

    }
}


SECOND = 1000


class GenMusic:
    def __init__(self, data):
        cfg = VocabConfig.from_json(
            "./../model/src/tokenizer/vocab_config.json")
        tempo = random.randint(14, 18) * 5
        self.midi = convert_str_to_midi(cfg, ' '.join(
            map(str, data)), tempo).save("./soundFont/mdiOut.mid")
        fs = FluidSynth(
            "./soundFont/SGM-v2.01-NicePianosGuitarsBass-V1.2.sf2")
        fs.midi_to_audio('./soundFont/mdiOut.mid', './soundFont/output.flac')
        self.pianoRoll = AudioSegment.from_file("./soundFont/output.flac")
        fillName = random.choice(os.listdir(
            "./loops/vinyl")) if random.randint(0, 10) > 5 else random.choice(os.listdir(
                "./loops/vinyl"))
        self.fill = AudioSegment.from_file(f"./loops/vinyl/{fillName}")
        drum = random.choice(os.listdir(
            f"./loops/drumloop{tempo}"))
        self.drum = AudioSegment.from_file(f"./loops/drumloop{tempo}/{drum}")

    def mix_lines(self, music_len=60):
        # drums_line = self.drum[5*SECOND: (music_len + 5)*SECOND]
        # piano_line = self.pianoRoll[: music_len*SECOND]
        # noise_line = self.fill[: music_len*SECOND]

        # piano = piano_line.low_pass_filter(35)
        # piano = piano + 18

        # noise = noise_line - 15

        # piano = piano.fade_in(4*SECOND)
        # noise = noise.fade_in(10*SECOND)

        music = self.fill.overlay(
            self.pianoRoll,
            # position=10*SECOND
        ).overlay(
            self.drum,
            # position=15*SECOND
        )

        # music = music.fade_out(8*SECOND)

        return music
