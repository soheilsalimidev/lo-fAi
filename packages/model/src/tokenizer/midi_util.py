import json
import random
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor, log
from typing import Dict, Iterator, List, Optional, Tuple

import mido


@dataclass
class VocabConfig:
    # Number of note events. Should be 128.
    note_events: int
    # Number of wait events. Configurable, must evenly divide max_wait_time.
    wait_events: int
    # Max wait time in milliseconds to be represented by a single token.
    max_wait_time: int
    # Number of velocity events. Should be 128 (or 100? need to check midi standard)
    velocity_events: int
    # Number of bins to quantize velocity into. Should evenly divide velocity_events.
    velocity_bins: int
    # Exponential scaling factor for velocity bin sizes. 1.0 = linear scaling.
    velocity_exp: float
    # Whether to sort tokens by instrument, note. This should improve data reducibility.
    do_token_sorting: bool
    # Whether tokens should be represented as combined instrument/note/velocity tokens, or separate tokens for each.
    unrolled_tokens: bool
    # If non-zero, notes held for this many seconds will be automatically released during str->midi decoding.
    decode_end_held_note_delay: float
    # If true, repeated notes will be automatically released before playing again during str->midi decoding.
    decode_fix_repeated_notes: bool
    # Manual override for velocity bins. Each element is the max velocity value for that bin by index.
    velocity_bins_override: Optional[List[int]] = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.max_wait_time % self.wait_events != 0:
            raise ValueError("max_wait_time must be exactly divisible by wait_events")
        if self.velocity_bins < 2:
            raise ValueError("velocity_bins must be at least 2")
        if self.velocity_bins_override:
            print("VocabConfig is using velocity_bins_override. Ignoring velocity_exp.")
            if len(self.velocity_bins_override) != self.velocity_bins:
                raise ValueError(
                    "velocity_bins_override must have same length as velocity_bins"
                )
        if self.velocity_exp <= 0:
            raise ValueError("velocity_exp must be greater than 0")

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            config = json.load(f)
        return cls(**config)


class VocabUtils:
    def __init__(self, cfg: VocabConfig) -> None:
        self.cfg = cfg

    @lru_cache(maxsize=128)
    def format_wait_token(self, wait: int) -> str:
        return f"t{wait}"

    @lru_cache(maxsize=128)
    def format_note_token(self, note: int, velocity_bin: int) -> str:
        return f"{note:x}:{velocity_bin:x}"

    def format_unrolled_note(self, note: int) -> str:
        return f"n{note:x}"

    def format_unrolled_velocity(self, velocity_bin: int) -> str:
        return f"v{velocity_bin:x}"

    def velocity_to_bin(self, velocity: float) -> int:
        velocity = max(0, min(velocity, self.cfg.velocity_events - 1))
        if self.cfg.velocity_bins_override:
            for i, v in enumerate(self.cfg.velocity_bins_override):
                if velocity <= v:
                    return i
            return 0
        binsize = self.cfg.velocity_events / (self.cfg.velocity_bins - 1)
        if self.cfg.velocity_exp == 1.0:
            return ceil(velocity / binsize)
        else:
            return ceil(
                (
                    self.cfg.velocity_events
                    * (
                        (
                            self.cfg.velocity_exp
                            ** (velocity / self.cfg.velocity_events)
                            - 1.0
                        )
                        / (self.cfg.velocity_exp - 1.0)
                    )
                )
                / binsize
            )

    def bin_to_velocity(self, bin: int) -> int:
        if self.cfg.velocity_bins_override:
            return self.cfg.velocity_bins_override[bin]
        binsize = self.cfg.velocity_events / (self.cfg.velocity_bins - 1)
        if self.cfg.velocity_exp == 1.0:
            return max(0, ceil(bin * binsize - 1))
        else:
            return max(
                0,
                ceil(
                    self.cfg.velocity_events
                    * log(
                        ((self.cfg.velocity_exp - 1) * binsize * bin)
                        / self.cfg.velocity_events
                        + 1,
                        self.cfg.velocity_exp,
                    )
                    - 1
                ),
            )

    def delta_to_wait_ids(self, delta_ms: float) -> Iterator[int]:
        def roundi(f: float):
            return ceil(f - 0.5)

        max_wait_ms = self.cfg.max_wait_time
        div = max_wait_ms / self.cfg.wait_events

        # if delta_ms // max_wait_ms > 512:  # arbitrary limit to avoid excessive time_shifts
        #    raise ValueError("delta_time is too large")
        if delta_ms > max_wait_ms * 10:
            delta_ms = max_wait_ms * 10  # truncate time

        for _ in range(floor(delta_ms / max_wait_ms)):
            yield roundi(max_wait_ms / div)
        leftover_time_shift = roundi((delta_ms % max_wait_ms) / div)
        if leftover_time_shift > 0:
            yield leftover_time_shift

    def prog_data_to_token_data(
        self, program: int, channel: int, note: int, velocity: float
    ) -> Optional[Tuple[int, int]]:
        # if channel == 9:
        #     if self.cfg._ch10_bin_int == -1:
        #         return None
        #     return self.cfg._ch10_bin_int, note, self.velocity_to_bin(velocity)
        return note, self.velocity_to_bin(velocity)

    def prog_data_list_to_token_data_list(
        self, data: List[Tuple[int, int, int, float]]
    ) -> Iterator[Tuple[int, int]]:
        for d in data:
            token_data = self.prog_data_to_token_data(*d)
            if token_data is not None:
                yield token_data

    # def sort_token_data(
    #     self, data: List[Tuple[int, int, int]]
    # ) -> List[Tuple[int, int, int]]:
    #     # ensure order is preserved for tokens with the same instrument, note
    #     data = [(i, n, v, x) for x, (i, n, v) in enumerate(data)]
    #     data.sort(key=lambda x: (x[0] != self.cfg._ch10_bin_int, x[0], x[1], x[3]))
    #     return [(i, n, v) for i, n, v, _ in data]
    #
    def data_to_wait_tokens(self, delta_ms: float) -> List[str]:
        if delta_ms == 0.0:
            return []
        return [self.format_wait_token(i) for i in self.delta_to_wait_ids(delta_ms)]

    def wait_token_to_delta(self, token: str) -> float:
        return self.cfg.max_wait_time / self.cfg.wait_events * int(token[1:])

    def note_token_to_data(self, token: str) -> Tuple[int, int, int]:
        note_str, velocity_str = token.strip().split(":")
        instr_bin = 0
        note = int(note_str, base=16)
        velocity = self.bin_to_velocity(int(velocity_str, base=16))
        return instr_bin, note, velocity


@dataclass
class AugmentValues:
    instrument_bin_remap: Dict[int, int]
    velocity_mod_factor: float
    transpose_semitones: int
    time_stretch_factor: float

    @classmethod
    def default(cls) -> "AugmentValues":
        return cls(
            instrument_bin_remap={},
            velocity_mod_factor=1.0,
            transpose_semitones=0,
            time_stretch_factor=1.0,
        )


@dataclass
class AugmentConfig:
    # The number of times to augment each MIDI file. The dataset size will be multiplied by this number.
    augment_data_factor: int
    # A list of instrument names to randomly swap with each other.
    instrument_mixups: List[List[str]]
    # A list of percentages to change the note velocity by. 0.0 = no change. 0 is included by default.
    velocity_mod_pct: List[float]
    # A list of semitones to transpose by. 0 is included by default.
    transpose_semitones: List[int]
    # A list of percentages to stretch the tempo by. 0.0 = no stretch. 0 is included by default.
    time_stretch_pct: List[float]
    # Random seed to use for reproducibility.
    seed: int

    cfg: VocabConfig

    def __post_init__(self):
        self.validate()
        if len(self.velocity_mod_pct) == 0:
            self.velocity_mod_pct = [0.0]
        if len(self.transpose_semitones) == 0:
            self.transpose_semitones = [0]
        if len(self.time_stretch_pct) == 0:
            self.time_stretch_pct = [0.0]

        self._instrument_mixups_int = [
            l for l in self._instrument_mixups_int if len(l) > 0
        ]  # remove empty lists
        self._instrument_pool_assignments = {}
        self._mixup_pools = []
        for pool_i, mixup_list in enumerate(self._instrument_mixups_int):
            pool = set()
            for i in mixup_list:
                pool.add(i)
                self._instrument_pool_assignments[i] = pool_i
            self._mixup_pools.append(pool)

    def validate(self):
        if self.augment_data_factor < 1:
            raise ValueError("augment_data_factor must be at least 1")
        used_instruments = set()
        for mixup_list in self.instrument_mixups:
            for n in mixup_list:
                if n in used_instruments:
                    raise ValueError(f"Duplicate instrument name: {n}")
                used_instruments.add(n)

    @classmethod
    def from_json(cls, path: str, cfg: VocabConfig):
        with open(path, "r") as f:
            config = json.load(f)
        config["cfg"] = cfg
        if "seed" not in config:
            config["seed"] = random.randint(0, 2**32 - 1)
        return cls(**config)

    def get_augment_values(self, filename: str) -> Iterator[AugmentValues]:
        # first yield default values
        yield AugmentValues.default()

        rng = random.Random(self.seed + hash(filename))
        for _ in range(int(self.augment_data_factor - 1)):
            # randomize order for each pool
            randomized_pools = [list(pool) for pool in self._mixup_pools]
            for pool in randomized_pools:
                rng.shuffle(pool)
            # distribute reassignments
            instrument_bin_remap = {}
            for i, pool in enumerate(randomized_pools):
                for j, instrument in enumerate(pool):
                    instrument_bin_remap[instrument] = randomized_pools[i - 1][j]
            yield AugmentValues(
                instrument_bin_remap=instrument_bin_remap,
                velocity_mod_factor=1.0 + rng.choice(self.velocity_mod_pct),
                transpose_semitones=rng.choice(self.transpose_semitones),
                time_stretch_factor=1.0 + rng.choice(self.time_stretch_pct),
            )


@dataclass
class FilterConfig:
    # Whether to filter out MIDI files with duplicate MD5 hashes.
    deduplicate_md5: bool
    # Minimum time delay between notes in a file before splitting into multiple documents.
    piece_split_delay: float
    # Minimum length of a piece in milliseconds.
    min_piece_length: float
    # filter get get only needed instrument
    instrument_type: List[str]

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            config = json.load(f)
        return cls(**config)


def mix_volume(velocity: int, volume: int, expression: int) -> float:
    return velocity * (volume / 127.0) * (expression / 127.0)


def convert_midi_to_str(
    cfg: VocabConfig,
    filter_cfg: FilterConfig,
    mid: mido.MidiFile,
    augment: AugmentValues | None = None,
) -> List[str]:
    utils = VocabUtils(cfg)
    # if augment is None:
    augment = AugmentValues.default()

    # filter out unknown meta messages before merge (https://github.com/mido/mido/pull/286)
    for i in range(len(mid.tracks)):
        mid.tracks[i] = [msg for msg in mid.tracks[i] if msg.type != "unknown_meta"]

    if len(mid.tracks) > 1:
        mid.tracks = [mido.merge_tracks(mid.tracks)]

    delta_time_ms = 0.0
    tempo = 500000
    channel_program = {i: 0 for i in range(16)}
    channel_volume = {i: 127 for i in range(16)}
    channel_expression = {i: 127 for i in range(16)}
    channel_notes = {i: {} for i in range(16)}
    channel_pedal_on = {i: False for i in range(16)}
    channel_pedal_events = {
        i: {} for i in range(16)
    }  # {channel: {(note, program) -> True}}
    started_flag = False

    output_list = []
    output = ["<start>"]
    output_length_ms = 0.0
    token_data_buffer: List[Tuple[int, int, int, float]] = []

    def flush_token_data_buffer():
        nonlocal token_data_buffer, output, cfg, utils, augment
        token_data = [
            x for x in utils.prog_data_list_to_token_data_list(token_data_buffer)
        ]
        if cfg.unrolled_tokens:
            for t in token_data:
                output += [
                    utils.format_unrolled_note(t[0]),
                    utils.format_unrolled_velocity(t[1]),
                ]
        else:
            output += [utils.format_note_token(*t) for t in token_data]
        token_data_buffer = []

    def consume_note_program_data(prog: int, chan: int, note: int, vel: float):
        nonlocal output, output_length_ms, started_flag, delta_time_ms, cfg, utils, token_data_buffer
        is_token_valid = (
            utils.prog_data_to_token_data(prog, chan, note, vel) is not None
        )
        if not is_token_valid:
            return
        if delta_time_ms > filter_cfg.piece_split_delay * 1000.0:
            # check if any notes are still held
            silent = True
            for channel in channel_notes.keys():
                if len(channel_notes[channel]) > 0:
                    silent = False
                    break
            if silent:
                flush_token_data_buffer()
                output.append("<end>")
                if output_length_ms > filter_cfg.min_piece_length * 1000.0:
                    output_list.append(" ".join(output))
                output = ["<start>"]
                output_length_ms = 0.0
                started_flag = False
        if started_flag:
            wait_tokens = utils.data_to_wait_tokens(delta_time_ms)
            if len(wait_tokens) > 0:
                flush_token_data_buffer()
                output_length_ms += delta_time_ms
                output += wait_tokens
        delta_time_ms = 0.0
        token_data_buffer.append((prog, chan, note, vel * augment.velocity_mod_factor))
        started_flag = True

    for msg in mid.tracks[0]:
        time_ms = mido.tick2second(msg.time, mid.ticks_per_beat, tempo) * 1000.0
        delta_time_ms += time_ms
        t = msg.type

        if msg.is_meta:
            if t == "set_tempo":
                tempo = msg.tempo * augment.time_stretch_factor
            continue

        def handle_note_off(ch, prog, n):
            if channel_pedal_on[ch]:
                channel_pedal_events[ch][(n, prog)] = True
            else:
                consume_note_program_data(prog, ch, n, 0)
                if n in channel_notes[ch]:
                    del channel_notes[ch][n]

        if t == "program_change":
            channel_program[msg.channel] = msg.program
        elif t == "note_on":
            if msg.velocity == 0:
                handle_note_off(msg.channel, channel_program[msg.channel], msg.note)
            else:
                if (msg.note, channel_program[msg.channel]) in channel_pedal_events[
                    msg.channel
                ]:
                    del channel_pedal_events[msg.channel][
                        (msg.note, channel_program[msg.channel])
                    ]
                consume_note_program_data(
                    channel_program[msg.channel],
                    msg.channel,
                    msg.note,
                    mix_volume(
                        msg.velocity,
                        channel_volume[msg.channel],
                        channel_expression[msg.channel],
                    ),
                )
                channel_notes[msg.channel][msg.note] = True
        elif t == "note_off":
            handle_note_off(msg.channel, channel_program[msg.channel], msg.note)
        elif t == "control_change":
            if msg.control == 7 or msg.control == 39:  # volume
                channel_volume[msg.channel] = msg.value
            elif msg.control == 11:  # expression
                channel_expression[msg.channel] = msg.value
            elif msg.control == 64:  # sustain pedal
                channel_pedal_on[msg.channel] = msg.value >= 64
                if not channel_pedal_on[msg.channel]:
                    for note, program in channel_pedal_events[msg.channel]:
                        handle_note_off(msg.channel, program, note)
                    channel_pedal_events[msg.channel] = {}
            elif msg.control == 123:  # all notes off
                for channel in channel_notes.keys():
                    for note in list(channel_notes[channel]).copy():
                        handle_note_off(channel, channel_program[channel], note)
        else:
            pass

    flush_token_data_buffer()
    output.append("<end>")

    if output_length_ms > filter_cfg.min_piece_length * 1000.0:
        output_list.append(" ".join(output))
    return output_list


def generate_program_change_messages(cfg: VocabConfig):
    for bin_name, channel in cfg.bin_channel_map.items():
        if channel == 9:
            continue
        program = cfg._instrument_names_str_to_int[
            cfg.bin_name_to_program_name[bin_name]
        ]
        yield mido.Message("program_change", program=program, time=0, channel=channel)
    yield mido.Message("program_change", program=0, time=0, channel=9)


@dataclass
class DecodeState:
    total_time: float  # milliseconds
    delta_accum: float  # milliseconds
    current_bin: int
    current_note: int
    active_notes: Dict[Tuple[int, int], float]  # { (channel, note): time started, ... }


def token_to_midi_message(
    utils: VocabUtils, token: str, state: DecodeState, end_token_pause: float = 3.0
) -> Iterator[Tuple[Optional[mido.Message], DecodeState]]:
    if state is None:
        state = DecodeState(
            total_time=0.0,
            delta_accum=0.0,
            current_bin=utils.cfg._short_instrument_names_str_to_int[
                utils.cfg.short_instr_bin_names[0]
            ],
            current_note=0,
            active_notes={},
        )
    token = token.strip()
    if not token:
        yield None, state
        return
    if token == "<end>":
        d = end_token_pause * 1000.0
        state.delta_accum += d
        state.total_time += d
        if utils.cfg.decode_end_held_note_delay != 0.0:
            # end held notes
            for (channel, note), start_time in list(state.active_notes.items()).copy():
                ticks = int(mido.second2tick(state.delta_accum / 1000.0, 480, 500000))
                state.delta_accum = 0.0
                del state.active_notes[(channel, note)]
                yield mido.Message(
                    "note_off", note=note, time=ticks, channel=channel
                ), state
        yield None, state
        return
    if token.startswith("<"):
        yield None, state
        return

    if utils.cfg.unrolled_tokens:
        if token[0] == "t":
            d = utils.wait_token_to_delta(token)
            state.delta_accum += d
            state.total_time += d
        elif token[0] == "n":
            state.current_note = int(token[1:], base=16)
        elif token[0] == "i":
            state.current_bin = utils.cfg._short_instrument_names_str_to_int[token[1:]]
        elif token[0] == "v":
            current_velocity = utils.bin_to_velocity(int(token[1:], base=16))
            channel = utils.cfg.bin_channel_map[
                utils.cfg.bin_instrument_names[state.current_bin]
            ]
            ticks = int(mido.second2tick(state.delta_accum / 1000.0, 480, 500000))
            state.delta_accum = 0.0
            if current_velocity > 0:
                yield mido.Message(
                    "note_on",
                    note=state.current_note,
                    velocity=current_velocity,
                    time=ticks,
                    channel=channel,
                ), state
            else:
                yield mido.Message(
                    "note_off",
                    note=state.current_note,
                    velocity=0,
                    time=ticks,
                    channel=channel,
                ), state
    else:
        if token[0] == "t" and token[1].isdigit():  # wait token
            d = utils.wait_token_to_delta(token)
            state.delta_accum += d
            state.total_time += d
            if utils.cfg.decode_end_held_note_delay != 0.0:
                # remove notes that have been held for too long
                for (channel, note), start_time in list(
                    state.active_notes.items()
                ).copy():
                    if (
                        state.total_time - start_time
                        > utils.cfg.decode_end_held_note_delay * 1000.0
                    ):
                        ticks = int(
                            mido.second2tick(state.delta_accum / 1000.0, 480, 500000)
                        )
                        state.delta_accum = 0.0
                        del state.active_notes[(channel, note)]
                        yield mido.Message(
                            "note_off", note=note, time=ticks, channel=channel
                        ), state
                        return
        else:  # note token
            bin, note, velocity = utils.note_token_to_data(token)
            channel = utils.cfg.bin_channel_map[utils.cfg.bin_instrument_names[bin]]
            ticks = int(mido.second2tick(state.delta_accum / 1000.0, 480, 500000))
            state.delta_accum = 0.0
            if velocity > 0:
                if utils.cfg.decode_fix_repeated_notes:
                    if (channel, note) in state.active_notes:
                        del state.active_notes[(channel, note)]
                        yield mido.Message(
                            "note_off", note=note, time=ticks, channel=channel
                        ), state
                        ticks = 0
                state.active_notes[(channel, note)] = state.total_time
                yield mido.Message(
                    "note_on", note=note, velocity=velocity, time=ticks, channel=channel
                ), state
                return
            else:
                if (channel, note) in state.active_notes:
                    del state.active_notes[(channel, note)]
                yield mido.Message(
                    "note_off", note=note, time=ticks, channel=channel
                ), state
                return
    yield None, state


def str_to_midi_messages(utils: VocabUtils, data: str) -> Iterator[mido.Message]:
    state = None
    for token in data.split(" "):
        for msg, new_state in token_to_midi_message(utils, token, state):
            state = new_state
            if msg is not None:
                yield msg


def convert_str_to_midi(
    cfg: VocabConfig, data: str, tempo: int, meta_text: str = "Generated by AITUNE"
) -> mido.MidiFile:
    utils = VocabUtils(cfg)
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(tempo)
    if meta_text:
        track.append(mido.MetaMessage("text", text=meta_text, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    for msg in generate_program_change_messages(cfg):
        track.append(msg)

    data = data.replace("<start>", "").replace("<end>", "").replace("<pad>", "").strip()
    for msg in str_to_midi_messages(utils, data):
        track.append(msg)

    track.append(mido.MetaMessage("end_of_track", time=0))

    return mid
