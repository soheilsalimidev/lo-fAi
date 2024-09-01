import argparse
import functools
import hashlib
import io
import json
import multiprocessing
import os
import tarfile
import zipfile
from typing import Iterable, List, Optional, Tuple, Union

import midi_util
import mido
from midi_util import AugmentConfig, FilterConfig, VocabConfig
from tqdm import tqdm


def convert_midi_bytes_to_str(
    cfg: VocabConfig,
    filter_cfg: FilterConfig,
    aug_cfg: AugmentConfig | None,
    data: Tuple[str, bytes],
) -> Tuple[str, Union[str, List, None]]:
    filename, filedata = data
    try:
        mid = mido.MidiFile(file=io.BytesIO(filedata))
    except:
        return filename, None
    if mid.type not in (0, 1, 2):
        return filename, None
    if len(mid.tracks) == 0:
        return filename, None


    if aug_cfg is not None:
        return filename, [
            *(
                midi_util.convert_midi_to_str(cfg, filter_cfg, mid, augment)
                for augment in aug_cfg.get_augment_values(filename)
            )
        ]

    return filename, midi_util.convert_midi_to_str(cfg, filter_cfg, mid)


def midi_to_jsonl(
    cfg: VocabConfig,
    filter_cfg: FilterConfig,
    path: str,
    output: str,
    augment_config: Optional[AugmentConfig] = None,
    workers: int = 1,
):
    file_md5s = set()
    duplicate_file_count = 0

    def check_dedup(filebytes: bytes):
        nonlocal duplicate_file_count
        if filter_cfg.deduplicate_md5:
            # no need to hash whole file, and this is a main-thread hot path
            file_md5 = hashlib.md5(filebytes[:512])
            file_md5.update(filebytes[-256:])
            if file_md5 in file_md5s:
                duplicate_file_count += 1
                return True
            file_md5s.add(file_md5)
        return False

    pool = multiprocessing.Pool(workers)

    def file_generator() -> Iterable[Tuple[str, bytes]]:
        with zipfile.ZipFile(path, "r") as zip:
            for member in zip.infolist():
                if not member.is_dir() and member.filename.endswith((".mid", ".midi")):
                    filebytes = zip.read(member.filename)
                    if check_dedup(filebytes):
                        continue
                    yield (member.filename, filebytes)

    failed_file_count = 0
    total_file_count = 0

    # write results to jsonl file
    with open(output, "w") as f, open(output + ".failed", "w") as f_failed:
        for filename, result in tqdm(
            pool.imap(
                functools.partial(
                    convert_midi_bytes_to_str, cfg, filter_cfg, augment_config
                ),
                file_generator(),
                chunksize=48,
            )
        ):
            total_file_count += 1
            if result is not None:
                if type(result) is list:
                    for r in result:
                        f.write(json.dumps({"file": filename, "text": r}) + "\n")
                else:
                    f.write(json.dumps({"file": filename, "text": result}) + "\n")
            else:
                f_failed.write(filename + "\n")
                failed_file_count += 1

    total_file_count_dup = total_file_count + duplicate_file_count

    if filter_cfg.deduplicate_md5:
        print(
            f"Skipped {duplicate_file_count} duplicate files ({duplicate_file_count / (total_file_count_dup) * 100:.2f}%)"
        )
    print(
        f"Failed to convert {failed_file_count} files ({failed_file_count / total_file_count * 100:.2f}%)"
    )
    if failed_file_count == 0:
        os.remove(output + ".failed")
