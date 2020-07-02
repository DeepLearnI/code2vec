from __future__ import annotations


from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pebble import ProcessPool
import concurrent.futures as cf
from functools import partial
from pathlib import PosixPath
import os
import math
import itertools as it
from glob import iglob, glob
import subprocess as sp
from tqdm import tqdm
from typing import Iterator, List, Iterable, Tuple, Union

import re

from extractor import Extractor


Path = Union[str, PosixPath]


def process(fname: str, max_length: int, max_wdith: int) -> List[str]:
    extractor = Extractor(max_path_length=8, max_path_width=2)
    try:
        paths = extractor.extract_paths(fname)
    except (ValueError, SyntaxError, RecursionError):
        return list()
    return list(paths)


def write_lines(fname: str, lines: Iterable[str]) -> None:
    with open(fname, "a", encoding="ISO-8859-1") as stream:
        stream.writelines(map(mask_method_name, lines))


def mask_method_name(line: str) -> str:
    method_name, _, _ = line.partition(" ")
    pattern = re.compile(re.escape(f" {method_name},"))
    return pattern.sub(" METHOD_NAME,", line)


def to_str_path(list_path: List[str]) -> str:
    return f"{list_path[0]},{'|'.join(list_path[1:-1])},{list_path[-1]}"


def make_posix_path(path: Path) -> PosixPath:
    return PosixPath(path) if isinstance(path, str) else path


def concatenate_path_conext_files(mined_dir_path: Path) -> None:
    mined_dir_path = make_posix_path(mined_dir_path)
    dtq = tqdm(["train", "test", "val"], desc="concatenating ast path conext files")
    for _dir in dtq:
        file_dir = str(mined_dir_path / f"{_dir}")
        concate_sh = f"cat {file_dir}/*.c2v > {file_dir}/path_contexts.csv"
        sp.run(concate_sh, shell=True, check=True)

    for f in iglob(str(mined_dir_path / "*/*.c2v")):
        os.remove(f)

    print("Done concatenating all path_contexts from AST miner to a single file")


def source_files(data_dir: str):
    for fname in iglob(f"{data_dir}/*/**/[!setup]*.py", recursive=True):
        if os.path.isfile(fname) and not fname.startswith("test"):
            yield fname


def chunker(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # chunker('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return it.zip_longest(*args, fillvalue=fillvalue)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-maxlen", "--max_path_length", dest="max_path_length", required=False, default=8)
    parser.add_argument("-maxwidth", "--max_path_width", dest="max_path_width", required=False, default=2)
    parser.add_argument("-workers", "--max_workers", dest="max_workers", required=False, default=None)
    parser.add_argument("-in_dir", "--in_dir", dest="in_dir", required=True)
    parser.add_argument("-out_dir", "--out_dir", dest="out_dir", required=True)
    # parser.add_argument("-file", "--file", dest="file", required=False)
    args = parser.parse_args()

    TIMEOUT = 60 * 10
    MAX_WORKERS = int(args.max_workers)
    MAX_LENGTH = args.max_path_length
    MAX_WIDTH = args.max_path_width
    REPOS = args.in_dir
    OUTPUT = args.out_dir

    writes = list()
    futures = list()
    with ProcessPool(max_workers=MAX_WORKERS) as pool, ThreadPoolExecutor(
        max_workers=1
    ) as writer:
        futures = {
            pool.schedule(process, args=[fname, MAX_LENGTH, MAX_WIDTH], timeout=TIMEOUT): fname
            for fname in source_files(REPOS)
        }

        for future in tqdm(cf.as_completed(futures), total=len(futures)):
            fname = futures[future]
            splitted = fname.split("/")
            project = splitted[2]
            bin_ = splitted[1]
            c2v_file = f"{OUTPUT}/{bin_}/{project}.c2v"
            try:
                paths = future.result()
            except cf.TimeoutError:
                continue
            if paths:
                writes.append(writer.submit(partial(write_lines, c2v_file), paths))

        cf.wait(writes)

    concatenate_path_conext_files(mined_dir_path=OUTPUT)
