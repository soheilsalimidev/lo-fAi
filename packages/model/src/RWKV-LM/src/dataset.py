########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json
import math
import os
import random
import sys

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import Dataset

from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(
            f"Current vocab size = {self.vocab_size} (make sure it's correct)"
        )

        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        self.data_pile = None
        self.data_pile_size = 0

        if args.my_pile_stage > 0:
            print(self.data_size)
            # assert self.data_size == 332115325534 and self.vocab_size == 50277
            self.samples_per_epoch = args.epoch_steps * args.real_bsz
            assert self.samples_per_epoch == 40320
            dataset_slot = self.data_size // args.ctx_len
            if args.my_pile_stage != 4:
                assert MaybeIsPrime(args.magic_prime)
                assert args.magic_prime % 3 == 2
                assert (
                    args.magic_prime / dataset_slot > 0.99
                    and args.magic_prime / dataset_slot <= 1
                )

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        if args.data_type == "uint16":
            i = np.random.randint(0, self.data_size - 1)
            dix = self.data[i]
            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long)
        else:
            ctx_len = args.ctx_len
            req_len = ctx_len + 1
            magic_prime = args.magic_prime
            data = self.data

            if args.my_pile_stage > 0:
                ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

                if args.my_qa_mask > 0:
                    ii_orig = ii
                    if ii % 2 == 0:
                        ii = -1
                        data = self.data_pile
                    else:
                        ii = ii // 2
                if data == self.data_pile:
                    i = np.random.randint(0, self.data_pile_size - req_len)
                else:
                    if args.my_pile_stage == 4 or ii < args.my_random_steps:
                        # cheat: pick a random spot in dataset
                        if args.my_pile_version == 1:
                            i = np.random.randint(0, self.data_size - req_len)
                        else:
                            i = np.random.randint(0, self.data_size)
                    else:
                        ii = ii - args.my_random_steps
                        factor = (math.sqrt(5) - 1) / 2
                        factor = int(magic_prime * factor)
                        i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
                        i = i + args.my_pile_shift
                # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size} ii {ii} pos {round(i / self.data_size, 3)}")
            else:
                # cheat: pick a random spot in dataset
                i = np.random.randint(0, self.data_size - req_len)

            if args.data_type == "binidx":
                if args.my_pile_version == 1:
                    dix = data.get(idx=0, offset=i, length=req_len).astype(int)
                else:
                    # self.data : cutoff, chunk_count, data
                    for j in range(len(data)):
                        if i < data[j][0]:
                            ii = i
                            i = (i - (data[j - 1][0] if j > 0 else 0)) % data[j][1]
                            dix = (
                                data[j][2]
                                .get(idx=0, offset=i, length=req_len)
                                .astype(int)
                            )
                            # print(ii, j, i)
                            break
            elif args.data_type == "numpy":
                dix = data[i : i + req_len]
            else:
                dix = [self.stoi[s] for s in data[i : i + req_len]]

            if args.my_qa_mask == 1:
                if data == self.data_pile:
                    z = [1] * ctx_len
                else:
                    z = [0] * ctx_len
                    z_sum = 0
                    isGood = False
                    for i in range(3, ctx_len):
                        if (
                            dix[i] == 27
                            and dix[i - 1] == 34
                            and dix[i - 2] == 187
                            and dix[i - 3] == 187
                        ):
                            isGood = True
                        if dix[i] == 0:
                            isGood = False
                        if isGood:
                            z[i] = 1
                            z_sum += 1
                    if z_sum == 0:
                        z = [1] * ctx_len
                        i = np.random.randint(0, self.data_pile_size - req_len)
                        dix = self.data_pile.get(
                            idx=0, offset=i, length=req_len
                        ).astype(int)
                z = torch.tensor(z, dtype=torch.bfloat16)

            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long)

            # if ii_orig < 50:
            #     # if rank == 1:
            #     print('rank', rank, 'i', ii_orig, ii, i, 'x', x[:5], '...', x[-5:])
            # else:
            #     exit(0)

            if args.my_qa_mask == 1:
                return x, y, z

            return x, y
