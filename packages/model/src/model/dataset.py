import os
import random
import numpy as np
from torch import  tensor, long
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning.utilities import rank_zero_info
from binidx import MMapIndexedDataset

def FermatPrimalityTest(number):
    if number > 1:
        for time in range(3):
            randomNumber = random.randint(2, number) - 1
            if pow(randomNumber, number - 1, number) != 1:
                return False
        return True
    else:
        return False


def MillerRabinPrimalityTest(number):
    if number == 2:
        return True
    elif number == 1 or number % 2 == 0:
        return False
    oddPartOfNumber = number - 1
    timesTwoDividNumber = 0
    while oddPartOfNumber % 2 == 0:
        oddPartOfNumber = oddPartOfNumber // 2
        timesTwoDividNumber = timesTwoDividNumber + 1

    for time in range(3):
        while True:
            randomNumber = random.randint(2, number) - 1
            if randomNumber != 0 and randomNumber != 1:
                break

        randomNumberWithPower = pow(randomNumber, oddPartOfNumber, number)

        if (randomNumberWithPower != 1) and (randomNumberWithPower != number - 1):
            iterationNumber = 1

            while (iterationNumber <= timesTwoDividNumber - 1) and (randomNumberWithPower != number - 1):
                randomNumberWithPower = pow(randomNumberWithPower, 2, number)
                iterationNumber = iterationNumber + 1
            if randomNumberWithPower != (number - 1):
                return False

    return True


class RegularDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(
            f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        if args.data_file.endswith('/'):
            d_all = []
            for p in os.listdir(args.data_file):
                if p.endswith(".idx"):
                    d_all += [p[:-4]]
            d_all.sort()
            rank_zero_info(d_all)
            exit(0)
        else:
            self.data = MMapIndexedDataset(args.data_file)
            self.data_size = len(
                self.data._bin_buffer) // self.data._index._dtype_size
            rank_zero_info(f"Data has {self.data_size} tokens.")

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args

        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        data = self.data

        # cheat: pick a random spot in dataset
        i = np.random.randint(0, self.data_size - req_len)

        dix = data.get(idx=0, offset=i, length=req_len).astype(int)

        x = tensor(dix[:-1], dtype=long)
        y = tensor(dix[1:], dtype=long)

        if args.my_qa_mask == 1:
            return x, y, z

        return x, y
