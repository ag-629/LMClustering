import torch, re, sys
from torch.utils.data import DataLoader
from torch.optim import Adam
from argparse import ArgumentParser
#from character_lm import forward
from dataloader import BibleData
from dataloader import PadCollate
from train import Trainer
from character_lm import LM
import numpy as np
import pandas as pd
from collections import defaultdict


def get_item(data_set, word):
  word = ''.join(data_set.decode(b)[1:])
  inp = data_set.encode(word, add_start_tag=True, add_end_tag=False)
# First char to end_tag. This offsets target to always be 1 char ahead.
  target = data_set.encode(word, add_start_tag=False, add_end_tag=True)
  
  return inp, target


if __name__ == '__main__':

  parser = ArgumentParser(description = 'Get confidences and normalization factor')
  parser.add_argument('path', help = 'path to bible')
  parser.add_argument('lang', help = "Language")
  args = parser.parse_args()
  
  model = torch.load('./'+args.lang+'.model', map_location=torch.device('cpu'))
  data_set = BibleData(args.path)
  data_loader = DataLoader(
        data_set, collate_fn=PadCollate(pad_idx=data_set.pad_idx), batch_size=1, shuffle=True
    )
  optimizer = Adam(model.parameters(), lr=0.001, weight_decay = 1e-5)#wd = 1e-5
  trainer = Trainer(
        model=model, optimizer=optimizer, dataloader=data_loader)


  confidences = defaultdict(list)

  for batch_in, batch_target in data_loader:
    for b, t in zip(batch_in, batch_target):
      inp, target = get_item(data_set, b)
      inp = inp[:-1]
      target = target[:-1]
      pairs = list(trainer.eval(inp.unsqueeze(0), target.unsqueeze(0), data_set))
      key = ''.join(data_set.decode(b)[1:])
      confidences[key] = [tup[1].item() for tup in pairs]
      
      
  out1 = open('confidences/'+args.lang+'/'+args.lang+'_confidences', 'w', encoding = 'utf-8')
  out1.write(str(dict(confidences)))
  out1.close()

