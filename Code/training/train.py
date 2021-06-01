import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import re, sys

import click
from tqdm import tqdm

from character_lm import LM
from dataloader import BibleData, ReverseBibleData, PadCollate


class Trainer:
    def __init__(self, model, optimizer, dataloader):
        self.model = model
        self.optimizer = optimizer

    def train_step(self, inp, target, loss_func, batch_size):
        """
        inp: input batch of size: batch_size x seq_len
        target: target batch of size: batch_size x seq_len
        batch_size: the batch_size
        """
        hidden = self.model.init_hidden(batch_size)
        self.optimizer.zero_grad()
        loss = 0

        # --> seq_len x batch_size
        inp = torch.transpose(inp, 0, 1)
        target = torch.transpose(target, 0, 1)

        # For each char in the batch, starting from idx 1 (we ignore the <S> tag)
        for i in range(len(inp)):
            # Each step gets the input character of the word at i.
            # This means we are using teacher forcing. We could isntead pass in the previously
            # predicted character
            output, hidden = self.model(inp[i], hidden)#calls forward function
            #print("inp[i]: ", inp[i], "output: ", output)
            # Squeeze off empty output sequence_len
            loss += loss_func(output.squeeze(0), target[i])

        loss.backward()
        self.optimizer.step()

        return loss.data / len(inp)

    def eval(self, inp, target, data_set, batch_size = 1):
        self.model.eval()
        hidden = self.model.init_hidden(batch_size)
        inp = torch.transpose(inp, 0, 1)
        target = torch.transpose(target, 0, 1)

        characters = []
        confidences = []
        
        for i in range(len(inp)):
            output, hidden = self.model(inp[i], hidden)
            #the model expect batchsize x seqlength x characterdim
            #but we only send batchsize=1 x seqlength=1 x characterdim
            #so the output 1 x 1 x outputvocabulary, first 2 dims are pointless, squeeze them both off
            #so now we have confidence over all outputs
            output = output.squeeze(0).squeeze(0)
            confidences.append(torch.exp(output[i]))#<---check
            #print('Confidence: ', torch.exp(output[target[i]]))<----- might be this
            characters.append(data_set.decode(target[i]))#<----should be index of target character

        return zip(characters, confidences)


@click.command()
@click.option("--train_data_path", type=str, required=True)
@click.option("--lang", type=str, required=True)
@click.option("--num_epochs", type=int)
@click.option("--learning_rate", type=float, default=.001)
@click.option("--batch_size", type=int, default=16)
@click.option("--embedding_size", type=int, default=128)
@click.option("--hidden_size", type=int, default=128)
#@click.option("--reverse/--no-reverse", default=False)
def main(train_data_path, lang, num_epochs, learning_rate, batch_size, embedding_size, hidden_size):
    #if reverse:
        #train_set = ReverseBibleData(train_data_path)
    train_set = BibleData(train_data_path)
        
    train_loader = DataLoader(
        train_set, collate_fn=PadCollate(pad_idx=train_set.pad_idx), batch_size=batch_size, shuffle=True
    )
    
    # TODO: Do we want bidirectional?
    model = LM(
        input_size=train_set.character_size, embedding_size=embedding_size,
        hidden_size=hidden_size, output_size=train_set.character_size
    )
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)
    trainer = Trainer(
        model=model, optimizer=optimizer, dataloader=train_loader,
    )

    loss_func = model.get_loss_func(train_set.pad_idx)


    for batch_in, batch_target in train_loader:#calling defined dataset's 'get item' function
        print(f"example batch")
        for b, t in zip(batch_in, batch_target):
            print(f"INPUT: {train_set.decode(b)} OUTPUT: {train_set.decode(t)}")

        break
    
    avg_loss = 5000.0
    for i in range(num_epochs):
        print(f"EPOCH: {i}")
        batch_losses = []
        for batch_in, batch_target in tqdm(train_loader, desc='Training', position = 0, leave = True):
            #batch_in, batch_target = batch_in.to(device), batch_target.to(device)
            # Final batch is remainder, so can be variable len
            batch_size = batch_in.size(0)
            batch_losses.append(trainer.train_step(batch_in, batch_target, loss_func, batch_size))
            
        print('batch loss: ', sum(batch_losses)/len(batch_losses))
        if sum(batch_losses)/len(batch_losses) < avg_loss:
            avg_loss = sum(batch_losses)/len(batch_losses)
            print(f"Average loss for epoch {i}: {avg_loss}")
            model_fn ='./'+lang+'.model'
            print(f"Saving model to {model_fn}")
            torch.save(model, model_fn)

        else:
          break
if __name__=='__main__':
    main()
