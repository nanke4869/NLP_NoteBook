import argparse
import numpy as np
import torch
import torch.nn as nn

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps


def make_batch(seq_data, sequence_length, vocab_size):
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (sequence_length - len(seq[i]))

        input = [number_dict[n] for n in seq[0]]
        output = [number_dict[n] for n in ('S' + seq[1])]
        target = [number_dict[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(vocab_size)[input])
        output_batch.append(np.eye(vocab_size)[output])
        target_batch.append(target)

    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_dim, dropout):
        super(Seq2Seq, self).__init__()

        self.enc_cell = nn.RNN(vocab_size, hidden_dim, dropout=dropout)
        self.dec_cell = nn.RNN(vocab_size, hidden_dim, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, enc_input, enc_hidden, dec_input):
        enc_input = enc_input.transpose(0, 1)  # enc_input: [max_len(n_step=seq_len, time step), batch_size, n_class=vocab_size]
        dec_input = dec_input.transpose(0, 1)  # dec_input: [max_len(=n_step, time step), batch_size, n_class]

        _, enc_states = self.enc_cell(enc_input, enc_hidden)
        # enc_states: [num_layers(=1)*num_directions(=1), batch_size, n_hidden]
        outputs, _ = self.dec_cell(dec_input, enc_states)
        # outputs: [max_len+1(=6), batch_size, num_directions(=1) * n_hidden(=128)]

        model = self.fc(outputs)  # model: [max_len+1(=6), batch_size, n_class]
        return model


# Test
def translate(word):
    seq_data = [[word, 'P'*len(word)]]
    input_batch, output_batch, _ = make_batch(seq_data, sequence_length, vocab_size)

    # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
    hidden = torch.zeros(1, 1, hidden_dim)
    output = model(input_batch, hidden, output_batch)
    # output : [max_len+1(=6), batch_size(=1), n_class]

    predict = output.data.max(2, keepdim=True)[1] # select n_class dimension
    decoded = [char_arr[i] for i in predict]
    end = decoded.index('E')
    translated = ''.join(decoded[:end])

    return translated.replace('P', '')


if __name__ == '__main__':
    sequence_length = 5  # max sequence length = n_step
    hidden_dim = 128
    dropout = 0.5
    lr = 0.001
    epochs = 5000
    train_iter = 1000

    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    number_dict = {n: i for i, n in enumerate(char_arr)}
    seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]
    vocab_size = len(number_dict)  # = n_class
    batch_size = len(seq_data)

    model = Seq2Seq(vocab_size, hidden_dim, dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    input_batch, output_batch, target_batch = make_batch(seq_data, sequence_length, vocab_size)

    # Training
    for epoch in range(epochs):
        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = torch.zeros(1, batch_size, hidden_dim)

        optimizer.zero_grad()
        # input_batch : [batch_size, max_len(=n_step, time step), n_class]
        # output_batch : [batch_size, max_len+1(=n_step, time step) (becase of 'S' or 'E'), n_class]
        # target_batch : [batch_size, max_len+1(=n_step, time step)], not one-hot

        output = model(input_batch, hidden, output_batch)
        # output : [max_len+1, batch_size, n_class]
        output = output.transpose(0, 1)  # [batch_size, max_len+1(=6), n_class]

        loss = 0
        for i in range(0, len(target_batch)):
            # output[i] : [max_len+1, n_class], target_batch[i] : [max_len+1]
            loss += criterion(output[i], target_batch[i])
        if (epoch + 1) % train_iter == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()


    # test
    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('upp ->', translate('upp'))



