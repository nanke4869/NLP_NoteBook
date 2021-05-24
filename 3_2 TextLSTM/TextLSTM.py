import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def make_batch(seq_data, word_dict, vocab_size):
    input_batch, target_batch = [], []

    for seq in seq_data:
        input = [word_dict[w] for w in seq[: -1]]  # 'm', 'a' , 'k' is input
        target = word_dict[seq[-1]]  # 'e' is target
        input_batch.append(np.eye(vocab_size)[input])
        target_batch.append(target)

    return input_batch, target_batch


class TextLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(TextLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(vocab_size, hidden_dim)
        self.W = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.b = nn.Parameter(torch.ones(vocab_size))

    def forward(self, X):
        input = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]

        hidden_state = torch.zeros(1, len(X), self.hidden_dim)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = torch.zeros(1, len(X), self.hidden_dim)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model


if __name__ == '__main__':
    sequence_length = 3  # number of cells(= number of Step)
    hidden_dim = 128  # number of hidden units in one cell
    lr = 0.001
    epochs = 7000
    train_iter = 1000

    char_arr = [c for c in 'abcdefghijklmnopqrstuvwxyz']
    word_dict = {n: i for i, n in enumerate(char_arr)}
    number_dict = {i: w for i, w in enumerate(char_arr)}
    vocab_size = len(word_dict)

    seq_data = ['make', 'need', 'coal', 'word', 'love', 'hate', 'live', 'home', 'hash', 'star']

    model = TextLSTM(vocab_size, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    input_batch, target_batch = make_batch(seq_data, word_dict, vocab_size)
    input_batch = torch.FloatTensor(input_batch)  # input_batch : [batch_size, n_step, n_class]
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % train_iter == 0:
            print('| Epoch:', '%04d' % (epoch + 1), '| cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print([sen[: 3] for sen in seq_data], '->', [number_dict[n.item()] for n in predict.squeeze()])
