import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def make_batch(sentence, max_len, vocab_size):
    input_batch, target_batch = [], []

    words = sentence.split()
    for i, word in enumerate(words[: -1]):
        input = [word_dict[w] for w in words[: (i+1)]]
        input = input + [0] * (max_len - len(input))  # padding
        target = word_dict[words[i+1]]
        print(input)
        input_batch.append(np.eye(vocab_size)[input])
        target_batch.append(target)

    return input_batch, target_batch


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(BiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(vocab_size, hidden_dim, bidirectional=True)
        self.W = nn.Linear(hidden_dim*2, vocab_size, bias=False)
        self.b = nn.Parameter(torch.ones(vocab_size))

    def forward(self, X):
        input = X.transpose(0, 1)  # input : [n_step, batch_size, n_class]

        hidden_state = torch.zeros(1 * 2, len(X), self.hidden_dim)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1 * 2, len(X), self.hidden_dim)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model


if __name__ == '__main__':
    hidden_dim = 5  # number of hidden units in one cell
    lr = 0.001
    epochs = 10000
    train_iter = 1000

    sentence = (
        'Born as the bright summer flowers '
        'Do not withered undefeated fiery demon rule '
        'Heart rate and breathing to bear the load of the cumbersome Bored'
    )

    word_dict = {w: i for i, w in enumerate(list(set(sentence.split())))}
    number_dict = {i: w for i, w in enumerate(list(set(sentence.split())))}
    vocab_size = len(word_dict)
    max_len = len(sentence.split())

    model = BiLSTM(vocab_size, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    input_batch, target_batch = make_batch(sentence, max_len, vocab_size)
    input_batch = torch.FloatTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)


    # Training
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(input_batch)
        loss = criterion(output, target_batch)
        if (epoch + 1) % train_iter == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(sentence)
    print([number_dict[n.item()] for n in predict.squeeze()])