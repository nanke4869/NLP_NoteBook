import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def make_batch(sentences, word_dict, vocab_size):
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[w] for w in word[:-1]]
        target = word_dict[word[-1]]

        input_batch.append(np.eye(vocab_size)[input])
        target_batch.append(target)

    return input_batch, target_batch


class TextCNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(TextCNN, self).__init__()
        self.rnn = nn.RNN(vocab_size, hidden_dim)
        self.W = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.b = nn.Parameter(torch.ones(vocab_size))

    def forward(self, X, hidden):
        X = X.transpose(0, 1)  # X : [n_step=sequence_length, batch_size, n_class=vocab_size]
        outputs, hidden = self.rnn(X, hidden)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        outputs = outputs[-1]  # 只使用最终时刻的输出作为特征, [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model


if __name__ == '__main__':
    sequence_length = 2
    hidden_dim = 5
    lr = 0.001
    epochs = 7000
    train_iter = 1000

    sentences = ["i like dog", "i love coffee", "i hate milk"]
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)
    batch_size = len(sentences)

    model = TextCNN(vocab_size, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    input_batch, target_batch = make_batch(sentences, word_dict, vocab_size)
    input_batch = torch.FloatTensor(input_batch)  # input_batch : [batch_size, n_step, n_class]
    target_batch = torch.LongTensor(target_batch)

    # Training
    for epoch in range(epochs):
        optimizer.zero_grad()

        # hidden : [num_layers * num_directions, batch, hidden_size]
        hidden = torch.zeros(1, batch_size, hidden_dim)
        output = model(input_batch, hidden)
        # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)

        if (epoch + 1) % train_iter == 0:
            print('| Epoch:', '%04d' % (epoch + 1), '| cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # predict
    hidden = torch.zeros(1, batch_size, hidden_dim)
    predict = model(input_batch, hidden).data.max(1, keepdim=True)[1]
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])
    '''
    [['i', 'like'], ['i', 'love'], ['i', 'hate']] -> ['dog', 'coffee', 'milk']
    '''
