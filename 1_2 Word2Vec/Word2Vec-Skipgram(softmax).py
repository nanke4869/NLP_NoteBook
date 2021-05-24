import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_batch(skip_grams, batch_size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(vocab_size)[skip_grams[i][0]])  # one-hot向量
        random_labels.append(skip_grams[i][1])

    return random_inputs, random_labels


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Word2Vec, self).__init__()

        self.W = nn.Linear(vocab_size, embedding_size, bias=False)
        self.WT = nn.Linear(embedding_size, vocab_size, bias=False)

    def forward(self, X):
        # X : [batch_size, vocab_size]
        hidden_layer = self.W(X)  # [batch_size, embedding_size]
        output_layer = self.WT(hidden_layer)  # [batch_size, vocab_size]
        output = nn.functional.softmax(output_layer, dim=1)

        return output


if __name__ == '__main__':
    batch_size = 2
    embedding_size = 2
    lr = 0.001
    epochs = 7000
    train_iter = 1000
    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    word_sequence = " ".join(sentences).split()

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    # Make skip gram of one size window
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        input = word_dict[word_sequence[i]]
        targets = [word_dict[word_sequence[i-1]], word_dict[word_sequence[i+1]]]
        for w in targets:
            skip_grams.append([input, w])

    model = Word2Vec(vocab_size, embedding_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    for epoch in range(epochs):
        input_batch, target_batch = random_batch(skip_grams, batch_size)
        input_batch = torch.Tensor(input_batch).to(device)
        target_batch = torch.Tensor(target_batch).to(device, torch.long)  # expected scalar type Long

        optimizer.zero_grad()
        output = model(input_batch)

        # output : [batch_size, vocab_size], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, target_batch)
        if (epoch+1) % train_iter == 0:
            print('| Epoch:', '%04d' % (epoch + 1), '| cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()