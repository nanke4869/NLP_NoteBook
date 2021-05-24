import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, num_filters, filter_sizes, vocab_size, embedding_size, num_classes):
        super(TextCNN, self).__init__()
        self.num_filters_total = num_filters * len(filter_sizes)
        self.W = nn.Embedding(vocab_size, embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

    def forward(self, X):
        # X : [batch_size, sequence_length]
        embedded_chars = self.W(X)  # [batch_size, sequence_length, embedding_size]
        embedded_chars = embedded_chars.unsqueeze(1)  # add channel(=1) [batch_size, channel(=1), sequence_length, embedding_size]

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            # conv : [input_channel(=1), output_channel(=3), (filter_height, filter_width), bias_option]
            h = F.relu(conv(embedded_chars))
            mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))
            # mp : ((filter_height, filter_width))
            pooled = mp(h).permute(0, 3, 2, 1)
            # pooled : [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3)]
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(filter_sizes))  # [batch_size(=6), output_height(=1), output_width(=1), output_channel(=3) * 3]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])  # [batch_size(=6), output_height * output_width * (output_channel * 3)]
        model = self.Weight(h_pool_flat) + self.Bias  # [batch_size, num_classes]
        return model


if __name__ == '__main__':
    embedding_size = 2
    sequence_length = 3
    num_classes = 2
    filter_sizes = [2, 2, 2]
    num_filters = 3
    epoches = 5000
    train_iter = 1000

    sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
    labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    vocab_size = len(word_dict)

    model = TextCNN(num_filters, filter_sizes, vocab_size, embedding_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = torch.LongTensor([np.asarray([word_dict[n] for n in sen.split()]) for sen in sentences])
    targets = torch.LongTensor([out for out in labels])  # To using Torch Softmax Loss function

    # Training
    for epoch in range(epoches):
        optimizer.zero_grad()
        output = model(inputs)

        # output : [batch_size, num_classes], target_batch : [batch_size] (LongTensor, not one-hot)
        loss = criterion(output, targets)
        if (epoch + 1) % train_iter == 0:
            print('| Epoch:', '%04d' % (epoch + 1), '| cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Test
    test_text = ['sorry hate you']
    tests = [np.asarray([word_dict[n] for n in text.split()]) for text in test_text]
    test_batch = torch.LongTensor(tests)

    predict = model(test_batch).data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "is Bad Mean...")
    else:
        print(test_text, "is Good Mean!!")
