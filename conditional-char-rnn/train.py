from data import *
from model import *
import torch
import time
import math

n_hidden = 128
rnn = RNN(n_categories, n_letters, n_hidden, n_letters)

# gpu cuda
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("device: " + str(device))
rnn.to(device)
# input = lineToTensor('Albert')
# hidden = torch.zeros(1, n_hidden)
# output, next_hidden = rnn(input[0], hidden)
# print(output)

criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)


def train(category_tensor, input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()
    category_tensor = category_tensor.to(device)
    input_line_tensor = input_line_tensor.to(device)
    hidden = hidden.to(device)
    target_line_tensor = target_line_tensor.to(device)

    rnn.zero_grad()
    loss = 0
    for i in range(input_line_tensor.size()[0]):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        loss += criterion(output, target_line_tensor[i])
    loss.backward()

    optimizer.step()

    return output, loss.item() / input_line_tensor.size(0)


n_iters = 100000
print_every = 5000
plot_every = 500
current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


start = time.time()

for iter in range(1, n_iters + 1):
    out_put, loss = train(*randomTrainingExample())
    current_loss += loss
    if iter % print_every == 0:
        print('%d  %d%% (%s) %.4f' % (
            iter, iter / n_iters * 100, timeSince(start), loss))

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

print("save rnn model to conditional-char-rnn.pt")
torch.save(rnn, 'conditional-char-rnn.pt')

import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)

plt.show()
