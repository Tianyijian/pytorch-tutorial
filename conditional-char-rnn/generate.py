from data import *
import sys

max_length = 20
n_hidden = 128

# gpu cuda
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
rnn = torch.load('conditional-char-rnn.pt', map_location=device)
rnn.to(device)


def sample(category, start_letter='A'):
    with torch.no_grad():
        category_tensor = categoryTensor(category)
        input = lineToTensor(start_letter)
        hidden = rnn.initHidden()

        category_tensor = category_tensor.to(device)
        input = input.to(device)
        hidden = hidden.to(device)

        output_name = start_letter
        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = lineToTensor(letter)
            input = input.to(device)

        return output_name


def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))


if __name__ == '__main__':
    # samples('Russian', 'RUS')
    # samples('German', 'GER')
    # samples('Spanish', 'SPA')
    # samples('Chinese', 'CHI')

    if len(sys.argv) < 2:
        print("Usage: generate.py [language]")
        sys.exit()
    else:
        language = sys.argv[1]
        samples(language)
