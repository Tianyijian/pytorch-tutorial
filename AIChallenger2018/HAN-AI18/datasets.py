from torch.utils.data import Dataset, DataLoader
import torch
import os


class HANDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_folder, split, aspect):
        """
        :param data_folder: folder where data files are stored
        :param split: data file name
        :param aspect: 
        """
        self.split = split
        self.aspect = aspect

        # Load data
        self.data = torch.load(os.path.join(data_folder, split))

        print(split + "  Aspect: " + str(aspect))
        for i in range(3):
            print("id:{} label:{}".format(self.data['ids'][i], self.data['labels'][i][aspect]))

    def __getitem__(self, i):
        return torch.LongTensor(self.data['docs'][i]), \
               torch.LongTensor([self.data['sentences_per_document'][i]]), \
               torch.LongTensor(self.data['words_per_sentence'][i]), \
               torch.LongTensor([self.data['labels'][i][self.aspect]])

    def __len__(self):
        return len(self.data['labels'])
