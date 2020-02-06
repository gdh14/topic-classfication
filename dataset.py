from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TopicTextData(Dataset):
    """
    The dataset for the topic classification.

    Args:
        tokens_id_ls: A list of tokens that has been converted to index
                        based on the vocabulary build on training set.
                        [[t11, t12, t13, ...], [t21, t22, ...], ...]
        label_ls: A list of topic labels in index format. 
                    [topic_idx_1, topic_idx_2, ...]
    """
    def __init__(self, tokens_id_ls, label_ls):
        self.tokens_id_ls = tokens_id_ls
        self.label_ls = label_ls
    
    def __len__(self):
        return len(self.label_ls)

    def __getitem__(self, index):
        tokens_id_tensor = torch.LongTensor(self.tokens_id_ls[index])

        label = self.label_ls[index]
        return tokens_id_tensor, label

def collate_with_padding(batch):
    tokens_id_tensor_ls = [item[0] for item in batch]
    label_ls = [item[1] for item in batch]

    tokens_id_tensor_padded = (pad_sequence(tokens_id_tensor_ls)
                              .transpose(1, 0)) # (B, max_sent_len)
    label_tensor = torch.LongTensor(label_ls)

    return tokens_id_tensor_padded, label_ls
