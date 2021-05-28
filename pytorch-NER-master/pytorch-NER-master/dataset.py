from torch.utils.data import Dataset
from utils import normalize_word
import torch
import pickle

class MyLoaderDataset(Dataset):
    def __init__(self, filenames, word_vocab, label_vocab, alphabet, number_normalized, augment_chance = 0.0):
        self.filenames = filenames
        self.augment_chance = augment_chance

        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.alphabet = alphabet
        self.number_normalized = number_normalized
        self.sentences = []
        self.labels = []

        for doc_id in filenames:
            with open(f'C:\projects\personal\kaggle\kaggle_coleridge_initiative\data/crf_x_data/{doc_id}.pkl', 'rb') as f:
                doc_X = pickle.load(f)
                self.sentences.extend(doc_X)

            with open(f'C:\projects\personal\kaggle\kaggle_coleridge_initiative\data/crf_y_data/{doc_id}.pkl', 'rb') as f:
                doc_y = pickle.load(f)
                self.labels.extend(doc_y)

        assert len(self.labels) == len(self.sentences)


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        text_id = []
        label_id = []

        text = self.sentences[item]
        label = self.labels[item]

        text = reversed(text)
        label = reversed(label)

        seq_char_list = list()

        for word in text:
            text_id.append(self.word_vocab.word_to_id(word))
        text_tensor = torch.tensor(text_id).long()
        for label_ele in label:
            label_id.append(self.label_vocab.label_to_id(label_ele))
        label_tensor = torch.tensor(label_id).long()

        for word in text:
            char_list = list(word)
            char_id = list()
            for char in char_list:
                char_id.append(self.alphabet.char_to_id(char))
            seq_char_list.append(char_id)
        
        return {'text': text_tensor, 'label': label_tensor, 'char': seq_char_list}


class MyDataset(Dataset):
    def __init__(self, file_path, word_vocab, label_vocab, alphabet, number_normalized):
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.alphabet = alphabet
        self.number_normalized = number_normalized
        texts, labels = [], []
        text, label = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if len(line) > 2:
                    pairs = line.strip().split()
                    word = pairs[0]
                    if self.number_normalized:
                        word = normalize_word(word)
                    text.append(word)
                    label.append(pairs[-1])

                else:
                    if len(text) > 0:
                        texts.append(text)
                        labels.append(label)

                    text, label = [], []

        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        text_id = []
        label_id = []
        text = self.texts[item]
        label = self.labels[item]
        seq_char_list = list()

        for word in text:
            text_id.append(self.word_vocab.word_to_id(word))
        text_tensor = torch.tensor(text_id).long()
        for label_ele in label:
            label_id.append(self.label_vocab.label_to_id(label_ele))
        label_tensor = torch.tensor(label_id).long()

        for word in text:
            char_list = list(word)
            char_id = list()
            for char in char_list:
                char_id.append(self.alphabet.char_to_id(char))
            seq_char_list.append(char_id)

        print(len(seq_char_list))
        return {'text': text_tensor, 'label': label_tensor, 'char': seq_char_list}
