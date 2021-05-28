import random
import torch
import numpy as np
import argparse
import os
from utils import WordVocabulary, LabelVocabulary, Alphabet, build_pretrain_embedding, my_collate_fn, lr_decay
import time
from dataset import MyDataset, MyLoaderDataset
from torch.utils.data import DataLoader
from model import NamedEntityRecog
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from train import train_model, evaluate
import random
import pickle

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

with open(f'C:\projects\personal\kaggle\kaggle_coleridge_initiative\data/train_idx.pkl', 'rb') as f:
    train_idx = pickle.load(f)

with open(f'C:\projects\personal\kaggle\kaggle_coleridge_initiative\data/val_idx.pkl', 'rb') as f:
    val_idx = pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Named Entity Recognition Model')
    parser.add_argument('--word_embed_dim', type=int, default=100)
    parser.add_argument('--word_hidden_dim', type=int, default=100)
    parser.add_argument('--char_embedding_dim', type=int, default=50)
    parser.add_argument('--char_hidden_dim', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pretrain_embed_path', default='data/glove.6B.100d.txt')
    parser.add_argument('--savedir', default='data/model/')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--feature_extractor', choices=['lstm', 'cnn'], default='lstm')
    parser.add_argument('--use_char', type=bool, default=True)
    parser.add_argument('--train_path', default='data/train')
    parser.add_argument('--dev_path', default='data/val')
    parser.add_argument('--test_path', default='data/val')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--number_normalized', type=bool, default=True)
    parser.add_argument('--use_crf', type=bool, default=True)
    parser.add_argument('--save_vocabularies', type=bool, default=True)
    parser.add_argument('--n_train_repeat', type=int, default=1)
    args = parser.parse_args()

    # Get filenames
    """train_example_names = [fn.split('.')[0] for fn in os.listdir('C:\projects\personal\kaggle\kaggle_coleridge_initiative\data\processed_data')]
    print(f'# train examples : {len(train_example_names)}')
    docIdx = train_example_names.copy()
    random.seed(42)
    random.shuffle(docIdx)"""

    """train_ratio = 0.85
    n_train = int(len(docIdx) * train_ratio)
    n_val = len(docIdx) - n_train

    train_idx = docIdx[:n_train]
    val_idx = docIdx[n_train:]

    # Repeat training examples
    train_idx_rep = []
    for _ in range(args.n_train_repeat):
        train_idx_rep.extend(train_idx)

    train_idx = train_idx_rep"""

    ##########################

    
    use_gpu = torch.cuda.is_available()
    print('use_crf:', args.use_crf)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    eval_path = "evaluation"
    eval_temp = os.path.join(eval_path, "temp")
    eval_script = os.path.join(eval_path, "conlleval")

    """if not os.path.isfile(eval_script):
        raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)"""
    if not os.path.exists(eval_temp):
        os.makedirs(eval_temp)

    pred_file = eval_temp + '/pred.txt'
    score_file = eval_temp + '/score.txt'

    model_name = args.savedir + '/' + args.feature_extractor + str(args.use_char) + str(args.use_crf)

    print('Preparing vocabularies...')
    word_vocab = WordVocabulary(train_idx, args.number_normalized, args.save_vocabularies)
    label_vocab = LabelVocabulary(train_idx[:10], args.save_vocabularies)
    alphabet = Alphabet(train_idx, args.save_vocabularies)

    print('Building pretrain embedding...')
    emb_begin = time.time()
    pretrain_word_embedding = build_pretrain_embedding(args.pretrain_embed_path, word_vocab, args.word_embed_dim)
    emb_end = time.time()
    emb_min = (emb_end - emb_begin) % 3600 // 60
    print('build pretrain embed cost {}m'.format(emb_min))

    print('Creating datasets...')
    train_dataset = MyLoaderDataset(train_idx, word_vocab, label_vocab, alphabet, args.number_normalized, augment_chance = 0.8)
    dev_dataset = MyLoaderDataset(val_idx, word_vocab, label_vocab, alphabet, args.number_normalized, augment_chance = 0.0)
    #test_dataset = MyDataset(args.test_path, word_vocab, label_vocab, alphabet, args.number_normalized)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)

    print('Creating NER model...')
    model = NamedEntityRecog(word_vocab.size(), args.word_embed_dim, args.word_hidden_dim, alphabet.size(),
                             args.char_embedding_dim, args.char_hidden_dim,
                             args.feature_extractor, label_vocab.size(), args.dropout,
                             pretrain_embed=pretrain_word_embedding, use_char=args.use_char, use_crf=args.use_crf,
                             use_gpu=use_gpu)
    if use_gpu:
        print('Using GPU.')
        model = model.cuda()
    else:
        print('Using CPU.')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    train_begin = time.time()
    print('train begin', '-' * 50)
    print()
    print()

    writer = SummaryWriter('log')
    batch_num = -1
    best_f1 = -1
    early_stop = 0

    print('Starting training...')
    for epoch in range(args.epochs):
        epoch_begin = time.time()
        print('train {}/{} epoch'.format(epoch + 1, args.epochs))
        optimizer = lr_decay(optimizer, epoch, 0.05, args.lr)
        batch_num = train_model(train_dataloader, model, optimizer, batch_num, writer, use_gpu)
        new_f1 = evaluate(dev_dataloader, model, word_vocab, label_vocab, pred_file, score_file, eval_script, use_gpu)
        print('f1 is {} at {}th epoch on dev set'.format(new_f1, epoch + 1))
        print('new f1:', new_f1)
        if new_f1 > best_f1:
            best_f1 = new_f1
            print('new best f1 on dev set:', best_f1)
            early_stop = 0
            torch.save(model.state_dict(), model_name)
        else:
            early_stop += 1

        epoch_end = time.time()
        cost_time = epoch_end - epoch_begin
        print('train {}th epoch cost {}m {}s'.format(epoch + 1, int(cost_time / 60), int(cost_time % 60)))
        print()

        if early_stop > args.patience:
            print('early stop')
            break

    train_end = time.time()
    train_cost = train_end - train_begin
    hour = int(train_cost / 3600)
    min = int((train_cost % 3600) / 60)
    second = int(train_cost % 3600 % 60)
    print()
    print()
    print('train end', '-' * 50)
    print('train total cost {}h {}m {}s'.format(hour, min, second))
    print('-' * 50)

    """model.load_state_dict(torch.load(model_name))
    test_acc = evaluate(test_dataloader, model, word_vocab, label_vocab, pred_file, score_file, eval_script, use_gpu)
    print('test acc on test set:', test_acc)"""
