from utils import get_mask
import os
import codecs
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from sklearn.metrics import f1_score

"""def calc_score(y_true, y_pred, beta=0.5):
    TP = 0
    FP = 0
    FN = 0
    for i in range(len(y_true)):
        y_true_i = y_true[i]
        y_pred_i = y_pred[i]
        FP += len(y_pred_i)
        for j in range(len(y_true_i)):
            if y_true_i[j] in y_pred_i:
                TP += 1
                FP -= 1
            else:
                FN += 1
    F_beta = (1+beta**2)*TP/((1+beta**2)*TP + beta**2*FN + FP)
    return F_beta"""

def train_model(dataloader, model, optimizer, batch_num, writer, use_gpu=False):
    model.train()
    for batch in tqdm(dataloader):
        batch_num += 1
        model.zero_grad()
        batch_text, seq_length, word_perm_idx = batch['text']
        batch_label, _, _ = batch['label']
        char_inputs = batch['char']
        char_inputs = char_inputs[word_perm_idx]
        char_dim = char_inputs.size(-1)
        char_inputs = char_inputs.contiguous().view(-1, char_dim)
        if use_gpu:
            batch_text = batch_text.cuda()
            batch_label = batch_label.cuda()
            char_inputs = char_inputs.cuda()
        mask = get_mask(batch_text)
        loss = model.neg_log_likelihood_loss(batch_text, seq_length, char_inputs, batch_label, mask)
        #writer.add_scalar('loss', loss, batch_num)
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        del loss, mask

    return batch_num


def evaluate(dataloader, model, word_vocab, label_vocab, pred_file, score_file, eval_script, use_gpu=False):
    model.eval()
    prediction = []
    for batch in dataloader:
        batch_text, seq_length, word_perm_idx = batch['text']
        batch_label, _, _ = batch['label']
        char_inputs = batch['char']
        char_inputs = char_inputs[word_perm_idx]
        char_dim = char_inputs.size(-1)
        char_inputs = char_inputs.contiguous().view(-1, char_dim)
        if use_gpu:
            batch_text = batch_text.cuda()
            batch_label = batch_label.cuda()
            char_inputs = char_inputs.cuda()
        mask = get_mask(batch_text)
        with torch.no_grad():
            tag_seq = model(batch_text, seq_length, char_inputs, batch_label, mask)

        for line_tesor, labels_tensor, predicts_tensor in zip(batch_text, batch_label, tag_seq):
            for word_tensor, label_tensor, predict_tensor in zip(line_tesor, labels_tensor, predicts_tensor):
                if word_tensor.item() == 0:
                    break
                line = ' '.join(
                    [word_vocab.id_to_word(word_tensor.item()), label_vocab.id_to_label(label_tensor.item()),
                     label_vocab.id_to_label(predict_tensor.item())])
                prediction.append(line)
            #prediction.append('')

    prediction = [p.split(' ') for p in prediction]
    labels = [p[1] for p in prediction]
    preds = [p[2] for p in prediction]

    score = f1_score(labels, preds, pos_label= '1', average= 'binary')

    """with open(pred_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, pred_file, score_file))

    eval_lines = [l.rstrip() for l in codecs.open(score_file, 'r', 'utf8')]
    new_f1 = -1
    for i, line in enumerate(eval_lines):
        print(line)
        if i == 1:
            new_f1 = float(line.strip().split()[-1])
            break"""

    return score
