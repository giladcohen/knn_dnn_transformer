import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import dataset
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import copy
import time
import argparse
import json
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from huawei_project.utils import generate_square_subsequent_mask, batchify, get_batch
from huawei_project.models import TransformerModel
from research.utils import set_logger

parser = argparse.ArgumentParser(description='Training networks using PyTorch')
parser.add_argument('--checkpoint_dir', default='/disk5/gilad/logs/huawei/debug2', type=str, help='checkpoint dir')

# optimization:
parser.add_argument('--lr', default=5.0, type=float, help='learning rate')
parser.add_argument('--epochs', default=300, type=int, help='number of epochs')

# knn:
parser.add_argument('--k', default=3, type=int, help='k nearest neighbors')

# LR scheduler
parser.add_argument('--lr_scheduler', default='reduce_on_plateau', type=str, help='reduce_on_plateau/multi_step')
parser.add_argument('--factor', default=0.75, type=float, help='LR schedule factor')
parser.add_argument('--patience', default=2, type=int, help='LR schedule patience')
parser.add_argument('--cooldown', default=0, type=int, help='LR cooldown')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)
# dumping args to txt file
os.makedirs(args.checkpoint_dir, exist_ok=True)
with open(os.path.join(args.checkpoint_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = os.path.join(args.checkpoint_dir, 'ckpt.pth')
log_file = os.path.join(args.checkpoint_dir, 'log.log')
set_logger(log_file)
logger = logging.getLogger()

logger.info('==> Preparing data..')
train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# train_iter was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

train_writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'train'))
val_writer   = SummaryWriter(os.path.join(args.checkpoint_dir, 'val'))
test_writer  = SummaryWriter(os.path.join(args.checkpoint_dir, 'test'))

bptt = 35
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
logger.info('==> Building model..')
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = args.lr
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=args.factor,
    patience=args.patience,
    verbose=True,
    cooldown=args.cooldown
)

knn = KNeighborsClassifier(args.k, n_jobs=20)


def train(model: nn.Module) -> None:
    global global_step
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    predicted = []
    labels = []
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        batch_size = data.size(0)
        if batch_size != bptt:  # only on last batch
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(data, src_mask, field='logits')
        logits = output.view(-1, ntokens)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        _, preds = logits.max(1)
        preds = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        predicted.extend(preds)
        labels.extend(targets_np)
        num_corrected = np.sum(preds == targets_np)
        acc = num_corrected / targets.size(0)

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = optimizer.param_groups[0]['lr']
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            train_writer.add_scalar('losses/cross_entropy', loss, global_step)
            train_writer.add_scalar('metrics/acc', 100.0 * acc, global_step)
            train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            train_writer.add_scalar('ms_per_batch', ms_per_batch, global_step)
            start_time = time.time()
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                      f'loss {loss:5.2f} | acc {acc}')

        global_step += 1

    N = batch + 1
    train_loss = total_loss / N
    predicted = np.asarray(predicted)
    labels = np.asarray(labels)
    train_acc = 100.0 * np.mean(predicted == labels)
    logger.info('Epoch #{} (TRAIN): lr={}\t loss={}\tacc={:.4f}'.format(epoch, lr, train_loss, train_acc))


def evaluate(model: nn.Module, eval_data: Tensor) -> None:
    global best_acc, best_model, global_step

    model.eval()  # turn on evaluation mode
    predicted = []
    labels = []
    epoch_start_time = time.time()
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            batch_size = data.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask, field='logits')
            logits = output.view(-1, ntokens)
            total_loss += batch_size * criterion(logits, targets).item()

            _, preds = logits.max(1)
            preds = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            predicted.extend(preds)
            labels.extend(targets_np)

    val_loss = total_loss / (len(eval_data) - 1)
    predicted = np.asarray(predicted)
    labels = np.asarray(labels)
    val_acc = 100.0 * np.mean(predicted == labels)

    val_writer.add_scalar('losses/loss', val_loss, global_step)
    val_writer.add_scalar('metrics/acc', val_acc, global_step)

    if val_acc > best_acc:
        best_acc = val_acc
        logger.info('Found new best model!')
        best_model = copy.deepcopy(model)

    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'val loss {val_loss:5.2f} | val acc {val_acc:8.2f}')
    print('-' * 89)

    logger.info('Epoch #{} (VAL): loss={}\tacc={:.4f}\tbest_acc={}'.format(epoch, val_loss, val_acc, best_acc))
    scheduler.step(metrics=val_acc)


def test(model: nn.Module, test_data: Tensor) -> None:
    global best_acc, best_model, global_step

    model.eval()  # turn on evaluation mode
    predicted = []
    labels = []
    epoch_start_time = time.time()
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, test_data.size(0) - 1, bptt):
            data, targets = get_batch(test_data, i, bptt)
            batch_size = data.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask, field='logits')
            logits = output.view(-1, ntokens)
            total_loss += batch_size * criterion(logits, targets).item()

            _, preds = logits.max(1)
            preds = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            predicted.extend(preds)
            labels.extend(targets_np)

    test_loss = total_loss / (len(test_data) - 1)
    predicted = np.asarray(predicted)
    labels = np.asarray(labels)
    test_acc = 100.0 * np.mean(predicted == labels)

    test_writer.add_scalar('losses/loss', test_loss, global_step)
    test_writer.add_scalar('metrics/acc', test_acc, global_step)

    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'test loss {test_loss:5.2f} | test acc {test_acc:8.2f}')
    print('-' * 89)

    logger.info('Epoch #{} (TEST): loss={}\tacc={:.4f}'.format(epoch, test_loss, test_acc))

def train_knn():
    global knn
    model.eval()  # turn on eval mode
    train_embeddings = []
    labels = []
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    with torch.no_grad():
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i, bptt)
            batch_size = data.size(0)
            if batch_size != bptt:  # only on last batch
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask, field='embeddings')
            train_embeddings.extend(output.cpu().numpy().reshape(-1, 200))
            targets_np = targets.cpu().numpy()
            labels.extend(targets_np)

    train_embeddings = np.asarray(train_embeddings)
    labels = np.asarray(labels)
    knn.fit(train_embeddings, labels)

def test_knn():
    global knn
    model.eval()  # turn on eval mode
    test_embeddings = []
    labels = []
    probs_list = []
    logits_list = []
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    with torch.no_grad():
        for batch, i in enumerate(range(0, test_data.size(0) - 1, bptt)):
            if batch < 2:
                print('calc for batch={}'.format(batch))
                data, targets = get_batch(test_data, i, bptt)
                batch_size = data.size(0)
                if batch_size != bptt:  # only on last batch
                    src_mask = src_mask[:batch_size, :batch_size]
                embeddings, logits = model(data, src_mask, field=None)
                logits = logits.view(-1, ntokens)
                logits_list.extend(logits.cpu().numpy())
                probs = F.softmax(logits, dim=1).cpu().numpy()
                embeddings = embeddings.cpu().numpy().reshape(-1, 200)

                probs_list.extend(probs)
                test_embeddings.extend(embeddings)
                targets_np = targets.cpu().numpy()
                labels.extend(targets_np)

    test_embeddings = np.asarray(test_embeddings)
    labels = np.asarray(labels)
    logits = np.asarray(logits_list)
    knn_probs = knn.predict_proba(test_embeddings)
    knn_acc = np.mean(knn_probs.argmax(axis=1) == labels)
    logits_t = torch.tensor(logits)
    knn_probs_t = torch.tensor(knn_probs)
    kl = nn.KLDivLoss(reduction='batchmean')
    kl_knn_dnn = kl(F.log_softmax(logits_t, dim=1), knn_probs_t)

    test_writer.add_scalar('metrics/knn_acc', knn_acc, global_step)
    test_writer.add_scalar('metrics/knn_dnn_kl', kl_knn_dnn, global_step)

def flush():
    train_writer.flush()
    val_writer.flush()
    test_writer.flush()
    logger.handlers[0].flush()


best_model = None
global_step = 0
best_acc = 0.0
epoch = 0

logger.info('Testing epoch 0')
test(model, test_data)

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train(model)
    evaluate(model, val_data)
    train_knn()
    test(model, test_data)
    test_knn()
flush()


# # stats
# train_data_1d = train_data.cpu().numpy().ravel()
# counts = np.bincount(train_data_1d)
