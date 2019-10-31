from MODEL1 import RNN
from model2 import CNN
from data import getdata
import torch
import torch.optim as optim
import torch.nn as nn
import time
import spacy


def predict_sentiment(model, sentence, min_len=5):
    train_iterator, valid_iterator, test_iterator, INPUT_DIM, prem, pad_idx, unk_idx, TEXT, label = getdata()
    nlp = spacy.load('en')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def main():
    train_iterator, valid_iterator, test_iterator, INPUT_DIM, prem, pad_idx, unk_idx, text, label = getdata()
    choice = input('which model you would like to use?')
    if choice == 'rnn':

        EMBEDDING_DIM = 100
        HIDDEN_DIM = 256
        OUTPUT_DIM = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
        optimizer = optim.SGD(model.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        model = model.to(device)
        criterion = criterion.to(device)
        N_EPOCHS = 5
        testing = input('Would you like to test ?')
        try:
            if testing == 'yes':
                sentence = input('please enter the sentence')
                model.load_state_dict(torch.load('tut1-model.pt'))
                print(predict_sentiment(model, sentence))
            if testing == 'no':
                best_valid_loss = float('inf')
                for epoch in range(N_EPOCHS):

                    start_time = time.time()

                    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
                    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

                    end_time = time.time()

                    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        torch.save(model.state_dict(), 'tut1-model.pt')

                    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
                    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        except:
            print('something went wrong')

    if choice == 'cnn':
        EMBEDDING_DIM = 100
        N_FILTERS = 100
        FILTER_SIZES = [3, 4, 5]
        OUTPUT_DIM = 1
        DROPOUT = 0.5
        model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, pad_idx)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.embedding.weight.data[unk_idx] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[pad_idx] = torch.zeros(EMBEDDING_DIM)
        optimizer = optim.Adam(model.parameters())

        criterion = nn.BCEWithLogitsLoss()

        model = model.to(device)
        criterion = criterion.to(device)
        N_EPOCHS = 5
        try:
            if testing == 'yes':
                sentence = input('please enter the sentence')
                model.load_state_dict(torch.load('tut2-model.pt'))
                print(predict_sentiment(model, sentence))

            if testing == 'no':
                best_valid_loss = float('inf')

                for epoch in range(N_EPOCHS):

                    start_time = time.time()

                    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
                    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

                    end_time = time.time()

                    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        torch.save(model.state_dict(), 'tut2-model.pt')

                    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
                    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

        except:
            print('something went wrong')


if __name__ == '__main__':
    main()
