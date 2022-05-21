from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import pandas as pd
import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from transformers import GPT2Tokenizer


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 1000
EOS_token = 1
SOS_token = 0
hidden_size = 256


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def __repr__(self):
        most_popular_words = sorted(
            self.word2count.keys(), key=lambda word: self.word2count[word], reverse=True
        )[:10]
        most_popular_words = ", ".join(most_popular_words)
        return f"Language: {self.name} | Num words: {self.n_words} | Most popular: {most_popular_words}"


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")
    df = pd.read_csv('happiness_provokers', encoding='utf-8')

    # Get pairs and normalize
    pairs1 = list(zip([normalizeString(s) for s in list(df['query'].values)],
                     [normalizeString(s) for s in list(df.reply.values)]))

    # Reverse pairs, make Lang instances
    if reverse:
        pairs1 = [list(reversed(p)) for p in pairs1]
        input_lang1 = Lang(lang2)
        output_lang1 = Lang(lang1)
    else:
        input_lang1 = Lang(lang1)
        output_lang1 = Lang(lang2)

    return input_lang1, output_lang1, pairs1


def prepareData(lang1, lang2, reverse=False):
    input_lang2, output_lang2, pairs2 = readLangs(lang1, lang2, reverse)
    pairs2 = pairs2[::50]
    print("Read %s sentence pairs" % len(pairs2))
    print("Counting words...")
    for pair in pairs2:
        input_lang2.addSentence(pair[0])
        output_lang2.addSentence(pair[1])
    print("Counted words:")
    print(input_lang2.name, input_lang2.n_words)
    print(output_lang2.name, output_lang2.n_words)
    return input_lang2, output_lang2, pairs2


input_lang, output_lang, pairs = prepareData('eng_user', 'eng_emotion_provoker')


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # num_embedding = vocab_size_fra
        self.embedder = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # (batch_size, num_words) -> (batch_size, num_words, dim_1)
        embeddings = self.embedder(input).view(1, 1, -1)
        # (batch_size, num_words, dim_2)
        output, hidden = self.gru(embeddings, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


tokens = torch.randint(0, 1000, size=(128, 40))
embedder = nn.Embedding(1000, 128)  # здесь лежит матрица
onehot = torch.nn.functional.one_hot(tokens, num_classes=1000)
embeddings_first_way = embedder(tokens)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedder = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # (batch_size, num_words, dim)
        # (1, 1, num_words * dim)
        output = self.embedder(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # (batch_size, num_words, dim) -> (batch_size, num_words, num_classes)
        # (batch_size, num_words, vocab_size_eng)
        output = self.out(output[0])
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(
                torch.cat((embedded[0], hidden[0]), 1)
            ),
            dim=1
        )
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0),
            encoder_outputs.unsqueeze(0)
        )

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Generator:
    def __init__(self):
        self.encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
        self.attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

        self.encoder1.load_state_dict(torch.load('blablaenc.pt'))
        self.attn_decoder1.load_state_dict(torch.load('blabladeco.pt'))

    def indexesFromSentence(self, lang, sentence):
        answer = []
        for word in sentence.split(' '):
            added = None
            try:
                added = lang.word2index[word]
            except KeyError:
                added = random.randint(1, 100)
            answer.append(added)
        return answer

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def tensorsFromPair(self, pair):
        input_tensor = self.tensorFromSentence(input_lang, pair[0])
        target_tensor = self.tensorFromSentence(output_lang, pair[1])
        return (input_tensor, target_tensor)

    def evaluate(self, encoder, decoder, sentence, max_length=MAX_LENGTH):
        with torch.no_grad():
            input_tensor = self.tensorFromSentence(input_lang, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                         encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(max_length, max_length)

            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    def generator(self, text):
        text = normalizeString(text)
        output_words, _ = self.evaluate(self.encoder1, self.attn_decoder1, text)
        output_message = ' '.join(output_words).capitalize()
        output_message = output_message.replace('<eos>', '')
        return output_message
