import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import torch
import transformers
import datetime


class Classifier:
    def __init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion", use_fast=True)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            "nateraw/bert-base-uncased-emotion").cuda()

        self.pred = transformers.pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=0,
                                          return_all_scores=True)

    def to_integer(self, str_time):
        dt_time = datetime.datetime.strptime(str_time, '%S/%M/%H/%d/%m/%Y')
        return 10000000000 * dt_time.year + 100000000 * dt_time.month + 1000000 * dt_time.day + 10000 * dt_time.hour + 100 * dt_time.minute + dt_time.second

    def get_sent(self, text):
        sents = self.pred([text])[0]
        answer = {}
        for sent in sents:
            emotion = sent['label']
            score = sent['score']
            answer[emotion] = score
        return answer

    def get_plot(self, messages, user_id):
        ys, xs = [], []
        for date in messages:
            xs.append(self.to_integer(date))
            ys.append(self.get_sent(messages[date]))

        datasets_dict = {}
        plot_dicts = []

        for emotion in list(ys[0].keys()):
            datasets_dict[emotion] = []
            for point in ys:
                datasets_dict[emotion].append(point[emotion])

        for emotion in datasets_dict:
            plot_dicts.append({'Time': xs, f'{emotion}_level': datasets_dict[emotion]})

        # Data
        sadness_level = pd.DataFrame(plot_dicts[0])
        joy_level = pd.DataFrame(plot_dicts[1])
        love_level = pd.DataFrame(plot_dicts[2])
        anger_level = pd.DataFrame(plot_dicts[3])
        fear_level = pd.DataFrame(plot_dicts[4])
        surprise_level = pd.DataFrame(plot_dicts[5])

        plt.legend(loc='upper left')

        fig = plt.figure(figsize=(15, 10))

        # multiple line plot
        plt.plot('Time', 'sadness_level', data=sadness_level, marker='o', color='blue', linewidth=2)
        plt.plot('Time', 'love_level', data=love_level, marker='o', color='red', linewidth=2)
        plt.plot('Time', 'joy_level', data=joy_level, marker='o', color='green', linewidth=2)
        plt.plot('Time', 'fear_level', data=fear_level, marker='o', color='brown', linewidth=2)
        plt.plot('Time', 'anger_level', data=anger_level, marker='o', color='black', linewidth=2)
        plt.plot('Time', 'surprise_level', data=surprise_level, marker='o', color='orange', linewidth=2)

        fig.savefig(f'img/{str(user_id)}.png')
