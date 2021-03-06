import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import torch
import random
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
        return 60 * 60 * 24 * 31 * 365 * dt_time.year + 60 * 60 * 24 * 31 * dt_time.month + 60 * 60 * 24 * dt_time.day + 60 * 60 * dt_time.hour + 60 * dt_time.minute + dt_time.second

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

        plt.legend(['sadness', 'love', 'joy', 'fear', 'anger', 'surprise'])

        fig.savefig(f'img/{str(user_id)}.png')

    def get_temper(self, vector, user_id):
        import numpy as np
        import matplotlib.pyplot as plt

        x = ['choleric', 'sanguine', 'phlegmatic', 'melancholic']
        y = np.exp(np.array(vector) / np.array(vector).max())

        fig, ax = plt.subplots()

        ax.bar(x, y)

        ax.set_facecolor('seashell')
        fig.set_facecolor('floralwhite')
        fig.set_figwidth(12)
        fig.set_figheight(6)

        fig.savefig(f'img/{str(user_id)}.png')

    def get_difference(self, vector_user, vector_average, user_id):
        import numpy as np
        import matplotlib.pyplot as plt

        x = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        y = [vector_user[u] - vector_average[a] for u, a in zip(list(vector_user.keys()), list(vector_average.keys()))]

        fig, ax = plt.subplots()

        ax.bar(x, y)

        ax.set_facecolor('seashell')
        fig.set_facecolor('floralwhite')
        fig.set_figwidth(12)
        fig.set_figheight(6)

        fig.savefig(f'img/av{str(user_id)}.png')

    def get_reccomendations(self, vector_user, vector_average, feelings_d):
        feelings_of_interest = ['sadness', 'anger', 'fear']
        recommends = []
        for feeling in feelings_of_interest:
            if abs(vector_user[feeling] - vector_average[feeling]) >= feelings_d:
                if abs(vector_user[feeling] - vector_average[feeling]) >= 2 * feelings_d:
                    recommends.append(random.choice([f'I strongly recommend to pay attention on your {feeling} feelings',
                                                     f'It seems you suffer from {feeling}. Take care!']))
                else:
                    recommends.append(
                        random.choice([f'Your {feeling} seems to be unusual. Go for a walk and enjoy the life',
                                       f'{feeling} is your enemy!']))
        return recommends

    def dist_from_freq(self, messages, messages1):
        from scipy.fft import fft
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
        print(plot_dicts[3])
        print(datasets_dict)
        x, y = plot_dicts[3]['Time'], plot_dicts[3]['anger_level']
        yf = fft(y)

        ys1, xs1 = [], []
        for date in messages1:
            xs1.append(self.to_integer(date))
            ys1.append(self.get_sent(messages1[date]))

        datasets_dict1 = {}
        plot_dicts1 = []

        for emotion in list(ys1[0].keys()):
            datasets_dict1[emotion] = []
            for point in ys1:
                datasets_dict1[emotion].append(point[emotion])

        for emotion in datasets_dict1:
            plot_dicts1.append({'Time': xs1, f'{emotion}_level': datasets_dict1[emotion]})

        x1, y1 = plot_dicts1[3]['Time'], plot_dicts1[3]['anger_level']
        yf1 = fft(y1)

        return np.mean(np.diff(np.array(yf), n=len(np.array(yf)) - 1) - np.diff(np.array(yf1), n=len(np.array(yf1)) - 1))

    def get_closest_users(self, users, user, n_users=10):
        if len(user.results) > 0:
            n_users = min(n_users, len(users))
            top = []
            sou = {}
            for user_id in users:
                if user_id != 0 and user_id != user.session:
                    user1 = users[user_id]
                    d = self.dist_from_freq(user.results, user1.results)
                    if d not in sou:
                        sou[d] = []
                    sou[d].append(user1)
            for i in range(min(n_users, len(sorted(list(sou.keys()), reverse=True)))):
                for u in sou[sorted(list(sou.keys()), reverse=True)[i]]:
                    if len(top) < n_users:
                        top.append(u)
            return top
        return None