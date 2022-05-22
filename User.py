import json
import numpy as np
import datetime
from EmotuionClassifier import Classifier


clf = Classifier()


class User(object):

    def __init__(self, message_chat_id, username):
        self.results = {}  # Временные записи состояния
        self.session = message_chat_id  # Id чата
        self.temper = np.array([0, 0, 0, 0])  # Темперамент
        self.emotions = None  # Эмоции которые ощущает пользователь прямо сейчас
        self.chat_state = 1  # Состояние общения в текущий момент
        self.additional_chat_state = None
        self.username = username

    def get_message(self):
        return self.message

    def get_results(self):
        return self.results

    def get_my_average(self, from_date='01/01/01/01/01/1980'):
        from_date = datetime.datetime.strptime(from_date, '%S/%M/%H/%d/%m/%Y')
        most_current_num = 0
        emotions_average = {'sadness': 0,
                            'joy': 0,
                             'love': 0,
                             'anger': 0,
                             'fear': 0,
                             'surprise': 0
        }

        for entry in self.results:
            if datetime.datetime.strptime(entry, '%S/%M/%H/%d/%m/%Y') >= from_date:
                for emotion in clf.get_sent(self.results[entry]):
                    emotions_average[emotion] += float(clf.get_sent(self.results[entry])[emotion])
                most_current_num += 1

        for emotion in emotions_average:
            emotions_average[emotion] /= float(most_current_num)

        return emotions_average

    def get_result_by_date(self, date):
        return self.results[date]

    def put_result(self, date):
        self.results[date] = "Result"

    def __repr__(self):
        return self.username

