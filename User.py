import json


class User(object):

    def __init__(self, message_chat_id):
        self.results = {}  # Временные записи состояния
        self.session = message_chat_id  # Id чата
        self.temper = None  # Темперамент
        self.emotions = None  # Эмоции которые ощущает пользователь прямо сейчас
        self.chat_state = 1  # Состояние общения в текущий момент
        self.additional_chat_state = None


    def get_message(self):
        return self.message

    def get_results(self):
        return self.results

    def get_result_by_date(self, date):
        return self.results[date]

    def put_result(self, date):
        self.results[date] = "Result"

    def __repr__(self):
        return json.JSONEncoder()

