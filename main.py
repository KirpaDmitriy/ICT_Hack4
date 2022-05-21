import random

import telebot
import config
from dateutil.parser import parse
from User import User
import keyboard
import datetime
from telebot import types
import json
from EmotuionClassifier import Classifier
from TextGenerator import Generator
bot = telebot.TeleBot(config.TOKEN)

users = {}

emotion_classifier = Classifier()

text_generator = Generator()

print("I'm alive\n")


@bot.message_handler(commands=['start'])
def welcome(message):
    if message.chat.id in users:
        bot.send_message(message.chat.id, "You are nice <3")
    else:
        users[message.chat.id] = User(message.chat.id)
    bot.send_message(message.chat.id, "Hello my name is Ben", reply_markup=keyboard.markup_after_start)


@bot.message_handler(content_types=['text'])
def result(message):
    if message.chat.type == 'private':
        print(users[message.chat.id].chat_state)
        if users[message.chat.id].chat_state == 2 and message.text != 'Polls' and message.text != 'Stats' and message.text != 'Dialogue' and message.text != 'Stop':
            now = datetime.datetime.now()
            string_now = now.strftime('%S/%M/%H/%d/%m/%Y')
            users[message.chat.id].results[string_now] = message.text
            bot.send_message(message.chat.id, text_generator.generator(message.text))

        if users[message.chat.id].chat_state == 4:
            bot.send_message(message.chat.id, 'Beeeeeeeeeeeeeeeen', reply_markup=keyboard.markup_after_dialogue)
            foo = ['1', '2', '3', '4', '5']
            bot.send_audio(message.chat.id, audio=open(f'ben/{random.choice(foo)}.mp3', 'rb'))

        if message.text == 'Stats':
            markup_after_stats = types.InlineKeyboardMarkup(row_width=3)
            item1 = types.InlineKeyboardButton("Yesterday", callback_data='yesterday')
            item2 = types.InlineKeyboardButton("Today", callback_data='today')
            item3 = types.InlineKeyboardButton("All time", callback_data='your_date')
            markup_after_stats.add(item2, item1, item3)
            bot.send_message(message.chat.id, "Choose time", reply_markup=markup_after_stats)

        if is_date(message.text):
            bot.send_message(message.chat.id, message.text)

        if message.text == 'Dialogue':
            users[message.chat.id].chat_state = 2
            bot.send_message(message.chat.id, "Ben", reply_markup=keyboard.markup_after_dialogue)

        if message.text == 'Stop' and users[message.chat.id].chat_state == 2:
            users[message.chat.id].chat_state = 1
            bot.send_message(message.chat.id, "Saved", reply_markup=keyboard.markup_after_start)

        if message.text == 'Stop' and users[message.chat.id].chat_state == 4:
            users[message.chat.id].chat_state = 1
            bot.send_message(message.chat.id, "Good bye", reply_markup=keyboard.markup_after_start)

        if message.text == 'Polls':
            users[message.chat.id].chat_state = 3

        if message.text == 'SuperRelax':
            users[message.chat.id].chat_state = 4
            bot.send_audio(message.chat.id, audio=open(f'ben/6.mp3', 'rb'))


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    try:
        if call.message:

            if call.data == 'today':
                bot.send_message(call.message.chat.id, 'Ваш отчет за сегодня')
                now = datetime.datetime.now()
                emotion_classifier.get_plot(users[call.message.chat.id].results, call.message.chat.id)
                bot.send_photo(call.message.chat.id, photo=open(f'img/{str(call.message.chat.id)}.png', 'rb'))

            if call.data == 'yesterday':
                bot.send_message(call.message.chat.id, "Ваш отчет за вчера")
                now = datetime.datetime.now()
                emotion_classifier.get_plot(users[call.message.chat.id].results, call.message.chat.id)

            if call.data == 'your_date':
                bot.send_message(call.message.chat.id, "Your result")
                emotion_classifier.get_plot(users[call.message.chat.id].results, call.message.chat.id)

        bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, text = "Моя статистика", reply_markup=None)
    except Exception as e:
        print(repr(e))


def is_date(string, fuzzy=False):
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


bot.polling(none_stop=True)
