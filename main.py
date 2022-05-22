import random
import numpy as np
import telebot
import config
from User import User
import keyboard
import datetime
from telebot import types
import json
from EmotuionClassifier import Classifier
from TextGenerator import Generator
from Polls import polls

bot = telebot.TeleBot(config.TOKEN)

users = {}
fiction_user = User(0, 'stub')
fiction_user.results = {'01/01/01/01/01/1989': 'I am happy',
                        '01/02/01/01/01/1989': 'The skyes are blue',
                        '01/03/01/01/01/1989': 'The grass is green'
                        }
users[0] = fiction_user

emotion_classifier = Classifier()

text_generator = Generator()

print("I'm alive\n")


def average_user():
    answer = {'sadness': 0,
                            'joy': 0,
                             'love': 0,
                             'anger': 0,
                             'fear': 0,
                             'surprise': 0
    }
    for user in users:
        user_class = users[user]
        average_current = user_class.get_my_average()
        for emotion in answer:
            answer[emotion] += average_current[emotion]
    for emotion in answer:
        answer[emotion] /= len(users)
    return answer


def feelings_dispersions():
    answer = {'sadness': 0,
                            'joy': [],
                             'love': [],
                             'anger': [],
                             'fear': [],
                             'surprise': []
    }
    for user in users:
        user_class = users[user]
        average_current = user_class.get_my_average()
        for emotion in answer:
            answer[emotion].append(average_current[emotion])
    for emotion in answer:
        answer[emotion] = np.std(answer[emotion])
    return answer


@bot.message_handler(commands=['start'])
def welcome(message):
    if message.chat.id in users:
        bot.send_message(message.chat.id, "You are nice <3")
    else:
        users[message.chat.id] = User(message.chat.id, message.from_user.username)
    bot.send_message(message.chat.id, "Hello, I\'m your personal diary. Hope you feel good", reply_markup=keyboard.markup_after_start)


@bot.message_handler(content_types=['text'])
def result(message):
    if message.chat.type == 'private':
        print(users[message.chat.id].chat_state)
        if users[message.chat.id].chat_state == 2 and \
                message.text != 'Polls' and message.text != 'Stats' and message.text != 'Dialogue' and message.text != 'Stop':
            now = datetime.datetime.now()
            string_now = now.strftime('%S/%M/%H/%d/%m/%Y')
            users[message.chat.id].results[string_now] = message.text
            bot.send_message(message.chat.id, text_generator.generator(message.text))

        if users[message.chat.id].chat_state == 5 and \
                message.text != 'Polls' and message.text != 'Stats' and message.text != 'Dialogue' and message.text != 'Stop':
            now = datetime.datetime.now()
            string_now = now.strftime('%S/%M/%H/%d/%m/%Y')
            users[message.chat.id].results[string_now] = message.text

        if message.text == 'Stop' and users[message.chat.id].chat_state == 3:
            users[message.chat.id].chat_state = 1
            bot.send_message(message.chat.id, "OK", reply_markup=keyboard.markup_after_start)

        if message.text != 'Dialogue' and users[message.chat.id].chat_state == 1:
            bot.send_message(message.chat.id, "OK", reply_markup=keyboard.markup_after_start)

        if users[message.chat.id].chat_state == 3:
            state = users[message.chat.id].additional_chat_state
            questions, current_num, _ = state
            current_question = questions[current_num]
            current_answers = list(polls[0][current_question].keys())
            current_answers_vectors = list(polls[0][current_question].values())
            markup_after_poll = types.InlineKeyboardMarkup(row_width=2)
            item1 = types.InlineKeyboardButton(current_answers[0], callback_data='yes')
            item2 = types.InlineKeyboardButton(current_answers[1], callback_data='no')
            markup_after_poll.add(item1, item2)
            users[message.chat.id].additional_chat_state[2] = current_answers_vectors
            bot.send_message(message.chat.id, current_question, reply_markup=markup_after_poll)

        if message.text == 'PossibleFriends':
            bot.send_message(message.chat.id, ', '.join(map(lambda nick: '@' + nick.__repr__(), emotion_classifier.get_closest_users(users, users[message.chat.id]))))

        if users[message.chat.id].chat_state == 4:
            foo = ['1', '2', '3', '4', '5']
            bot.send_audio(message.chat.id, audio=open(f'ben/{random.choice(foo)}.mp3', 'rb'))

        if message.text == 'Stats':
            markup_after_stats = types.InlineKeyboardMarkup(row_width=4)
            item1 = types.InlineKeyboardButton("Recommendation", callback_data='recc')
            item2 = types.InlineKeyboardButton("Report", callback_data='today')
            item4 = types.InlineKeyboardButton('My Temper', callback_data='temp')
            markup_after_stats.add(item2, item1, item4)
            bot.send_message(message.chat.id, "Choose time", reply_markup=markup_after_stats)

        if message.text == 'Dialogue' and users[message.chat.id].chat_state == 1:
            users[message.chat.id].chat_state = 2
            bot.send_message(message.chat.id, "Hi! I am gonna be your buddy", reply_markup=keyboard.markup_after_dialogue)

        if message.text == 'Stop' and users[message.chat.id].chat_state == 2:
            users[message.chat.id].chat_state = 1
            bot.send_message(message.chat.id, "Saved", reply_markup=keyboard.markup_after_start)

        if message.text == 'Stop' and users[message.chat.id].chat_state == 4:
            users[message.chat.id].chat_state = 1
            bot.send_message(message.chat.id, "Good bye", reply_markup=keyboard.markup_after_start)

        if message.text == 'Stop' and users[message.chat.id].chat_state == 5:
            users[message.chat.id].chat_state = 1
            bot.send_message(message.chat.id, "Good bye", reply_markup=keyboard.markup_after_start)

        if message.text == 'Polls':
            users[message.chat.id].chat_state = 3
            bot.send_message(message.chat.id, "Are you ready?", reply_markup=keyboard.markup_after_pollkill)
            users[message.chat.id].additional_chat_state = [[k for k in polls[0]], 0, None]

        if message.text == 'Diary':
            users[message.chat.id].chat_state = 5
            bot.send_message(message.chat.  id, "I'm your diary. You can write anything you want into me", reply_markup=keyboard.markup_after_dialogue)

        if message.text == 'SuperRelax':
            users[message.chat.id].chat_state = 4
            bot.send_message(message.chat.id, '*Phone rings* Beeen', reply_markup=keyboard.markup_after_dialogue)
            bot.send_audio(message.chat.id, audio=open(f'ben/6.mp3', 'rb'))


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    try:
        if call.message:
            if call.data in ['today', 'recc', 'your_date', 'temp']:
                if call.data == 'today':
                    if len(users[call.message.chat.id].results) != 0:
                        bot.send_message(call.message.chat.id, 'Your today report')
                        now = datetime.datetime.now()
                        emotion_classifier.get_plot(users[call.message.chat.id].results, call.message.chat.id)
                        bot.send_photo(call.message.chat.id, photo=open(f'img/{str(call.message.chat.id)}.png', 'rb'))
                    else:
                        bot.send_message(call.message.chat.id, 'Your havent talked to me yet :\'(')

                if call.data == 'recc':
                    if len(users[call.message.chat.id].results) != 0:
                        recs = emotion_classifier.get_reccomendations(users[call.message.chat.id].get_my_average(),
                                                                      average_user(), call.message.chat.id)
                        message = 'You are perfect!'
                        if len(recs) != 0:
                            message = ' '.join(recs)
                        bot.send_message(call.message.chat.id, message)
                        now = datetime.datetime.now()
                        emotion_classifier.get_difference(users[call.message.chat.id].get_my_average(), average_user(), call.message.chat.id)
                        bot.send_photo(call.message.chat.id, photo=open(f'img/av{str(call.message.chat.id)}.png', 'rb'))
                    else:
                        bot.send_message(call.message.chat.id, 'Your havent talked to me yet :\'(')

                if call.data == 'your_date':
                    bot.send_message(call.message.chat.id, "Your result")
                    emotion_classifier.get_plot(users[call.message.chat.id].results, call.message.chat.id)

                if call.data == 'temp':
                    if any(users[call.message.chat.id].temper.tolist()):
                        bot.send_message(call.message.chat.id, "Your result")
                        emotion_classifier.get_temper(users[call.message.chat.id].temper, call.message.chat.id)
                        bot.send_photo(call.message.chat.id, photo=open(f'img/{str(call.message.chat.id)}.png', 'rb'))

                        bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                      text="My stats",
                                      reply_markup=None)
                    else:
                        bot.send_message(call.message.chat.id, "You need to pass the survey first ;-]")
            else:
                if call.data == 'yes':
                    vector = users[call.message.chat.id].additional_chat_state[2][0]
                    users[call.message.chat.id].temper += np.array(vector)
                    users[call.message.chat.id].additional_chat_state[1] += 1
                    if users[call.message.chat.id].additional_chat_state[1] == len(users[call.message.chat.id].additional_chat_state[0]):
                        users[call.message.chat.id].chat_state = 1
                    result(call.message)
                if call.data == 'no':
                    vector = users[call.message.chat.id].additional_chat_state[2][1]
                    users[call.message.chat.id].temper += np.array(vector)
                    users[call.message.chat.id].additional_chat_state[1] += 1
                    if users[call.message.chat.id].additional_chat_state[1] == len(users[call.message.chat.id].additional_chat_state[0]):
                        users[call.message.chat.id].chat_state = 1
                    result(call.message)
                bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id,
                                      text="Answer is accepted",
                                      reply_markup=None)

    except Exception as e:
        print(repr(e))


bot.polling(none_stop=True)
