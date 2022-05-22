from telebot import types


markup_after_start = types.ReplyKeyboardMarkup(resize_keyboard=True)
item1 = types.KeyboardButton("Polls")
item2 = types.KeyboardButton("Stats")
item5 = types.KeyboardButton("Diary")
item8 = types.KeyboardButton("PossibleFriends")
item3 = types.KeyboardButton("Dialogue")
item4 = types.KeyboardButton("SuperRelax")  # проверить что пройден опрос, добавить кнопку рекомендаций
markup_after_start.add(item1, item2, item3, item4, item5, item8)


markup_after_dialogue = types.ReplyKeyboardMarkup(resize_keyboard=True)
item4 = types.KeyboardButton("Stop")
markup_after_dialogue.add(item4)

markup_after_pollkill = types.ReplyKeyboardMarkup(resize_keyboard=True)
item5 = types.KeyboardButton("Start")
item6 = types.KeyboardButton("Stop")
markup_after_pollkill.add(item5, item6)



