from telebot import types


markup_after_start = types.ReplyKeyboardMarkup(resize_keyboard=True)
item1 = types.KeyboardButton("Polls")
item2 = types.KeyboardButton("Stats")
item3 = types.KeyboardButton("Dialogue")
item4 = types.KeyboardButton("SuperRelax")
markup_after_start.add(item1, item2, item3, item4)


markup_after_dialogue = types.ReplyKeyboardMarkup(resize_keyboard=True)
item4 = types.KeyboardButton("Stop")
markup_after_dialogue.add(item4)



