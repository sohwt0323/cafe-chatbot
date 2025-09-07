# cafe-chatbot

run this to train model
py scripts/train_lr.py
py scripts/train_nb.py
py scripts/train_svm.py

run this to start the project
py -m app.server

run this for compare algotithm (it will record the true intent and each algorithm predict intent)
py scripts/compare_algorithms.py
