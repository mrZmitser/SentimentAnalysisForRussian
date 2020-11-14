import os

import reviews.rawreview as rr
import pathlib
import pandas as pd
import corpus_reader_tools.readertools as rt
from sklearn.linear_model import LogisticRegression

from corpus_reader_tools.pickledcorpusreader import PickledCorpusReader as PickleCorp
from corpus_reader_tools import vectorizedcorpus


def parse_raw_reviews():
    path = input("get path\n")
    reviews = rr.get_reviews_from_raw_onliner(path)

    directory = pathlib.Path(path).parent.absolute()
    for rv in reviews:
        rr.create_review_file(directory, rv)


def print_corpus_description(corpus):
    print("Корневая папка корпуса : " + corpus.root.path)
    desc = rt.describe_corp(corpus, None, corpus.categories())
    from prettytable import PrettyTable
    table = PrettyTable(['Характерстика', 'Значение'])
    for tag in desc:
        if type(desc[tag]) == int:
            table.add_row([tag, f"{desc[tag]:.{0}f}"])
        else:
            table.add_row([tag, f"{desc[tag]:.{3}f}"])

    print(table)


def get_marks(new_corp):
    return list(open(new_corp.root + '\\marks.txt', 'r').read())


def set_marks(new_corp):
    marks = get_marks(new_corp)
    i = 0
    for fileid in new_corp.fileids():
        i += 1
        if i <= len(marks):
            continue
        print(">Для выхода нажмите 0<")
        print('>Файл', i, 'из', len(new_corp.fileids()), '<')
        print('>Как вы оцениваете данный текст?<')
        words = list(new_corp.words(fileid))
        symbs = 0
        print('--------------------------------------------------------------')
        for word in words:
            print(word, end=" ")
            symbs += len(word)
            if symbs > 80:
                print()
                symbs = 0
        print('\n--------------------------------------------------------------')
        mark = input('>1 - neg, 2 - neutr, 3 - pos<\n')
        if mark == '0':
            break
        marks.append(mark)
    file = open(new_corp.root + '\\marks.txt', 'w')
    for mark in marks:
        file.write(mark)
    file.close()
    return marks


class TextUI:
    delim = "*" * 80
    corp = None
    cur_command = "-"
    vect_corp = None
    log_model = None

    def menu(self):
        while True:
            print(self.delim)
            print('*', '{:*^76}'.format("Анализатор тональности текста на русском языке"), '*')
            print('*', '{: ^76}'.format("© Дмитрий Горбач, 2020"), '*')
            print('*', '{: ^76}'.format("Текущий корпус: " + str(self.corp)), '*')
            print(self.delim)
            print('{:*^80}'.format("Команды"))
            print("\t0.\tВыход")
            print('{:*^80}'.format("Работа с корпусом текста"))
            print("\t1.\tИнформация о корпусе")
            print("\t2.\tОткрыть корпус")
            print("\t3.\tРазметить выбранный корпус")
            print("\t4.\tОткрыть папку в проводнике")
            print("\t5.\tПрочитать readme")
            print("\t6.\tПрочитать citations.bib")
            print("\t7.\tПрочитать лицензию")
            print('{:*^80}'.format("Работа с моделью ИИ"))
            print("\t8.\tВекторизовать корпус")
            print("\t9.\tТренировать модель")
            print("\t10.\tПредсказать тональность документов из корпуса")
            print(self.delim)
            self.cur_command = input(">\t")
            if self.cur_command == '0':
                break
            elif self.cur_command == '1':
                if not (self.corp is None):
                    print_corpus_description(self.corp)
                else:
                    print("Корпус не выбран")
            elif self.cur_command == '2':
                directory = input("Введите путь к корпусу:")
                if not os.path.exists(directory):
                    print("Корпус не найден")
                    continue
                meta = rt.get_meta(directory)
                if meta is not None and meta[0] is not None and meta[0] == "tagged":
                    try:
                        self.corp = rt.read_processed_corpus(directory)
                    except:
                        print("Произошла ошибка")
                        self.corp = None
                        continue
                else:
                    try:
                        self.corp = rt.read_corpus(directory)
                    except:
                        print("Произошла ошибка")
                        self.corp = None
                        continue
                print("Корпус прочитан")
            elif self.cur_command == '3':
                if type(self.corp) == PickleCorp:
                    print("Корпус уже размечен")
                else:
                    directory = input("Введите путь для нового корпуса: ")
                    print("Идёт преобразование...")
                    try:
                        new_files = list(rt.transform(self.corp, directory))
                    except:
                        print("Произошла ошибка")
                        continue
                    print("Количество преобразованных файлов: ", len(new_files))
            elif self.cur_command == '4':
                if not os.path.isdir(self.corp.root):
                    print("Ошибка")
                else:
                    os.system("start " + self.corp.root)
            elif self.cur_command == '5':
                try:
                    print(self.corp.readme())
                except:
                    print("Ошибка")
            elif self.cur_command == '6':
                try:
                    print(self.corp.citation())
                except:
                    print("Ошибка")
            elif self.cur_command == '7':
                try:
                    print(self.corp.license())
                except:
                    print("Ошибка")
            elif self.cur_command == '8':
                if type(self.corp) != PickleCorp:
                    print("Невозможно работать с неразмеченным корпусом")
                else:
                    self.vect_corp = vectorizedcorpus.VectorizedCorpus()
                    with_lemma = input("Введите 1 для использования начальных форм слов, 0 - для исходных") == "1"
                    with_pos = input("Введите 1 для различения частей речей, 0 - для неразличения") == "1"
                    print("Выполняется TF-IDF-векторизация корпуса...")
                    self.vect_corp.vectorize_tagged_corpus(self.corp, with_lemma, with_pos)
            elif self.cur_command == '9':
                if self.vect_corp is None:
                    print("Для начала выполните векторизацию текста")
                elif not os.path.exists(self.corp.root + "\\marks.txt"):
                    print("Нет файла marks.txt для тренировки модели")
                else:
                    print("Файлов в корпусе: ", len(self.corp.fileids()))
                    print("Укажите диапозон для тренировки")
                    beg = int(input("Начало: "))
                    beg -= 1
                    if beg < 0 or beg >= len(self.corp.fileids()):
                        print("Начало диапазона не может быть < 1 и больше количества файлов")
                        continue
                    end = int(input("Конец: "))
                    if end <= beg or end > len(self.corp.fileids()) + 1:
                        print("Конец диапазона не может быть меньше начала и больше количества файлов")
                        continue
                    print("Выполняется тренировка...")
                    self.log_model = LogisticRegression()
                    preds = list(open(self.corp.root + "\\marks.txt").read())
                    self.log_model.fit(X=self.vect_corp.tfidf_matrix[beg:end], y=preds[beg:end])
            elif self.cur_command == "10":
                if self.log_model is None:
                    print("Для начала выполните тренировку модели")
                else:
                    print("Файлов в корпусе: ", len(self.corp.fileids()))
                    beg = int(input("Начало: "))
                    beg -= 1
                    if beg < 0 or beg >= len(self.corp.fileids()):
                        print("Начало диапазона не может быть < 1 и больше количества файлов")
                        continue
                    end = int(input("Конец: "))
                    if end <= beg or end > len(self.corp.fileids()) + 1:
                        print("Конец диапазона не может быть меньше начала и больше количества файлов")
                        continue
                    preds = self.log_model.predict(X=self.vect_corp.tfidf_matrix[beg:end])
                    true_preds = list(open(self.corp.root + "\\marks.txt").read())
                    eqs = 0
                    for i in range(beg, end):
                        print("Файл " + str(i) + " : " + "). Предсказание : " + str(preds[i]))
                        if true_preds[i] == preds[i]:
                            eqs += 1
                    print("Проаналзировано", end - beg, "файлов. Точность:", float(eqs) * 100 / end - beg, "%")
            else:
                print("Неправильная команда")
            print("Нажмите любую клавишу для продолжения...")
            input()


def main():
    print("Loading...")
    ui = TextUI()
    ui.menu()


main()
