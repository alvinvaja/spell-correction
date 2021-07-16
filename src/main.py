import sys
import numpy as np
import pandas as pd
import random
import requests
import re
import math
import PyQt5.QtWidgets as qtw
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QDialog
        
class Utils:
    def __init__(self, bigramFileName, unigramFileName):
        self.bigramFileName = bigramFileName
        self.unigramFileName = unigramFileName
        
        self.loadResources()
    
    def loadResources(self):
        self.bigram = self.loadBigram(self.bigramFileName)
        self.unigram = self.loadUnigram(self.unigramFileName)
        
    def loadBigram(self, fileName):
        df = pd.read_csv(fileName, sep='#', header=None);

        freq = {}
        for key, value in df.iterrows():
            freq[tuple([value[0], value[1]])] = value[2]

        return freq
    
    def loadUnigram(self, fileName):
        df = pd.read_csv(fileName, sep='#', header=None);

        freq = {}
        for key, value in df.iterrows():
            freq[value[0]] = value[1]

        return freq
    
    def getLevDist(self, string1, string2):
        if len(string1) < len(string2):
            return self.getLevDist(string2, string1)
        if len(string2) == 0:
            return len(string1)

        arr1 = np.array(list(string1))
        arr2 = np.array(list(string2))

        last_row = np.arange(arr2.size + 1)
        for s in arr1:
            #insertion
            current_row = last_row + 1

            #substitution
            current_row[1:] = np.minimum(current_row[1:], np.add(last_row[:-1], arr2 != s))

            #deletion
            current_row[1:] = np.minimum(current_row[1:], current_row[:-1] + 1)

            #update rows
            last_row = current_row

        return last_row[-1]
    
    def isEntitasWord(self, word):
        if len(word) <= 0:
            return True
        for i in range(0, len(word)):
            if (word[i] >= 'a' and word[i] <= 'z') or word[i] == '-':
                cnt = 1
            else:
                return True
        
        return False
    
    def isRepeatedWord(self, word):
        dashCount = 0
        for c in word:
            dashCount += c == '-'

        return dashCount == 1

    def separateRepeatedWord(self, word):
        index = 0
        for i in range(len(word)):
            if word[i] == '-':
                index = i
                break

        return [word[:index], word[index+1:]]

    def isAlphaNum(self, c):
        return c >= 'a' and c <= 'z' or c >= 'A' and c <= 'Z' or c >= '0' and c <= '9'

    def getPrefixNonAlphaNum(self, word):
        pref = ""
        for c in word:
            if self.isAlphaNum(c):
                break;
            pref += c

        return pref

    def getSuffixNonAlphaNum(self, word):
        suff = ""
        word = word[::-1]
        for c in word:
            if self.isAlphaNum(c):
                break;
            suff += c

        return suff[::-1]
    
    def extractWord(self, word):
        index = 0
        for c in word:
            if self.isAlphaNum(c):
                break
            index += 1

        word = word[index:]

        index = len(word)
        for i in range(len(word) - 1, -1, -1):
            c = word[i]
            if self.isAlphaNum(c):
                break
            index -= 1

        return word[:index]
    

class SpellCheck:
    def __init__(self, bigramFileName, unigramFileName):
        self.typoText = ''
        self.originalText = ''
        self.solutionText = ''
        
        self.utils = Utils(bigramFileName, unigramFileName)
        self.setBigram(self.utils.bigram)
        self.setUnigram(self.utils.unigram)
        
    def setBigram(self, bigram):
        self.bigram = bigram
    
    def setUnigram(self, unigram):
        self.unigram = unigram                
        self.corpus = list(self.unigram.keys())
        self.generateDeletes(self.corpus)
        
    def setOriginalText(self, text):
        self.originalText = text
                        
    def generateDeletes(self, unigram, dist = 1):
        self.deletes = {}
        queue = []
        
        words = unigram

        for word in words:
            queue.append([word, word])

        while dist > 0:
            dist -= 1

            next_queue = []
            visitedWord = {}

            for [word, parent] in queue:
                if type(word) == str and len(word) > 1:
                    for c in range(len(word)):
                        word_without_c = word[:c] + word[c+1:]

                        if word_without_c in self.deletes:
                            if parent not in self.deletes[word_without_c]:
                                self.deletes[word_without_c].append(parent)
                        else:
                            self.deletes[word_without_c] = [parent]

                        if word_without_c not in visitedWord:
                            visitedWord[word_without_c] = 1
                            next_queue.append([word_without_c, parent])

            queue = next_queue
                        
    def generateSuggestions(self, word, depth = 2, limit = 15):
        suggestions = []
        queue = [word]

        if word in self.deletes:
            for candidate in self.deletes[word]:
                if candidate not in suggestions:
                    suggestions.append(candidate)

        while depth > 0:
            depth -= 1

            next_queue = []

            for w in queue:
                if type(word) == str and len(w) > 1:
                    
                    if w in self.unigram and w not in suggestions:
                        suggestions.append(w)
                    
                    for c in range(len(w)):
                        w_without_c = w[:c] + w[c+1:]

                        if w_without_c in self.deletes:
                            for candidate in self.deletes[w_without_c]:
                                if candidate not in suggestions:
                                    suggestions.append(candidate)

                        if w_without_c not in next_queue:
                            next_queue.append(w_without_c)

            queue = next_queue

        def sortByFrequencies(suggest_word):
            return self.unigram[suggest_word]
            #coba hapus editDistance

        suggestions.sort(key=sortByFrequencies, reverse=True)

        return suggestions[:limit]
    
    def solve(self, text, depth = 2, limit = 15):
        stop_symbol = []
        stop_symbol_counter = 0
        for i in range(len(text)):
            c = text[i]
            if i + 1 < len(text):
                next_c = text[i + 1]
            else:
                next_c = ''

            if c in ['.', '!', '?'] and next_c == ' ':
                stop_symbol.append(c)

        new_text = []

        sentences = re.split('\. |\! |\? ', text)
        for sentence in sentences:
            sentence = sentence.strip()

            if len(sentence) > 0:
                new_sentence = self.solveSentence(sentence, depth, limit)
                if stop_symbol_counter < len(stop_symbol):
                    new_sentence += stop_symbol[stop_symbol_counter]
                    stop_symbol_counter += 1
                new_text.append(new_sentence)

        self.solutionText = ' '.join(new_text).strip()
    
    def solveSentence(self, sentence, depth, limit):

        repeated_index = {}
        suggestion_arr = []
        prefix = []
        suffix = []
        sentences = []

        words = sentence.split()

        i = 0

        for word in words:
            word = word.strip()
            word_suggestions = []
            
            if len(word) <= 0:
                continue

            if self.utils.isRepeatedWord(word):
                [word1, word2] = self.utils.separateRepeatedWord(word)

                prefix.append(self.utils.getPrefixNonAlphaNum(word1))
                suffix.append(self.utils.getSuffixNonAlphaNum(word1))
                word1 = self.utils.extractWord(word1)
                
                origWord1 = word1
                word1 = word1.lower()
                
                if self.utils.isEntitasWord(origWord1):
                    word_suggestions = [origWord1]
                elif word1 not in self.corpus:
                    word_suggestions = self.generateSuggestions(word1, depth, limit)
                else:
                    word_suggestions = [word1]

                if len(word_suggestions) == 0:
                    word_suggestions = [word1]

                suggestion_arr.append(word_suggestions)
                sentences.append(word1)

                word_suggestions = []

                prefix.append(self.utils.getPrefixNonAlphaNum(word2))
                suffix.append(self.utils.getSuffixNonAlphaNum(word2))
                word2 = self.utils.extractWord(word2)
                
                origWord2 = word2
                word2 = word2.lower()

                if self.utils.isEntitasWord(origWord2):
                    word_suggestions = [origWord2]
                elif word2 not in self.corpus:
                    word_suggestions = self.generateSuggestions(word2, depth, limit)
                else:
                    word_suggestions = [word2]

                if len(word_suggestions) == 0:
                    word_suggestions = [word2]

                suggestion_arr.append(word_suggestions)
                sentences.append(word2)

                repeated_index[i] = 1

                i += 2

            else:

                prefix.append(self.utils.getPrefixNonAlphaNum(word))
                suffix.append(self.utils.getSuffixNonAlphaNum(word))
                word = self.utils.extractWord(word)
                
                origWord = word
                word = word.lower()

                if self.utils.isEntitasWord(origWord):
                    word_suggestions = [origWord]
                elif word not in self.corpus:
                    word_suggestions = self.generateSuggestions(word, depth, limit)
                else:
                    word_suggestions = [word]

                if len(word_suggestions) == 0:
                    word_suggestions = [word]

                suggestion_arr.append(word_suggestions)
                sentences.append(word)

                i += 1

        #prob(word2 | word1) = count(word1, word2) / count(word2)
        #if bigram not exists, return -inf
        def getPriorProbability(firstWord, secondWord):
            freq_first_word = self.unigram[firstWord] if firstWord in self.unigram else 1
            freq_second_word = self.unigram[secondWord] if secondWord in self.unigram else 1

            tup = tuple([firstWord, secondWord])
            return math.log(self.bigram[tup] / freq_first_word) if tup in self.bigram else -1e6


        #dp starts here
        #dp state
        #pos = current word position
        #num = state taking num-th suggestion for current word
        
        n = len(suggestion_arr)

        dp = {}

        def calc(pos, num):
            key = tuple([pos, num])

            if pos >= n - 1:
                dp[key] = 0
                return 0

            if pos >= 0 and key in dp:
                return dp[key]

            best = -1e10
            for i in range(len(suggestion_arr[pos + 1])):
                if pos == -1:
                    best = max(best, calc(pos + 1, i))
                else:
                    best = max(best, getPriorProbability(suggestion_arr[pos][num], suggestion_arr[pos + 1][i]) + calc(pos + 1, i))

            if pos >= 0:
                dp[key] = best

            return best

        totalBayes = calc(-1, 0)

        #reconstruct sentence
        new_sentence = []

        pointer = 0
        best = -1e10
        for i in range(len(suggestion_arr[0])):

            key = tuple([0, i])
            if dp[key] > best:
                best = dp[key]
                pointer = i

        new_sentence.append(suggestion_arr[0][pointer])

        sum = 0
        for i in range(1, n):
            maxValue = -1e10
            indexTaken = 0

            for j in range(len(suggestion_arr[i])):
                key = tuple([i, j])
                bigramValue = getPriorProbability(new_sentence[-1], suggestion_arr[i][j])

                if sum + dp[key] + bigramValue > maxValue:
                    maxValue = sum + dp[key] + bigramValue
                    indexTaken = j

            bigramValue = getPriorProbability(new_sentence[-1], suggestion_arr[i][indexTaken])
            sum += bigramValue
            new_sentence.append(suggestion_arr[i][indexTaken])

        sentences = []

        i = 0
        while i < len(new_sentence):
            if i in repeated_index:
                word = prefix[i] + new_sentence[i] + suffix[i] + '-' + prefix[i + 1] + new_sentence[i + 1] + suffix[i + 1]
                i += 1
            else:
                word = prefix[i] + new_sentence[i] + suffix[i]

            sentences.append(word)
            i += 1

        return ' '.join(sentences)

    
class WorkerThread(QThread):

    update_resource = pyqtSignal(SpellCheck)
    
    def run(self):
        spellCheck = SpellCheck('BigramFinal.csv', 'UnigramFinal.csv')
        self.update_resource.emit(spellCheck)
    
    
class Main(qtw.QWidget):
    def __init__(self):
        super().__init__()
        
        self.setWindowIcon(QtGui.QIcon('spellcheck.png'))
        self.setWindowTitle('Spell Checker')
        
        self.initLoadingUI()
        self.loadInitialResources()
    
    def loadInitialResources(self):
        self.worker = WorkerThread()
        self.worker.start()
        self.worker.update_resource.connect(self.updateResource)
        self.worker.finished.connect(self.postResourceLoad)
        
    def updateResource(self, val):
        self.spellCheck = val
        
    def postResourceLoad(self):
        self.movie.stop()
        self.movieLabel.hide()
        self.loadingLabel.hide()
        self.close()
        self.initMainUI()
        self.show()
    
    def initLoadingUI(self):
        self.setGeometry(100, 100, 320, 320)
        self.movie = QtGui.QMovie('spinner_transparent.gif')
        self.movieLabel = qtw.QLabel(self)
        self.movieLabel.setMovie(self.movie)
        self.movieLabel.resize(256, 256)
        self.movieLabel.move(32, 10)
        
        loadingText = 'Loading Bigram and Unigram...'

        self.loadingLabel = qtw.QLabel(self)
        self.loadingLabel.setText(loadingText)
        self.loadingLabel.setFont(QtGui.QFont('Arial', 12))
        self.loadingLabel.resize(320, 40)
        self.loadingLabel.move(30, 265)
                            
        self.show()
        self.movie.start()
    
    def initMainUI(self):
        self.setWindowIcon(QtGui.QIcon('spellcheck.png'))
        self.setWindowTitle('Spell Checker')
        self.setGeometry(100, 100, 1500, 750)
        
        
        self.initButtons()
        self.initDropDowns()
        self.initLabels()
        self.initTextBoxes()
        self.initTitles()
        self.initRadioButtons()
        self.initStats()
        
    def initTextBoxes(self):
        #spell check boxes
        self.typoBox = qtw.QPlainTextEdit(self)
        self.typoBox.move(50, 100)
        self.typoBox.resize(300, 100)

        self.solutionTextBox = qtw.QPlainTextEdit(self)
        self.solutionTextBox.setReadOnly(True)
        self.solutionTextBox.move(500, 100)
        self.solutionTextBox.resize(300, 100)
        
        self.correctedTextBox = qtw.QPlainTextEdit(self)
        self.correctedTextBox.setReadOnly(True)
        self.correctedTextBox.move(50, 250)
        self.correctedTextBox.resize(300, 80)        
        
        self.feedbackBox = qtw.QPlainTextEdit(self)
        self.feedbackBox.setReadOnly(True)
        self.feedbackBox.move(950, 275)
        self.feedbackBox.resize(100, 100)
        
        #generate typo text boxes
        self.originalBox = qtw.QPlainTextEdit(self)
        self.originalBox.move(50, 425)
        self.originalBox.resize(300, 100)
        
        self.typoBox2 = qtw.QPlainTextEdit(self)
        self.typoBox2.setReadOnly(True)
        self.typoBox2.move(500, 425)
        self.typoBox2.resize(300, 100)

        self.solutionTextBox2 = qtw.QPlainTextEdit(self)
        self.solutionTextBox2.setReadOnly(True)
        self.solutionTextBox2.move(950, 425)
        self.solutionTextBox2.resize(300, 100)
        
    def initButtons(self):
        self.solveButton = qtw.QPushButton('Solve', self, clicked = self.generateSolution)
        self.solveButton.move(385, 130)
        
        self.generateTypoButton = qtw.QPushButton('Generate Typo', self, clicked = self.generateTypo)
        self.generateTypoButton.move(385, 455)
        
        self.solveButton2 = qtw.QPushButton('Solve', self, clicked = self.generateSolution2)
        self.solveButton2.move(835, 455)
        
        self.generateFeedbackButton = qtw.QPushButton('Get Words', self, clicked = self.generateFeedback)
        self.generateFeedbackButton.move(1075, 295)
        self.generateFeedbackButton.resize(100, 28)
        
        self.updateFeedbackButton = qtw.QPushButton('Update Corpus', self, clicked = self.updateBigramUnigram)
        self.updateFeedbackButton.move(1075, 335)
        self.updateFeedbackButton.resize(100, 28)
        
        self.diffButton1 = qtw.QPushButton('View Differences', self)
        self.diffButton1.clicked.connect(lambda: self.viewDifferences(self.typoBox.toPlainText(), self.solutionTextBox.toPlainText()))
        self.diffButton1.resize(125, 28)
        self.diffButton1.move(385, 280)
        
        self.diffButton2 = qtw.QPushButton('View Differences', self)
        self.diffButton2.clicked.connect(lambda: self.viewDifferences(self.typoBox2.toPlainText(), self.solutionTextBox2.toPlainText()))
        self.diffButton2.resize(125, 28)
        self.diffButton2.move(1285, 630)
        
    def initDropDowns(self):
        #depth option
        self.depth = 2
        self.depthDropDown = qtw.QComboBox(self)
        
        for i in range(1, 4):
            self.depthDropDown.addItem(str(i))
            
        self.depthDropDown.resize(80, 28)
        self.depthDropDown.move(1075, 75)
        
        self.depthLabel = qtw.QLabel(self)
        self.depthLabel.setText('Depth Search')
        self.depthLabel.resize(125, 28)
        self.depthLabel.move(950, 75)
        
        #limit option
        self.limit = 15
        self.limitDropDown = qtw.QComboBox(self)
        
        for i in range(5, 26, 5):
            self.limitDropDown.addItem(str(i))
            
        self.limitDropDown.resize(80, 28)
        self.limitDropDown.move(1075, 125)
        
        self.limitLabel = qtw.QLabel(self)
        self.limitLabel.setText('Suggestion Limit')
        self.limitLabel.resize(125, 28)
        self.limitLabel.move(950, 125)
        
        #typo words option
        self.typoWordLabel = qtw.QLabel(self)
        self.typoWordLabel.setText('Number of Typo Words')
        self.typoWordLabel.resize(150, 28)
        self.typoWordLabel.move(1200, 75)
        
        self.typoWordBox = qtw.QPlainTextEdit(self)
        self.typoWordBox.setPlainText('15')
        self.typoWordBox.resize(80, 28)
        self.typoWordBox.move(1350, 75) 
        
        #typo letters option
        self.typoLetterLabel = qtw.QLabel(self)
        self.typoLetterLabel.setText('Number of Typo Letters')
        self.typoLetterLabel.resize(150, 28)
        self.typoLetterLabel.move(1200, 125)
        
        self.typoLetterBox = qtw.QPlainTextEdit(self)
        self.typoLetterBox.setPlainText('1')
        self.typoLetterBox.resize(80, 28)
        self.typoLetterBox.move(1350, 125)
        
        #set default option
        self.depthDropDown.setCurrentIndex(0)
        self.limitDropDown.setCurrentIndex(2)
        
    def initLabels(self):
        #spell check labels
        self.typoLabel = qtw.QLabel(self)
        self.typoLabel.setText('Typo Text')
        self.typoLabel.resize(100, 28)
        self.typoLabel.move(50, 75)
        
        self.solutionLabel = qtw.QLabel(self)
        self.solutionLabel.setText('Solution Text')
        self.solutionLabel.resize(100, 28)
        self.solutionLabel.move(500, 75)
        
        self.correctedLabel = qtw.QLabel(self)
        self.correctedLabel.setText('Corrected Words')
        self.correctedLabel.resize(100, 28)
        self.correctedLabel.move(50, 225)
        
        self.feedbackLabel = qtw.QLabel(self)
        self.feedbackLabel.setText('Add mistakenly corrected words to current bigram and unigram')
        self.feedbackLabel.resize(350, 28)
        self.feedbackLabel.move(950, 250)
        
        #generate typo labels
        self.originalLabel = qtw.QLabel(self)
        self.originalLabel.setText('Original Text')
        self.originalLabel.resize(100, 28)
        self.originalLabel.move(50, 400)
        
        self.typoLabel2 = qtw.QLabel(self)
        self.typoLabel2.setText('Typo Text')
        self.typoLabel2.resize(100, 28)
        self.typoLabel2.move(500, 400)
        
        self.solutionLabel2 = qtw.QLabel(self)
        self.solutionLabel2.setText('Solution Text')
        self.solutionLabel2.resize(100, 28)
        self.solutionLabel2.move(950, 400)
        
    def initTitles(self):
        titleFont = QtGui.QFont('Arial', 12)
        titleFont.setBold(True)
        
        self.title1 = qtw.QLabel(self)
        self.title1.setText('Spell Checker and Correction - Bahasa Indonesia')
        self.title1.setFont(titleFont)
        self.title1.move(50, 40)
        
        self.title2 = qtw.QLabel(self)
        self.title2.setText('Typo Generator')
        self.title2.setFont(titleFont)
        self.title2.move(50, 365)
        
        self.titleOption = qtw.QLabel(self)
        self.titleOption.setText('Parameters')
        self.titleOption.setFont(titleFont)
        self.titleOption.move(950, 40)
        
        self.titleFeedback = qtw.QLabel(self)
        self.titleFeedback.setText('Update Bigram and Unigram')
        self.titleFeedback.setFont(titleFont)
        self.titleFeedback.move(950, 225)
        
    def initRadioButtons(self):
        self.typoTypeLabel = qtw.QLabel(self)
        self.typoTypeLabel.setText('Typo Type')
        self.typoTypeLabel.resize(100, 28)
        self.typoTypeLabel.move(1300, 400)
        
        self.deletionRadio = qtw.QCheckBox(self)
        self.deletionRadio.setText('Deletion')
        self.deletionRadio.resize(100, 28)
        self.deletionRadio.move(1300, 430)
        
        self.substitutionRadio = qtw.QCheckBox(self)
        self.substitutionRadio.setText('Substitution')
        self.substitutionRadio.resize(100, 28)
        self.substitutionRadio.move(1300, 460)
        
        self.transpositionRadio = qtw.QCheckBox(self)
        self.transpositionRadio.setText('Transposition')
        self.transpositionRadio.resize(100, 28)
        self.transpositionRadio.move(1300, 490)
        
        self.deletionRadio.setChecked(True)
        
    def initStats(self):
        titleFont = QtGui.QFont('Arial', 12)
        titleFont.setBold(True)
    
        self.statsTitle = qtw.QLabel(self)
        self.statsTitle.setText('Stats')
        self.statsTitle.setFont(titleFont)
        self.statsTitle.move(50, 565)
        
        #correction stats
        self.correctionStats = qtw.QLabel(self)
        self.correctionStats.setText('Correction Stats')
        self.correctionStats.resize(100, 28)
        self.correctionStats.move(50, 590)
        
        self.correctLabel = qtw.QLabel(self)
        self.correctLabel.setText('Correct')
        self.correctLabel.resize(100, 28)
        self.correctLabel.move(50, 620)
        
        self.totalLabel = qtw.QLabel(self)
        self.totalLabel.setText('Total')
        self.totalLabel.resize(100, 28)
        self.totalLabel.move(50, 640)
        
        self.accuracyLabel = qtw.QLabel(self)
        self.accuracyLabel.setText('Accuracy')
        self.accuracyLabel.resize(100, 28)
        self.accuracyLabel.move(50, 660)
        
        self.correctValue = qtw.QLabel(self)
        self.correctValue.resize(100, 28)
        self.correctValue.move(125, 620)
        
        self.totalValue = qtw.QLabel(self)
        self.totalValue.resize(100, 28)
        self.totalValue.move(125, 640)
        
        self.accuracyValue = qtw.QLabel(self)
        self.accuracyValue.resize(100, 28)
        self.accuracyValue.move(125, 660)
        
        #detection stats
        self.detectionStats = qtw.QLabel(self)
        self.detectionStats.setText('Detection Stats')
        self.detectionStats.resize(100, 28)
        self.detectionStats.move(200, 590)
        
        self.correctLabel2 = qtw.QLabel(self)
        self.correctLabel2.setText('Correct')
        self.correctLabel2.resize(100, 28)
        self.correctLabel2.move(200, 620)
        
        self.totalLabel2 = qtw.QLabel(self)
        self.totalLabel2.setText('Total')
        self.totalLabel2.resize(100, 28)
        self.totalLabel2.move(200, 640)
        
        self.accuracyLabel2 = qtw.QLabel(self)
        self.accuracyLabel2.setText('Accuracy')
        self.accuracyLabel2.resize(100, 28)
        self.accuracyLabel2.move(200, 660)
        
        self.correctValue2 = qtw.QLabel(self)
        self.correctValue2.resize(100, 28)
        self.correctValue2.move(275, 620)
        
        self.totalValue2 = qtw.QLabel(self)
        self.totalValue2.resize(100, 28)
        self.totalValue2.move(275, 640)
        
        self.accuracyValue2 = qtw.QLabel(self)
        self.accuracyValue2.resize(100, 28)
        self.accuracyValue2.move(275, 660)
        
        self.correctBoxLabel = qtw.QLabel(self)
        self.correctBoxLabel.setText('Correct Replacements')
        self.correctBoxLabel.resize(125, 28)
        self.correctBoxLabel.move(400, 575)
        
        self.correctBox = qtw.QPlainTextEdit(self)
        self.correctBox.resize(400, 90)
        self.correctBox.move(400, 600)
        
        self.wrongBoxLabel = qtw.QLabel(self)
        self.wrongBoxLabel.setText('Wrong Replacements')
        self.wrongBoxLabel.resize(125, 28)
        self.wrongBoxLabel.move(850, 575)
        
        self.wrongBox = qtw.QPlainTextEdit(self)
        self.wrongBox.resize(400, 90)
        self.wrongBox.move(850, 600)
        
    def viewDifferences(self, typoText, solText):
        typoArr = typoText.split()
        solArr = solText.split()
        
        self.diffDialog = QDialog(self)
        self.diffDialog.setWindowIcon(QtGui.QIcon('spellcheck.png'))
        self.diffDialog.setWindowTitle('View Differences')
        self.diffDialog.setGeometry(100, 100, 950, 400)
        
        self.typoDiffBox = qtw.QTextEdit(self.diffDialog)
        self.typoDiffBox.resize(400, 150)
        self.typoDiffBox.move(50, 50)
        
        self.solDiffBox = qtw.QTextEdit(self.diffDialog)
        self.solDiffBox.resize(400, 150)
        self.solDiffBox.move(500, 50)
        
        self.closeButton = qtw.QPushButton('Close', self.diffDialog, clicked = self.closeDialog)
        self.closeButton.resize(50, 30)
        self.closeButton.move(450, 320)
        
        typo = ''
        sol = ''
        
        maxLen = max(len(typoArr), len(solArr))
        
        for i in range(maxLen):
            if i >= len(typoArr):
                typoWord = ''
            else:
                typoWord = typoArr[i]
            
            if i >= len(solArr):
                solWord = ''
            else:
                solWord = solArr[i]
            
            prefixTypo = self.spellCheck.utils.getPrefixNonAlphaNum(typoWord)
            suffixTypo = self.spellCheck.utils.getSuffixNonAlphaNum(typoWord)
            typoWord = self.spellCheck.utils.extractWord(typoWord)
            
            prefixSol = self.spellCheck.utils.getPrefixNonAlphaNum(solWord)
            suffixSol = self.spellCheck.utils.getSuffixNonAlphaNum(solWord)
            solWord = self.spellCheck.utils.extractWord(solWord)
            
            typo += prefixTypo
            sol += prefixSol
            
            if typoWord.lower() != solWord.lower():
                typo += '<span style=\"background: #FF8983;\">' + typoWord + '</span>'
                sol += '<span style=\"background: #6BDFB8;\">'+ solWord + '</span>'
            else:
                typo += typoWord
                sol += solWord
            
            typo += suffixTypo
            sol += suffixSol
            
            typo += ' '
            sol += ' '
        
        self.typoDiffBox.setText(typo.strip())
        self.solDiffBox.setText(sol.strip())
        
        self.diffDialog.show()
        
    def closeDialog(self):
        self.diffDialog.close()
        
    def generateSolution(self):
        inputText = self.typoBox.toPlainText()
        
        depth = self.depthDropDown.currentText()
        limit = self.limitDropDown.currentText()
        
        depth = int(depth)
        limit = int(limit)
        
        self.spellCheck.solve(inputText, depth, limit)        
        solutionText = self.spellCheck.solutionText
        
        input_arr = inputText.split()
        solution_arr = solutionText.split()
        
        corrected_words = ''
        
        for i in range(len(input_arr)):
            input_word = input_arr[i].strip()
            solution_word = solution_arr[i].strip()
            
            input_word = input_word.lower()
            solution_word = solution_word.lower()
            
            if len(input_word) > 0:
                if input_word != solution_word:
                    corrected_words += input_word + ' ==> ' + solution_word + '\n' 
        
        self.solutionTextBox.setPlainText(solutionText)
        self.correctedTextBox.setPlainText(corrected_words)
        
    def generateSolution2(self):
        typoText = self.typoBox2.toPlainText()
        
        depth = self.depthDropDown.currentText()
        limit = self.limitDropDown.currentText()
        
        depth = int(depth)
        limit = int(limit)
        
        self.spellCheck.solve(typoText, depth, limit)        
        
        origText = self.originalBox.toPlainText()
        origText = origText.lower()
        origText = origText.strip()

        solutionText = self.spellCheck.solutionText

        orig_arr = origText.split()
        typo_arr = typoText.split()
        solution_arr = solutionText.split()
        
        correct_words = ''
        wrong_words = ''
        correct = 0
        total = 0
        correctDetect = 0
        totalDetect = 0
        
        for i in range(len(orig_arr)):
            orig_word = orig_arr[i].strip()
            typo_word = typo_arr[i].strip()
            solution_word = solution_arr[i].strip()
            
            orig_word = orig_word.lower()
            typo_word = typo_word.lower()
            solution_word = solution_word.lower()
            
            #correction
            if len(orig_word) > 0 and orig_word != typo_word:
                if solution_word == orig_word:
                    correct += 1
                    correct_words += typo_word + ' ==> ' + solution_word + '\n'
                else:
                    wrong_words += typo_word + ' ==> ' + solution_word + ', Expected: ' + orig_word + '\n'
    
                total += 1
        
            #detection
            if len(orig_word) > 0:
                totalDetect += 1
                if orig_word != typo_word and typo_word == solution_word or orig_word == typo_word and typo_word != solution_word:
                    continue
                correctDetect += 1
                
        
        self.solutionTextBox2.setPlainText(solutionText)
        self.correctBox.setPlainText(correct_words)
        self.wrongBox.setPlainText(wrong_words)
        
        accuracy = round(100.0 * (correct / total if total > 0 else 1.0), 2)
        self.correctValue.setText(str(correct))
        self.totalValue.setText(str(total))
        self.accuracyValue.setText(str(accuracy) + '%')
        
        accuracyDetect = round(100.0 * (correctDetect / totalDetect if totalDetect > 0 else 1.0), 2)
        self.correctValue2.setText(str(correctDetect))
        self.totalValue2.setText(str(totalDetect))
        self.accuracyValue2.setText(str(accuracyDetect) + '%')
        
    def generateTypo(self):
        orig_text = self.originalBox.toPlainText()
        orig_text = orig_text.strip()
        words = orig_text.split()

        number_typo_word = self.typoWordBox.toPlainText()
        number_typo_word = re.sub(r'[^0-9]+', '', number_typo_word)
        if len(number_typo_word) > 0:
            number_typo_word = int(number_typo_word)
        else:
            number_typo_word = 15 #default
        
        number_typo_word = min(number_typo_word, len(words))
        number_typo_word = max(number_typo_word, 1)
        
        max_typo_letter_in_word = self.typoLetterBox.toPlainText()
        max_typo_letter_in_word = re.sub(r'[^0-9]+', '', max_typo_letter_in_word)
        if len(max_typo_letter_in_word) > 0:
            max_typo_letter_in_word = int(max_typo_letter_in_word)
        else:
            max_typo_letter_in_word = 1 #default
            
        max_typo_letter_in_word = max(max_typo_letter_in_word, 1)

        typo_index = np.arange(len(words))
        random.shuffle(typo_index)
        typo_index = typo_index[:number_typo_word]
        
        typo_type = []
        if self.deletionRadio.isChecked():
            typo_type.append(1)
        if self.substitutionRadio.isChecked():
            typo_type.append(2)
        if self.transpositionRadio.isChecked():
            typo_type.append(3)

        for index in typo_index:
            word = words[index].strip()
            
            if len(word) < 3 or self.spellCheck.utils.isEntitasWord(word) or len(typo_type) < 1:
                continue
            
            word = word.lower()

            typo_letter_index = np.arange(len(word))
            random.shuffle(typo_letter_index)
            typo_letter_count = random.randrange(1, min(len(word), max_typo_letter_in_word) + 1)
            typo_letter_index = typo_letter_index[:typo_letter_count]
            
            typo_index = random.randrange(0, len(typo_type))
            tipe = typo_type[typo_index]

            new_word = ''
    
            if tipe == 1: #deletion typo
                for i in range(len(word)):
                    if i in typo_letter_index and word[i] >= 'a' and word[i] <= 'z':
                        continue
                    new_word += word[i]
                
            elif tipe == 2: #substitution typo
                for i in range(len(word)):
                    if i in typo_letter_index and word[i] >= 'a' and word[i] <= 'z':
                        randomLowerLetter = chr(random.randint(ord('a'), ord('z')))
                        while word[i] == randomLowerLetter:
                            randomLowerLetter = chr(random.randint(ord('a'), ord('z')))
                        new_word += randomLowerLetter
                    else:
                        new_word += word[i]
                
            elif tipe == 3: #transposition typo
                i = 0
                while i < len(word) - 1:
                    if i in typo_letter_index and word[i] >= 'a' and word[i] <= 'z' and word[i + 1] >= 'a' and word[i + 1] <= 'z':
                        new_word += word[i + 1]
                        new_word += word[i]
                        i += 2
                    else:
                        new_word += word[i]
                        i += 1
                
                if len(new_word) < len(word):
                    new_word += word[-1:]

            words[index] = new_word

        self.typoBox2.setPlainText(' '.join(words))
        
    def generateFeedback(self):
        orig_text = self.originalBox.toPlainText()
        orig_text = orig_text.lower()
        orig_text = orig_text.strip()
        
        typo_text = self.typoBox2.toPlainText()
        typo_text = typo_text.lower()
        typo_text = typo_text.strip()
        
        sol_text = self.solutionTextBox2.toPlainText()
        sol_text = sol_text.lower()
        sol_text = sol_text.strip()
        
        orig_arr = orig_text.split()
        typo_arr = typo_text.split()
        sol_arr = sol_text.split()
        
        visitedWords = {}
        feedbackWords = ''
        for i in range(len(orig_arr)):
            orig_word = self.spellCheck.utils.extractWord(orig_arr[i].strip())
            typo_word = self.spellCheck.utils.extractWord(typo_arr[i].strip())
            sol_word = self.spellCheck.utils.extractWord(sol_arr[i].strip())
            
            if (orig_word == typo_word and orig_word != sol_word) or (orig_word != typo_word and typo_word == sol_word):
                if orig_word in self.spellCheck.unigram:
                    continue
                if orig_word not in visitedWords:
                    feedbackWords += orig_word + '\n'
                    visitedWords[orig_word] = 1

        self.feedbackBox.setPlainText(feedbackWords)
        
    def updateBigramUnigram(self):
        feedbackWords = self.feedbackBox.toPlainText()
        feedbackWords = feedbackWords.strip()
        
        words = feedbackWords.split()
        feedback = {}
        
        #update unigram
        for word in words:
            word = word.strip()
            feedback[word] = 1

            if word in self.spellCheck.unigram:
                self.spellCheck.unigram[word] += 1
            else:
                self.spellCheck.unigram[word] = 1
                
        #update bigram
        orig_text = self.originalBox.toPlainText()
        orig_text = orig_text.lower()
        orig_text = orig_text.strip()
        orig_arr = orig_text.split()

        for i in range(len(orig_arr)-1):
            word1 = orig_arr[i].strip()
            word2 = orig_arr[i+1].strip()
            tup = tuple([word1, word2])
            
            if word1 not in feedback and word2 not in feedback:
                continue
            
            if word1 in self.spellCheck.unigram and word2 in self.spellCheck.unigram:
                if tup in self.spellCheck.bigram:
                    self.spellCheck.bigram[tup] += 1
                else:
                    self.spellCheck.bigram[tup] = 1
                    
        self.feedbackBox.setPlainText('')
    
    def storeBigramUnigram(self):
        bigramName = 'BigramFinal.csv'
        unigramName = 'UnigramFinal.csv'
        
        self.storeBigram(bigramName)
        self.storeUnigram(unigramName)
    
    def storeBigram(self, fileName):

        def sortByCountBigram(bigram):
            return bigram[2]

        bigram_arr = []
    
        for key, value in self.spellCheck.bigram.items():
            bigram_arr.append([key[0], key[1], value])

        bigram_arr.sort(key=sortByCountBigram, reverse=True)

        df = pd.DataFrame(bigram_arr,
                          columns=['Kata 1', 'Kata 2', 'Jumlah'])

        df.to_csv(fileName, sep='#', index=False, header=False)
    
    def storeUnigram(self, fileName):

        def sortByCountUnigram(unigram):
            return unigram[1]

        unigram_arr = []
    
        for key, value in self.spellCheck.unigram.items():
            unigram_arr.append([key, value])

        unigram_arr.sort(key=sortByCountUnigram, reverse=True)

        df = pd.DataFrame(unigram_arr,
                          columns=['Kata', 'Jumlah'])
        
        df.to_csv(fileName, sep='#', index=False, header=False)
        
app = qtw.QApplication(sys.argv)
main = Main()
app.exec()
main.storeBigramUnigram()