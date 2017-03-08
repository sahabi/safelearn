#!/usr/bin/python

from Tkinter import *

class Widget(object):
    def __init__(self, avg):
        self.avg = avg
        self.root = Tk()
        self.root.geometry('100x110')
        self.text = Entry(self.root, bg = 'white')
        self.button = Button(self.root, text = "press me", command = self.insert)
        self.button.pack(padx = 2, pady = 2, anchor= E)
        self.text.pack()

    def insert(self):
        self.avg = float(self.text.get())

