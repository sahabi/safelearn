#!/usr/bin/python

try:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter 
except ImportError:
    # for Python3
    from tkinter import *   ## notice lowercase 't' in tkinter here

class Widget(object):
    def __init__(self, avg):
        self.avg = avg
        self.root = Tk()
        self.root.geometry('100x110')
        self.text = Entry(self.root, bg = 'white')
        self.button = Button(self.root, text = "press me", command = self.insert)
        self.button.pack(padx = 2, pady = 2, anchor= E)
        self.text.pack()
        self.root.mainloop()

    def insert(self):
        self.avg = float(self.text.get())
        self.root.destroy()

if __name__ == '__main__':
    w = Widget(1)
    #w.root.mainloop()
    print (w.avg)

