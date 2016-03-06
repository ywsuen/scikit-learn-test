# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 00:43:09 2016

@author: yatwong
"""

class Traverse2Dlist():
    def __init__(self, twoDList):
        self.listLen = len(twoDList)
        self.idx = [0 for i in range(self.listLen)]
        self.twoDList = twoDList
        return
    def getNext(self):
        pointer = self.listLen - 1
        while True:
            if self.idx[pointer]+1 > len(self.twoDList[pointer])-1:
                self.idx[pointer] = 0
                pointer -= 1
                if pointer == -1:
                    return None
            else:
                self.idx[pointer] += 1
                return self.idx