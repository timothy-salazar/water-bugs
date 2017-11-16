import numpy as np
import pandas as pd
from collections import deque
from matplotlib import pyplot as plt
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
import pickle
import time


class imageCycle:
    # def __init__(self):
    #     self.fig = plt.figure()
    #     self.ax = self.fig.add_subplot(111)
    #     self.Q = self.unpickle_queue()
    #     self.cid = self.fig.canvas.mpl_connect('button_press_event', self)
    #     self.cont_var = True
    def __init__(self,line):
        self.line = line
        self.cid = line.figure.canvas.mpl_connect('button_press_event',self)
        self.last_entry = ''
        self.Q = deque()

    def __call__(self,event):
        #time.sleep(.1)
        end_token = ";1;None;\n"
        Q = self.unpickle_queue()
        print('click', event)
        if event.inaxes!=self.line.axes: return
        a = self.Q.pop()
        self.last_entry = a
        b = imread(a)
        imshow(b)
        self.line.figure.canvas.draw()
        if event.dblclick:
            self.Q.appendleft(self.last_entry)
            self.Q.appendleft(a)
            print('Saving progress...')
            self.pickle_queue()
            print('Goodbye!')
            plt.close()
        elif event.button == 1:
            print('skipping entry',a)
        elif event.button == 3:
            print('saving',a,'to culled.txt...')
            with open("../data/culled.txt","a") as f:
                f.write(a+end_token)
            print('saved.')

        #self.pickle_queue()

    def read_meta(self):
        self.Q = deque()
        data_dict = pd.read_json('urlinfo.json')
        dirs = data_dict.directory.values
        for d in dirs:
            path_name = '../data/troutnut/{}/meta.txt'.format(d)
            df = pd.read_csv(path_name, sep=';')
            for f in df.file_name:
                im_path = '../data/troutnut/{}/{}.jpg'.format(d,f)
                self.Q.append(im_path)
        self.pickle_queue()

    def draw_next(self):
        self.a = self.Q.pop()
        b = imread(a)
        self.ax.imshow(b)
        plt.show()

    def pickle_queue(self):
        with open('pickle/imQ.pkl','wb') as f:
            pickle.dump(self.Q,f)
        # recovers the previously pickled queue
    def unpickle_queue(self):
        if len(self.Q) > 0:
            pass
        else:
            with open('pickle/imQ.pkl','rb') as f:
                self.Q = pickle.load(f)

if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("cycle-a-tron")
    line, = ax.plot([0],[0])
    imc = imageCycle(line)
    #imc.read_meta()
    plt.show()


#
#-------------------
# Last entry before code change
# ../data/troutnut/ephemeroptera/picture1996.jpg
#-------------------
#
#plecoptera 4456
#egaloptera/picture1480.jpg /6/3
#tri 3300, 3298
# note - many of the plecoptera images have different backgrounds
# these were not included, but they could increase the number of images
# i have available if I can get the image processing going well

#../data/troutnut/diptera/picture2883.jpg
#./data/troutnut/diptera/picture2883.jpg
#../data/troutnut/diptera/picture2882.jpg
#./data/troutnut/diptera/picture2885.jpg
# old code
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title('click to skip, click with 2 to save, double to quit')
# imcycle = imageCycle(ax)
# plt.show()
#
# fig, ax = plt.subplots()
# ax.plot(np.random.rand(10))
#
#
#
# def imcycle(self):
#     while self.cont_var == True:
#
#         yield
#
# def onclick(self,event):
#     print(event)
#     if event.dblclick:
#         self.cont_var == False
#
# def __call__(self, event):
#     end_token = "; None;\n"
#     print('click', event)
#     a = self.Q.pop()
#     b = imread(a)
#     self.ax.imshow(b)
#     if event.dblclick:
#         return
#
# if event.button == 1:
#     self.draw_next()
# elif event.button == 3:
#     with open("../data/culled.txt","a") as f:
#         f.write(self.a+end_token)
#     self.draw_next()
#
#
# def draw_next(Q,ax):
#     a = Q.pop()
#     b = imread(a)
#     ax.imshow(b)
#     # ax.draw()
#
#
# def onclick(event):
#     print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#           ('double' if event.dblclick else 'single', event.button,
#            event.x, event.y, event.xdata, event.ydata))
#     print(event)
# #    draw_next()
#
# #cid = fig.canvas.mpl_connect('button_press_event', onclick)
# def main():
#     plt.ion()
#     with open('pickle/imQ.pkl','rb') as f:
#         Q = pickle.load(f)
#     fig, ax = plt.subplots()
#     cid = fig.canvas.mpl_connect('button_press_event', onclick)
#     cont_var = True
#     while cont_var == True:
#         plt.waitforbuttonpress()
#         draw_next(Q,ax)
#
#
# if __name__ == '__main__':
#     main()
#
