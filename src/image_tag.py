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
    def __init__(self,ax,ay):
        self.ax = ax
        self.ay = ay
        self.cid = ax.figure.canvas.mpl_connect('button_press_event',self)
        self.cid2 = ay.figure.canvas.mpl_connect('button_press_event',self)
        self.last_entry = ''
        with open('pickle/imQ2.pkl','rb') as f:
            self.Q = pickle.load(f)
        columns=['file_name','ready','back_view','side_view','ruler','hand_nature',
                'multiple','contrast','noisy_background','other','choice_count']
        self.df = pd.DataFrame(columns = columns)
        # with open("../data/second_pass.txt","a") as f:
        #     f.write(';'.join(columns)+'\n')
        self.pause = False
        self.next_image()
        self.im_row = [0,0,0,0,0,0,0,0,0,0,0]

        #self.Q = self.unpickle_queue()

    # this function is a bit tricky. It uses the mpl_connect module from
    # matplotlib. In a nutshell, whenever there is a click on the screen
    # this function detects it, and the path to a picture is popped from
    # the queue. It's displayed, and the

    def save_quit(self,a):
        self.Q.appendleft(self.last_entry)
        self.Q.appendleft(a)
        print('Saving progress...')
        self.pickle_queue()
        print('Goodbye!')
        plt.close()

    def save_cont(self):
        # end_token = ";1;None;\n"
        print('saving',self.last_entry,'to second_pass.txt...')
        with open("../data/second_pass.txt","a") as f:
            f.write(str(self.im_row)[1:][:-1].replace(',',';') +';\n')
        ind = self.df.shape[0]
        self.df.loc[ind] = self.im_row
        self.pause = False
        print('saved.')
        self.next_image()
        self.im_row = [0,0,0,0,0,0,0,0,0,0,0]

    def next_image(self):
        self.a = self.Q.pop()
        b = imread(self.a)
        self.ax.imshow(b)
        self.ax.figure.canvas.draw()

    def __call__(self,event):
        Q = self.unpickle_queue()
        print('click', event)
        if event.inaxes!=self.ay.axes:
            return
        # if self.pause == False:
        #     self.next_image()
        #     self.im_row = [0,0,0,0,0,0,0,0,0,0,0]
        if event.button == 1:
            self.pause = True
            if 1100<event.x<1440:
                #Column 1: save, skip, back one
                if 558<event.y<856:
                    ########
                    # Save #
                    ########
                    if sum(self.im_row) == 0:
                        self.im_row[0] = self.a
                        self.im_row[1] = 1
                    else:
                        self.im_row[0] = self.a
                    self.save_cont()
                    # print('saving',self.last_entry,'to second_pass.txt...')
                    # with open("../data/second_pass.txt","a") as f:
                    #     f.write(self.last_entry+end_token)
                    # ind = self.df.shape[0]
                    # self.df.loc[ind] = self.im_row
                    # self.pause = False
                    # print('saved.')
                    # self.next_image()
                    # self.im_row = [0,0,0,0,0,0,0,0,0,0,0]
                elif 390<event.y<548:
                    # Skip
                    self.pause = False
                    print('skipping entry',self.a)
                    self.next_image()
                    self.im_row = [0,0,0,0,0,0,0,0,0,0,0]
                elif 238<event.y<385:
                    # Back One
                    pass
                elif 110<event.y<225:
                    # Quit
                    self.save_quit(self.a)

            elif 1448<event.x<1619:
                # column 2: back view, side view, ruler, hand/nature
                if 709<event.y<856:
                    # Back View
                    self.im_row[2] = 1
                elif 558<event.y<701:
                    # Side View
                    self.im_row[3] = 1
                elif 390<event.y<548:
                    # Ruler
                    self.im_row[4] = 1
                elif 238<event.y<385:
                    # Hand/nature
                    self.im_row[5] = 1
                elif 110<event.y<225:
                    # Quit
                    self.save_quit(self.a)
            elif 1625<event.x<1792:
                # column 3: multiple, contrast, noisy background, other
                if 709<event.y<856:
                    #Multiple
                    self.im_row[6] = 1
                elif 558<event.y<701:
                    # Contrast
                    self.im_row[7] = 1
                elif 390<event.y<548:
                    # Noisy Background
                    self.im_row[8] = 1
                elif 238<event.y<385:
                    # Other
                    self.im_row[9] = 1
                elif 110<event.y<225:
                    # Quit
                    self.save_quit(self.a)
            # print((event.x))
            # print('skipping entry',self.a)
        # elif event.button == 3:
        #     print('saving',self.last_entry,'to second_pass.txt...')
        #     with open("../data/second_pass.txt","a") as f:
        #         f.write(self.last_entry+end_token)
        #     print('saved.')
        if event.button == 3:
            if 1448<event.x<1619:
                # column 2: back view, side view, ruler, hand/nature
                if 709<event.y<856:
                    # Back View
                    self.im_row[2] = 1
                elif 558<event.y<701:
                    # Side View
                    self.im_row[3] = 1
                elif 390<event.y<548:
                    # Ruler
                    self.im_row[4] = 1
                elif 238<event.y<385:
                    # Hand/nature
                    self.im_row[5] = 1
                elif 110<event.y<225:
                    # Quit
                    self.save_quit(self.a)
            elif 1625<event.x<1792:
                # column 3: multiple, contrast, noisy background, other
                if 709<event.y<856:
                    #Multiple
                    self.im_row[6] = 1
                elif 558<event.y<701:
                    # Contrast
                    self.im_row[7] = 1
                elif 390<event.y<548:
                    # Noisy Background
                    self.im_row[8] = 1
                elif 238<event.y<385:
                    # Other
                    self.im_row[9] = 1
                elif 110<event.y<225:
                    # Quit
                    self.save_quit(self.a)
                else:
                    return
            self.save_cont()

        self.last_entry = self.a

    # this function builds a queue from the metadata files in the
    # image directories.
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

    # After I've culled the closeups, etc., I'll want to go through
    # and make a second pass - pulling out only the shots that I can use
    # without any further manipulation for a quick, initial run through
    def read_from_txt(self):
        self.Q = deque()
        df = pd.read_csv('../data/first_pass.txt',sep=';')
        for f in df.file_name:
            self.Q.append(f)
        self.pickle_queue()

    # def rebuild_queue(self):
    #     check_dir = False
    #     check_file = False
    #     print('check point 1')
    #     with open('../data/culled.txt','r') as f:
    #         for line in f:
    #             pass
    #         last = line
    #     print('check point 2')
    #     last_dir = last.split('/')[3]
    #     last_file = last.split('/')[4].split(';')[0]
    #     self.Q = deque()
    #     print('check point 3')
        #Q = deque()
        # data_dict = pd.read_json('urlinfo.json')
        # dirs = data_dict.directory.values
        # for d in dirs:
            # if d == last_dir:
            #     check_dir == True
            # if check_dir == True:
        # print('check point 4')
        # path_name = '../data/troutnut/ephemeroptera/meta.txt'
        # df = pd.read_csv(path_name, sep=';')
        # for f in df.file_name:
        #     #print('check point 5')
        #     if f == last_file.split('.')[0]:
        #         print('in 1')
        #         check_file = True
        #     if check_file == True:
        #         print('in 2')
        #         im_path = '../data/troutnut/ephemeroptera/{}.jpg'.format(f)
        #         self.Q.append(im_path)
        #                 #Q.append(im_path)
        # self.pickle_queue()
        #
    #
    # def draw_next(self):
    #     self.a = self.Q.pop()
    #     b = imread(a)
    #     self.ax.imshow(b)
    #     plt.show()

    def pickle_queue(self):
        with open('pickle/imQ2.pkl','wb') as f:
            pickle.dump(self.Q,f)
        # recovers the previously pickled queue
    def unpickle_queue(self):
        if len(self.Q) > 0:
            return
        else:
            with open('pickle/imQ2.pkl','rb') as f:
                self.Q = pickle.load(f)


    def shift_left(self):
    # Don't pay too much attention to this one - I screwed up
    # earlier (problem is solved now). Just a function to fix my
    # mistake.
    # this takes all the file names I read in earlier, looks up
    # their position in the 'read_meta' queue, and replaces them
    # with the entry one position before them.
        self.read_meta()
        origQ = np.array(self.Q)
        self.read_from_txt()
        self.Q.reverse()
        copyQ = np.array(self.Q)
        # a = np.unique(copyQ,return_counts=True)
        # b = a[1]
        # copyQ = [i for i in a[0]]
        # c = np.where(b != 1)
        # d = a[0][c]
        # s = [np.where(copyQ==i) for i in d]
        # [[ for j in i]for i in s]
        #
        #
        a = [np.where(origQ == i)[0][0] for i in copyQ]
        b = [i - 1 for i in a]
        self.Q = deque()
        [self.Q.append(i) for i in origQ[b]]
        with open("../data/left_shifted3.txt","a") as f:
            f.write('file_name;choice_count;extra;\n')
            for i in self.Q:
                f.write(i+';1;None;\n')
        self.pickle_queue()

if __name__ == '__main__':

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ay = fig.add_subplot(122)
    flag_screen = imread('program_docs/flag_screen.png')
    ay.imshow(flag_screen)
    ax.set_title("cycle-a-tron")
    line, = ax.plot([0],[0])
    imc = imageCycle(ax,ay)
    #imc.rebuild_queue()
    #imc.read_meta()
    #imc.read_from_txt()
    imc.shift_left()
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
