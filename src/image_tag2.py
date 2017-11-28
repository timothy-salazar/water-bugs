import numpy as np
import pandas as pd
from collections import deque
from matplotlib import pyplot as plt
from skimage import img_as_ubyte,img_as_float
from skimage.io import imread, imsave, imshow
from skimage.transform import resize, rescale
from skimage.restoration import inpaint, wiener
from skimage import filters
from skimage.morphology import disk
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from pylab import *
import pickle
import time
import pdb

#order_dict = {'Odonata-Anisoptera (Dragonflies)':'a_odonata'}

# trying to make this flexible
# I think that i'll have more options if I use a dataframe

class imageCycle:
    def __init__(self,ax,df,sort_mode='fine',save_mode='resize',imsize=299):
        #print('line 16')
        #pdb.set_trace()
        if len(ax) == 2:
            #print('line 18')
            self.ax = ax[0]
            self.ay = ax[1]
            self.cid = ax[0].figure.canvas.mpl_connect('button_press_event',self)
            self.cid2 = ax[1].figure.canvas.mpl_connect('button_press_event', self)
            self.display_flagscreen()

        else:
            #print('line 24')
            self.ax = ax[0]
            self.cid = ax[0].figure.canvas.mpl_connect('button_press_event',self)
        self.last_entry = ''
        self.inc = 0
        self.inc_t = 0
        self.df = df
        #self.Q = deque(df.file_path)
        self.im_row = [0,0,0,0,0,0,0,0,0,0,0]
        self.set_index()
        self.next_image()
        self.sort_mode = sort_mode
        self.save_mode = save_mode
        self.imsize = imsize
        self.RS = RectangleSelector(self.ax, self.onselect,
                               drawtype='box', useblit=True,
                               button=[1, 3],  # don't use middle button
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)
        columns=['file_name','ready','back_view','side_view','ruler','hand_nature',
                'multiple','contrast','noisy_background','other','choice_count']
        self.df_out = pd.DataFrame(columns = columns)

    # this function is a bit tricky. It uses the mpl_connect module from
    # matplotlib. In a nutshell, whenever there is a click on the screen
    # this function detects it, and the path to a picture is popped from
    # the queue.
    def __call__(self,event):
        #print('line 45')
        print('click', event)
        # note to self: change this to deal with both axis objects
        if event.inaxes==self.ax.axes:
            print('within ax axes')
            if self.sort_mode == 'fine':
                pass
            else:
                self.coarse_sort(event)
        elif event.inaxes==self.ay.axes:
            print('within ay axes')
            self.fine_sort(event)

    def accept_df(self,df):
        self.df = df
        self.inc_t = 0
        user_input = input('reset count?')
        if user_input == 'y':
            self.inc = 0

    def coarse_sort(self, event):
        #print('line 53')
        if len(self.Q) == 0:
            print('Queue is empty, quitting...')
            self.save_quit()
        if event.dblclick:
            print('double click...')
            self.save_quit()
        if event.button == 1:
            self.skip()
            print(self.a,'skipped.')
        elif event.button == 3:
            self.inc += 1
            print(self.a,'saved.')
        self.last_entry = self.a
        self.next_image()
        return
    # This function sorts using the flag screen - building up a
    # list of metadata that we can to decide which images we'll
    # feed our model as we become better at image processing
    # (I'll come back to this once I have an MVP)

    def fine_sort(self, event):
        print('entering fine_sort()')
        if event.button == 1:
            #self.pause = True
            if 1100<event.x<1440:
                #Column 1: save, skip, back one
                if 558<event.y<856:
                    if (event.x > 1268) and (event.y < 705):
                        ############
                        ### Crop ###
                        ############
                        print('cropping image')
                        self.get_rect()
                    else:
                        ############
                        ### Save ###
                        ############
                        print('saving...')
                        if sum(self.im_row) == 0:
                            self.im_row[0] = self.a
                            self.im_row[1] = 1
                        else:
                            self.im_row[0] = self.a
                        self.save_cont_fine()
                elif 390<event.y<548:
                    # Skip
                    #self.pause = False
                    print('skipping entry',self.a)
                    self.im_row = [0,0,0,0,0,0,0,0,0,0,0]
                    self.next_image()
                elif 238<event.y<385:
                    # Back One
                    print('going back one...')
                    self.back_one()
                elif 110<event.y<225:
                    # Quit
                    self.save_quit_fine()
            # columns=['file_name','ready','back_view','side_view','ruler','hand_nature',
            #         'multiple','contrast','noisy_background','other','choice_count']
            elif 1448<event.x<1619:
                # column 2: back view, crop, ruler, hand/nature
                if 709<event.y<856:
                    # Back View
                    print('adding "Back View" attribute')
                    self.im_row[2] = 1
                elif 558<event.y<701:
                    # Watermark
                    print('watermark...')
                    self.watermark()
                elif 390<event.y<548:
                    # Ruler
                    # for bugguide, I used this for trichoptera cases
                    print('adding "ruler" attribute')
                    self.im_row[4] = 1
                elif 238<event.y<385:
                    # Hand/nature
                    print('adding "Hand/Nature" attribute')
                    self.im_row[5] = 1
                elif 110<event.y<225:
                    # Quit
                    self.save_quit_fine()
            elif 1625<event.x<1792:
                # column 3: multiple, contrast, noisy background, other
                if 709<event.y<856:
                    # Side View
                    print('adding "Side View" attribute')
                    self.im_row[3] = 1
                elif 558<event.y<701:
                    # Contrast
                    print('adding "High Contrast" attribute')
                    self.im_row[7] = 1
                elif 390<event.y<548:
                    # Noisy Background
                    print('adding "Noisy Background" attribute')
                    self.im_row[8] = 1
                elif 238<event.y<385:
                    # Multiple/other
                    # most important note is that I used this for
                    # adult specimens (not for the first few, however, should
                    # fix that)
                    print('adding "Multiple/other" attribute')
                    self.im_row[6] = 1
                elif 110<event.y<225:
                    # Quit
                    self.save_quit_fine()
        if event.button == 3:
            if 1448<event.x<1619:
                # column 2: back view, side view, ruler, hand/nature
                if 709<event.y<856:
                    # Back View
                    self.im_row[2] = 1
                elif 558<event.y<701:
                    # watermark
                    print('watermark...')
                    self.watermark()
                elif 390<event.y<548:
                    # Ruler
                    self.im_row[4] = 1
                elif 238<event.y<385:
                    # Hand/nature
                    self.im_row[5] = 1
                elif 110<event.y<225:
                    # Quit
                    self.save_quit_fine()
            elif 1625<event.x<1792:
                # column 3: multiple, contrast, noisy background, other
                if 709<event.y<856:
                    # Side View
                    self.im_row[3] = 1
                elif 558<event.y<701:
                    # Contrast
                    self.im_row[7] = 1
                elif 390<event.y<548:
                    # Noisy Background
                    self.im_row[8] = 1
                elif 238<event.y<385:
                    #######################################
                    # Patrial view - but possibly useable#
                    ######################################
                    print("OTHER")
                    self.im_row[9] = 1
                elif 110<event.y<225:
                    # Quit
                    self.save_quit_fine()
                else:
                    return
            self.im_row[0] = self.a
            self.save_cont_fine()
        self.last_entry = self.a
        return

    def set_index(self):
        #print('line 36')
        self.df['index'] = np.arange(self.df.shape[0])
        self.df.set_index('index', inplace=True)

    def onselect(self, eclick, erelease):
        'eclick and erelease are matplotlib events at press and release'
        print(' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata))
        print(' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata))
        print(' used button   : ', eclick.button)
        self.x1, self.x2 = int(min(eclick.xdata, erelease.xdata)),int(max(eclick.xdata, erelease.xdata))
        self.y1, self.y2 = int(min(eclick.ydata, erelease.ydata)),int(max(eclick.ydata, erelease.ydata))
        # self.xdata1, self.ydata1 = eclick.xdata, eclick.ydata
        # self.xdata2, self.ydata2 = erelease.xdata, erelease.ydata
    # this displays the flag screen
    def display_flagscreen(self):
        flag_screen = imread('program_docs/flag_screen.png')
        self.ay.imshow(flag_screen)
        return

    def save_quit(self):
        self.Q.appendleft(self.last_entry)
        self.Q.appendleft(self.a)
        print('Saving progress...')
        print('Goodbye!')
        plt.close()

    def save_quit_fine(self):
        # self.Q.appendleft(self.last_entry)
        # self.Q.appendleft(self.a)
        print('Saving progress...')
        # with open('../data/resized/meta.txt','a') as f:
        #     for i in range(self.df_out.shape[0]):
        #         f.write(';'.join(self.df_out.iloc[i].values.astype('str'))+';\n')
        self.save_to_txt()
        print('Goodbye!')
        plt.close()

    def save_to_txt(self):
        print('Saving progress...')
        with open('../data/processed_2/meta.txt','a') as f:
            for i in range(self.df_out.shape[0]):
                f.write(';'.join(self.df_out.iloc[i].values.astype('str'))+';\n')

    def watermark(self):
        #self.mask = np.zeros(self.b.shape[:2])
        #self.mask[self.y1:self.y2, self.x1:self.x2] = 1
        # for layer in range(self.b.shape[-1]):
        #     self.b[np.where(self.mask)] = 0
        #self.ax.imshow(self.b)
        #self.ax.figure.canvas.draw()
        #time.sleep(3)
        psf = np.ones((5, 5)) / 25
        c = self.b[self.y1:self.y2, self.x1:self.x2]
        dim = c.shape
        if dim[0] < dim[1]:
            h = dim[0]//4
            #print(dim,h)
            #self.test = c
            c[h:2*h] = c[:h]
            c[2*h:3*h] = c[3*h:4*h]
        else:
            h = dim[1]//4
            #print(dim,h)
            #self.test = c
            c[:,h:2*h] = c[:,0:h]
            c[:,2*h:3*h] = c[:,3*h:4*h]


        #c = img_as_float(c)
        #c = np.dstack([filters.median(c[:,:,i],disk(10)) for i in range(3)])
        # After playing around a bit, this seems to be the magic combo
        c = filters.gaussian(np.dstack([filters.median(c[:,:,i],disk(10)) for i in range(3)]),
                             sigma=5,multichannel=True)
        #c = filters.gaussian(c, sigma=5, multichannel=True)
        c = img_as_ubyte(c)
        self.b[self.y1:self.y2, self.x1:self.x2] = c
        # this method was far too time intensive
        #self.b = inpaint.inpaint_biharmonic(self.b, self.mask, multichannel=True)
        self.ax.imshow(self.b)
        self.ax.figure.canvas.draw()


    def get_rect(self):
        print('entering get_rect()')
        # x1, x2 = int(min(self.xdata1, self.xdata2)),int(max(self.xdata1, self.xdata2))
        # y1, y2 = int(min(self.ydata1, self.ydata2)),int(max(self.ydata1, self.ydata2))
        self.b = self.b[self.y1:self.y2,self.x1:self.x2]
        self.ax.imshow(self.b)
        self.ax.figure.canvas.draw()
        return


    def pad_img(self):
        dim = self.b.shape
        pad_amt = abs(dim[0] - dim[1])//2
        pad_axis = 0 if dim[0] < dim[1] else 1
        if dim[0] < dim[1]:
            pad_arr = np.zeros((pad_amt,dim[0**pad_axis],3),dtype=uint8)
        else:
            pad_arr = np.zeros((dim[0**pad_axis],pad_amt,3),dtype=uint8)
        self.b = np.concatenate((pad_arr,self.b,pad_arr),axis=pad_axis)


    def save_cont_fine(self):
        self.inc += 1
        self.df_out.loc[self.inc_t] = self.im_row
        self.pad_img()
        print('No resize for now...')
        #print('resizing image...')
        #self.b = rescale(self.b,self.imsize/self.b.shape[0])
        im_path = '../data/processed_2/{}'.format(self.a.split('/')[4])
        print(im_path)
        imsave(im_path, self.b)
        print('saved.')
        self.next_image()
        self.im_row = [0,0,0,0,0,0,0,0,0,0,0]


        # print('entering save_cont_fine()')
        # self.df_out.loc[self.inc_t] = self.im_row
        # #self.pause = False
        # #if self.save_mode == 'resize':
        # print('resizing image...')
        # # self.pad_img()
        # dim = self.b.shape
        # print('dimensions of b:',dim)
        # pad_amt = abs(dim[0] - dim[1])//2
        # print('pad_amt:',pad_amt)
        # pad_axis = 0 if dim[0] < dim[1] else 1
        # print('pad axis:', pad_axis)
        # pad_arr = np.zeros((pad_amt,dim[0**pad_axis],3),dtype = uint8)
        # print('shape pad_arr:',shape(pad_arr))
        # c = self.b.copy()
        #
        # b_pad = np.concatenate((pad_arr,c), axis = pad_axis)
        # b_pad = np.concatenate((b_pad,pad_arr), axis = pad_axis)
        # print('shape b_pad:',b_pad.shape)
        # im_path = self.a.replace('troutnut','resized').replace('bug_guide','resized')
        # print(im_path)
        #self.c = self.b
        # d = resize(b_pad,self.imsize)
        # print('shape d:',d.shape)
        # imsave(im_path, d)
        # print('saved.')
        # self.next_image()
        # self.im_row = [0,0,0,0,0,0,0,0,0,0,0]
        # return

    def skip(self):
        print('entering skip()')
        # if self.save_mode != resize:
        #     skip_ind = np.where(self.df.file_path == self.a)[0][0]
        #     self.df.drop(skip_ind)
        # else:
        #     pass
        return

    def next_image(self):
        print('entering next_image()')
        if self.inc_t >= (self.df.shape[0] - 1):
            print('Finished current queue.')
            print('Saving and quitting')
            self.save_quit_fine()
        # if this doesn't work, we'll try clf
        self.ax.cla()
        #self.a = self.Q.pop()
        self.a = self.df.file_path.loc[self.inc_t]
        self.b = imread(self.a)
        self.ax.imshow(self.b)
        self.ax.set_title(self.a[18:]+': Saved Images: {}'.format(self.inc))
        self.ay.set_title(str(self.inc_t)+'images out of'+str(self.df.shape[0])+'so far')
        #self.ax.set_xlabel(str(self.inc_t)+'images so far')
        self.ax.figure.canvas.draw()
        self.inc_t += 1
        return

    def back_one(self):
        print('entering back_one()')
        self.inc_t -= 2
        self.next_image()
        # current = np.where(self.df.file_path == self.a)[0][0]
        # self.Q.append(self.df.file_path.iloc[last])
        return


















#
