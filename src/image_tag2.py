import numpy as np
import pandas as pd
from collections import deque
from matplotlib import pyplot as plt
from skimage import img_as_ubyte,img_as_float
from skimage.io import imread, imsave, imshow
from skimage.transform import resize, rescale
from skimage import filters
from skimage.morphology import disk
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from pylab import *
import pickle
import time



# The imageCycle() class object takes in a list of axis objects, a dataframe
# and some additional, optional arguments.
# It then connects to each of the axis objects using the mpl_connect() function
# detailed here: https://matplotlib.org/users/event_handling.html
# This passes button press events to the __call__() method when a mouse
# is clicked within an axis.
#
# Coarse Sort:
# This is a bare-bones method of sorting which I used for a first pass through
# the information. This mode requires a single axis object which it uses to
# display the images stored in a directory. A left click includes the image,
# and a right click discards the image.
#
# Fine Sort:
# This lets the user interact with a menu to either discard tag an image with
# any of 9 tags describing the image, crop the image by dragging a rectangular
# selector across the desired area, and remove watermarks by selecting the
# watermarked area with a rectangular selector. The images are then padded to
# a square shape, which prevents them from being warped when they are read
# in by Keras' image preprocessing function. They are saved along with the
# information from the tags in a separate directory.
#



class imageCycle:
    def __init__(self,ax,df,sort_mode='fine',save_mode='resize',imsize=299):
        if len(ax) == 2:
            self.ax = ax[0]
            self.ay = ax[1]
            self.cid = ax[0].figure.canvas.mpl_connect('button_press_event',self)
            self.cid2 = ax[1].figure.canvas.mpl_connect('button_press_event', self)
            self.display_flagscreen()
        else:
            self.ax = ax[0]
            self.cid = ax[0].figure.canvas.mpl_connect('button_press_event',self)
        self.last_entry = ''
        self.inc = 0
        self.inc_t = 0
        self.df = df
        self.im_row = [0,0,0,0,0,0,0,0,0,0,0]
        self.set_index()
        self.next_image()
        self.sort_mode = sort_mode
        self.save_mode = save_mode
        self.imsize = imsize
        self.RS = RectangleSelector(self.ax, self.onselect,
                               drawtype='box', useblit=True,
                               button=[1, 3],
                               minspanx=5, minspany=5,
                               spancoords='pixels',
                               interactive=True)
        columns=['file_name','ready','back_view','side_view','ruler','hand_nature',
                'multiple','contrast','noisy_background','other','choice_count']
        self.df_out = pd.DataFrame(columns = columns)

    # this function uses the mpl_connect module from matplotlib. In a
    # nutshell, whenever there is a click on the screen, this function detects
    # it. If it is within the axes, the event object is passed to either
    # fine_sort() or coarse_sort(), depending on the mode the imageCycle()
    # obect was initialized with.
    def __call__(self,event):
        print('click', event)
        if event.inaxes==self.ax.axes:
            print('within ax axes')
            if self.sort_mode == 'fine':
                pass
            else:
                self.coarse_sort(event)
        elif event.inaxes==self.ay.axes:
            print('within ay axes')
            self.fine_sort(event)

    # This allows a program to pass the object another dataframe
    # and reset the counts.
    def accept_df(self,df):
        self.df = df
        self.inc_t = 0
        user_input = input('reset count?')
        if user_input == 'y':
            self.inc = 0

    def coarse_sort(self, event):
        # This uses a deque to go through the items in the
        # dataframe. The user can choose to include or not include
        # images in the sorted imageset by either left or right clicking.
        # Doubleclick exits.
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
    def fine_sort(self, event):
        print('entering fine_sort()')
        if event.button == 1:
        # These values were found by clicking on the image and reading the
        # x,y information from the click event. This takes the x and y
        # information from a new click, and figures out which part of the
        # 'flag screen' image the user has clicked.
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
                    # Partial view - but possibly useable#
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
        # This just sets the index of the dataframe to be 0 to (length of
        # the dataframe). It makes things tidier.
        self.df['index'] = np.arange(self.df.shape[0])
        self.df.set_index('index', inplace=True)

    def onselect(self, eclick, erelease):
        # I don't need to print this data to the screen, but it doesn't hurt
        # anything, and sometimes it's useful to see what's going on
        # under the hood.
        'eclick and erelease are matplotlib events at press and release'
        print(' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata))
        print(' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata))
        print(' used button   : ', eclick.button)
        # Here I am taking the x and y data from the 'click' and 'release'
        # events, sorting them, and putting them into objects I can
        # access elsewhere.
        self.x1, self.x2 = int(min(eclick.xdata, erelease.xdata)),
                                int(max(eclick.xdata, erelease.xdata))
        self.y1, self.y2 = int(min(eclick.ydata, erelease.ydata)),
                                int(max(eclick.ydata, erelease.ydata))

    def display_flagscreen(self):
        # this displays the flag screen
        flag_screen = imread('program_docs/flag_screen.png')
        self.ay.imshow(flag_screen)
        return

    def save_quit(self):
        # For the coarse sort - saves and quits.
        self.Q.appendleft(self.last_entry)
        self.Q.appendleft(self.a)
        print('Saving progress...')
        print('Goodbye!')
        plt.close()

    def save_quit_fine(self):
        # For the fine_sort mode. Saves the data and closes the program.
        print('Saving progress...')
        self.save_to_txt()
        print('Goodbye!')
        plt.close()

    def save_to_txt(self):
        # For the fine_sort mode. This saves the data to a text file called
        # meta.txt, including all the tags the user created.
        print('Saving progress...')
        with open('../data/processed_2/meta.txt','a') as f:
            for i in range(self.df_out.shape[0]):
                f.write(';'.join(self.df_out.iloc[i].values.astype('str'))+';\n')

    def watermark(self):
        # This was a lot of trial and error, but I finally found a combination
        # of techniques that is both effective at removing watermarks, and
        # which doesn't take too much time (I have to sort through several
        # thousand of these, after all)
        psf = np.ones((5, 5)) / 25
        c = self.b[self.y1:self.y2, self.x1:self.x2]
        dim = c.shape
        # This mirrors the top quarter and bottom quarter of the selected
        # rectangle over the middle half.
        if dim[0] < dim[1]:
            h = dim[0]//4
            c[h:2*h] = c[:h]
            c[2*h:3*h] = c[3*h:4*h]
        else:
            h = dim[1]//4
            c[:,h:2*h] = c[:,0:h]
            c[:,2*h:3*h] = c[:,3*h:4*h]
        # After playing around a bit, this seems to be the magic combo:
        # I apply the median filter with a disk that's 10 across, over each
        # channel of the rectangle. Then I apply a gaussian filter to the
        # result.
        c = filters.gaussian(np.dstack([filters.median(c[:,:,i],disk(10)) for i in range(3)]),
                             sigma=5,multichannel=True)
        # I need to convert the array back to its original format, otherwise
        # it won't fit back in the image properly.
        c = img_as_ubyte(c)
        self.b[self.y1:self.y2, self.x1:self.x2] = c
        self.ax.imshow(self.b)
        self.ax.figure.canvas.draw()


    def get_rect(self):
        # This crops the image to the rectangle selected by the user.
        print('entering get_rect()')
        self.b = self.b[self.y1:self.y2,self.x1:self.x2]
        self.ax.imshow(self.b)
        self.ax.figure.canvas.draw()
        return

    def pad_img(self):
        # This pads the image to a square.
        dim = self.b.shape
        pad_amt = abs(dim[0] - dim[1])//2
        pad_axis = 0 if dim[0] < dim[1] else 1
        if dim[0] < dim[1]:
            pad_arr = np.zeros((pad_amt,dim[0**pad_axis],3),dtype=uint8)
        else:
            pad_arr = np.zeros((dim[0**pad_axis],pad_amt,3),dtype=uint8)
        self.b = np.concatenate((pad_arr,self.b,pad_arr),axis=pad_axis)


    def save_cont_fine(self):
        # This saves the image, along with the metadata provided by the
        # user.
        self.inc += 1
        self.df_out.loc[self.inc_t] = self.im_row
        self.pad_img()
        # I decided to resize the images in Keras instead, to give myself
        # some flexability down the line.
        print('No resize for now...')
        #print('resizing image...')
        #self.b = rescale(self.b,self.imsize/self.b.shape[0])
        im_path = '../data/processed_2/{}'.format(self.a.split('/')[4])
        print(im_path)
        imsave(im_path, self.b)
        print('saved.')
        self.next_image()
        self.im_row = [0,0,0,0,0,0,0,0,0,0,0]

    def skip(self):
        print('entering skip()')
        return

    def next_image(self):
        print('entering next_image()')
        if self.inc_t >= (self.df.shape[0] - 1):
            print('Finished current queue.')
            print('Saving and quitting')
            self.save_quit_fine()
        # Important! If the axis isn't cleared, MatPlotLib will keep
        # the images it's drawn in memory, and the program will slow down
        # to a crawl. This prevents that.
        self.ax.cla()
        # Reads and displays the next image in the dataframe.
        self.a = self.df.file_path.loc[self.inc_t]
        self.b = imread(self.a)
        self.ax.imshow(self.b)
        # it's nice to have an idea of where you are.
        self.ax.set_title(self.a[18:]+': Saved Images: {}'.format(self.inc))
        self.ay.set_title(str(self.inc_t)+'images out of'+str(self.df.shape[0])+'so far')
        self.ax.figure.canvas.draw()
        self.inc_t += 1
        return

    def back_one(self):
        # Go back one image.
        print('entering back_one()')
        self.inc_t -= 2
        self.next_image()
        return


















#
