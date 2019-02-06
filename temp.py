import cv2
import numpy as np
import math
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
import matplotlib.pylab as plt
from keras.models import model_from_json

'''def SUM(b, c, d):
    summ = 0
    pom = ann.predict(np.array(prepare_for_ann(b),np.float32))
    if(c < 10):
        summ += display_result(pom)
    elif(d < 10):
        summ -= display_result(pom)
    
    return summ'''

def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255. 
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255

def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()

def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona 
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
        
    return ready_for_ann


def convert_output(alphabet):
    '''Konvertovati alfabet u niz pogodan za obučavanje NM,
        odnosno niz čiji su svi elementi 0 osim elementa čiji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann():
    '''Implementacija veštačke neuronske mreže sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann
    
def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''
    X_train = np.array(X_train, np.float32) # dati ulazi
    y_train = np.array(y_train, np.float32) # zeljeni izlazi za date ulaze
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=2000, batch_size=1, verbose = 0, shuffle=False) 
      
    return ann

def winner(output): # output je vektor sa izlaza neuronske mreze
    '''pronaći i vratiti indeks neurona koji je najviše pobuđen'''
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    alphabet=[0,1,2,3,4,5,6,7,8,9]
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def line(lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            firstCoor = (x1, y1)
            secondCoor = (x2, y2)
    return firstCoor, secondCoor

def line1(lines1):
    if lines1 is not None:
        for line in lines1:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (100, 0, 255), 2)
            firstCoor1 = (x1, y1)
            secondCoor1 = (x2, y2)
    return firstCoor1, secondCoor1
def dilate(image):
    kernel = np.ones((3,3))
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3))
    return cv2.erode(image, kernel, iterations=1)

def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y
  
def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)
  
def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)
  
def unit(v):
    x,y = v
    mag = length(v)
    return (x/mag, y/mag)
  
def distance(p0,p1):
    return length(vector(p0,p1))
  
def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)
  
def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)

def pointToLine(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, r)

def image_gray(orig_frame):
    return cv2.cvtColor(orig_frame, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def select_roi(orig_frame, image_bin, a1, b1, a2, b2):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28. 
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    summ = 0
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
        c = (x, y)
        distanceLineDot1, flag = pointToLine(c, a1, b1)
        distanceLineDot2, flag2 = pointToLine(c, a2, b2)
        if distanceLineDot1 < 10 or distanceLineDot2 < 10:      
            if h < 100 and h > 7 and w > 1: 
                # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
                # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
                region = image_bin[y:y+h+1,x:x+w+1]
                regions_array.append([resize_region(region), (x,y,w,h)]) 
                cv2.rectangle(orig_frame,(x,y),(x+w,y+h),(0,255,0),2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]
    if len(sorted_regions) != 0:
        pom = ann.predict(np.array(prepare_for_ann(sorted_regions),np.float32))
        if distanceLineDot1 < 10:
            summ += sum(display_result(pom))
        elif distanceLineDot2 < 10:
            summ -= sum(display_result(pom))
    
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return orig_frame, sorted_regions, summ 


pomocniFajl = open('mreza.json', 'r')
mreza = pomocniFajl.read()
pomocniFajl.close()
ann = model_from_json(mreza)
ann.load_weights("mreza.h5")
 
video = cv2.VideoCapture("video/video-0.avi")
suma0 = 0
while True:
    ret, orig_frame = video.read()
    
    
    if not ret:
        video = cv2.VideoCapture("video/video-0.avi")
        break
 
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    low_blue = np.array([0, 0, 100])
    up_blue= np.array([20, 20, 255])
    low_green = np.array([0, 180, 0])
    up_green = np.array([20, 255, 20])
    mask1 = cv2.inRange(rgb, low_green, up_green)
    mask = cv2.inRange(rgb, low_blue, up_blue)
    edges = cv2.Canny(mask, 75, 150)
    edges1 = cv2.Canny(mask1, 75, 150)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
    lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
   
    
    origFrameGray = image_gray(frame)
    origFrameBin = image_bin(origFrameGray)
    a1, b1 = line(lines)
    a2, b2 = line1(lines1)
    a, b, c = select_roi(frame, origFrameBin, a1, b1, a2, b2)
    
    #suma = SUM(b, c, d)
    
    suma0 += c
    
   # print(d)
    
    
    

    cv2.imshow("frame", frame)
    cv2.imshow("edges1", edges1)
    cv2.imshow("edges", edges)
 
 
    key = cv2.waitKey(25)
    if key == 27:
        break

print(suma0)

video1 = cv2.VideoCapture("video/video-1.avi")
suma01 = 0
while True:
    ret, orig_frame = video1.read()
    
    
    if not ret:
        video1 = cv2.VideoCapture("video/video-1.avi")
        break
 
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    low_blue = np.array([0, 0, 100])
    up_blue= np.array([20, 20, 255])
    low_green = np.array([0, 180, 0])
    up_green = np.array([20, 255, 20])
    mask1 = cv2.inRange(rgb, low_green, up_green)
    mask = cv2.inRange(rgb, low_blue, up_blue)
    edges = cv2.Canny(mask, 75, 150)
    edges1 = cv2.Canny(mask1, 75, 150)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
    lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
   
    
    origFrameGray = image_gray(frame)
    origFrameBin = image_bin(origFrameGray)
    a1, b1 = line(lines)
    a2, b2 = line1(lines1)
    origFrameBin = erode(dilate(origFrameBin))
    a, b, c = select_roi(frame, origFrameBin, a1, b1, a2, b2)
    
    #suma = SUM(b, c, d)
    
    suma01 += c
    
   # print(d)
    
    
    

    cv2.imshow("frame", frame)
    cv2.imshow("edges1", edges1)
    cv2.imshow("edges", edges)
 
 
    key = cv2.waitKey(25)
    if key == 27:
        break

print(suma01)

video2 = cv2.VideoCapture("video/video-2.avi")
suma02 = 0
while True:
    ret, orig_frame = video2.read()
    
    
    if not ret:
        video2 = cv2.VideoCapture("video/video-2.avi")
        break
 
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    low_blue = np.array([0, 0, 100])
    up_blue= np.array([20, 20, 255])
    low_green = np.array([0, 180, 0])
    up_green = np.array([20, 255, 20])
    mask1 = cv2.inRange(rgb, low_green, up_green)
    mask = cv2.inRange(rgb, low_blue, up_blue)
    edges = cv2.Canny(mask, 75, 150)
    edges1 = cv2.Canny(mask1, 75, 150)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
    lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
   
    
    origFrameGray = image_gray(frame)
    origFrameBin = image_bin(origFrameGray)
    a1, b1 = line(lines)
    a2, b2 = line1(lines1)
    origFrameBin = erode(dilate(origFrameBin))
    a, b, c = select_roi(frame, origFrameBin, a1, b1, a2, b2)
    
    #suma = SUM(b, c, d)
    
    suma02 += c
    
   # print(d)
    
    
    

    cv2.imshow("frame", frame)
    cv2.imshow("edges1", edges1)
    cv2.imshow("edges", edges)
 
 
    key = cv2.waitKey(25)
    if key == 27:
        break

print(suma02)

video3 = cv2.VideoCapture("video/video-3.avi")
suma03 = 0
while True:
    ret, orig_frame = video3.read()
    
    
    if not ret:
        video3 = cv2.VideoCapture("video/video-3.avi")
        break
 
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    low_blue = np.array([0, 0, 100])
    up_blue= np.array([20, 20, 255])
    low_green = np.array([0, 180, 0])
    up_green = np.array([20, 255, 20])
    mask1 = cv2.inRange(rgb, low_green, up_green)
    mask = cv2.inRange(rgb, low_blue, up_blue)
    edges = cv2.Canny(mask, 75, 150)
    edges1 = cv2.Canny(mask1, 75, 150)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
    lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
   
    
    origFrameGray = image_gray(frame)
    origFrameBin = image_bin(origFrameGray)
    a1, b1 = line(lines)
    a2, b2 = line1(lines1)
    origFrameBin = erode(dilate(origFrameBin))
    a, b, c = select_roi(frame, origFrameBin, a1, b1, a2, b2)
    
    #suma = SUM(b, c, d)
    
    suma03 += c
    
   # print(d)
    
    
    

    cv2.imshow("frame", frame)
    cv2.imshow("edges1", edges1)
    cv2.imshow("edges", edges)
 
 
    key = cv2.waitKey(25)
    if key == 27:
        break

print(suma03)

video4 = cv2.VideoCapture("video/video-4.avi")
suma04 = 0
while True:
    ret, orig_frame = video4.read()
    
    
    if not ret:
        video4 = cv2.VideoCapture("video/video-4.avi")
        break
 
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    low_blue = np.array([0, 0, 100])
    up_blue= np.array([20, 20, 255])
    low_green = np.array([0, 180, 0])
    up_green = np.array([20, 255, 20])
    mask1 = cv2.inRange(rgb, low_green, up_green)
    mask = cv2.inRange(rgb, low_blue, up_blue)
    edges = cv2.Canny(mask, 75, 150)
    edges1 = cv2.Canny(mask1, 75, 150)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
    lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
   
    
    origFrameGray = image_gray(frame)
    origFrameBin = image_bin(origFrameGray)
    a1, b1 = line(lines)
    a2, b2 = line1(lines1)
    origFrameBin = erode(dilate(origFrameBin))
    a, b, c = select_roi(frame, origFrameBin, a1, b1, a2, b2)
    
    #suma = SUM(b, c, d)
    
    suma04 += c
    
   # print(d)
    
    
    

    cv2.imshow("frame", frame)
    cv2.imshow("edges1", edges1)
    cv2.imshow("edges", edges)
 
 
    key = cv2.waitKey(25)
    if key == 27:
        break

print(suma04)

video5 = cv2.VideoCapture("video/video-5.avi")
suma05 = 0
while True:
    ret, orig_frame = video5.read()
    
    
    if not ret:
        video5 = cv2.VideoCapture("video/video-5.avi")
        break
 
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    low_blue = np.array([0, 0, 100])
    up_blue= np.array([20, 20, 255])
    low_green = np.array([0, 180, 0])
    up_green = np.array([20, 255, 20])
    mask1 = cv2.inRange(rgb, low_green, up_green)
    mask = cv2.inRange(rgb, low_blue, up_blue)
    edges = cv2.Canny(mask, 75, 150)
    edges1 = cv2.Canny(mask1, 75, 150)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
    lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
   
    
    origFrameGray = image_gray(frame)
    origFrameBin = image_bin(origFrameGray)
    a1, b1 = line(lines)
    a2, b2 = line1(lines1)
    origFrameBin = erode(dilate(origFrameBin))
    a, b, c = select_roi(frame, origFrameBin, a1, b1, a2, b2)
    
    #suma = SUM(b, c, d)
    
    suma05 += c
    
   # print(d)
    
    
    

    cv2.imshow("frame", frame)
    cv2.imshow("edges1", edges1)
    cv2.imshow("edges", edges)
 
 
    key = cv2.waitKey(25)
    if key == 27:
        break

print(suma05)

video6 = cv2.VideoCapture("video/video-6.avi")
suma06 = 0
while True:
    ret, orig_frame = video6.read()
    
    
    if not ret:
        video6 = cv2.VideoCapture("video/video-6.avi")
        break
 
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    low_blue = np.array([0, 0, 100])
    up_blue= np.array([20, 20, 255])
    low_green = np.array([0, 180, 0])
    up_green = np.array([20, 255, 20])
    mask1 = cv2.inRange(rgb, low_green, up_green)
    mask = cv2.inRange(rgb, low_blue, up_blue)
    edges = cv2.Canny(mask, 75, 150)
    edges1 = cv2.Canny(mask1, 75, 150)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
    lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
   
    
    origFrameGray = image_gray(frame)
    origFrameBin = image_bin(origFrameGray)
    a1, b1 = line(lines)
    a2, b2 = line1(lines1)
    origFrameBin = erode(dilate(origFrameBin))
    a, b, c = select_roi(frame, origFrameBin, a1, b1, a2, b2)
    
    #suma = SUM(b, c, d)
    
    suma06 += c
    
   # print(d)
    
    
    

    cv2.imshow("frame", frame)
    cv2.imshow("edges1", edges1)
    cv2.imshow("edges", edges)
 
 
    key = cv2.waitKey(25)
    if key == 27:
        break

print(suma06)

video7 = cv2.VideoCapture("video/video-7.avi")
suma07 = 0
while True:
    ret, orig_frame = video7.read()
    
    
    if not ret:
        video7 = cv2.VideoCapture("video/video-7.avi")
        break
 
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    low_blue = np.array([0, 0, 100])
    up_blue= np.array([20, 20, 255])
    low_green = np.array([0, 180, 0])
    up_green = np.array([20, 255, 20])
    mask1 = cv2.inRange(rgb, low_green, up_green)
    mask = cv2.inRange(rgb, low_blue, up_blue)
    edges = cv2.Canny(mask, 75, 150)
    edges1 = cv2.Canny(mask1, 75, 150)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
    lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
   
    
    origFrameGray = image_gray(frame)
    origFrameBin = image_bin(origFrameGray)
    a1, b1 = line(lines)
    a2, b2 = line1(lines1)
    origFrameBin = erode(dilate(origFrameBin))
    a, b, c = select_roi(frame, origFrameBin, a1, b1, a2, b2)
    
    #suma = SUM(b, c, d)
    
    suma07 += c
    
   # print(d)
    
    
    

    cv2.imshow("frame", frame)
    cv2.imshow("edges1", edges1)
    cv2.imshow("edges", edges)
 
 
    key = cv2.waitKey(25)
    if key == 27:
        break

print(suma07)

video8 = cv2.VideoCapture("video/video-8.avi")
suma08 = 0
while True:
    ret, orig_frame = video8.read()
    
    
    if not ret:
        video8 = cv2.VideoCapture("video/video-8.avi")
        break
 
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    low_blue = np.array([0, 0, 100])
    up_blue= np.array([20, 20, 255])
    low_green = np.array([0, 180, 0])
    up_green = np.array([20, 255, 20])
    mask1 = cv2.inRange(rgb, low_green, up_green)
    mask = cv2.inRange(rgb, low_blue, up_blue)
    edges = cv2.Canny(mask, 75, 150)
    edges1 = cv2.Canny(mask1, 75, 150)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
    lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
   
    
    origFrameGray = image_gray(frame)
    origFrameBin = image_bin(origFrameGray)
    a1, b1 = line(lines)
    a2, b2 = line1(lines1)
    origFrameBin = erode(dilate(origFrameBin))
    a, b, c = select_roi(frame, origFrameBin, a1, b1, a2, b2)
    
    #suma = SUM(b, c, d)
    
    suma08 += c
    
   # print(d)
    
    
    

    cv2.imshow("frame", frame)
    cv2.imshow("edges1", edges1)
    cv2.imshow("edges", edges)
 
 
    key = cv2.waitKey(25)
    if key == 27:
        break

print(suma08)

video9 = cv2.VideoCapture("video/video-9.avi")
suma09 = 0
while True:
    ret, orig_frame = video9.read()
    
    
    if not ret:
        video9 = cv2.VideoCapture("video/video-9.avi")
        break
 
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    low_blue = np.array([0, 0, 100])
    up_blue= np.array([20, 20, 255])
    low_green = np.array([0, 180, 0])
    up_green = np.array([20, 255, 20])
    mask1 = cv2.inRange(rgb, low_green, up_green)
    mask = cv2.inRange(rgb, low_blue, up_blue)
    edges = cv2.Canny(mask, 75, 150)
    edges1 = cv2.Canny(mask1, 75, 150)
 
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
    lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 100, maxLineGap=40, minLineLength = 100)
   
    
    origFrameGray = image_gray(frame)
    origFrameBin = image_bin(origFrameGray)
    a1, b1 = line(lines)
    a2, b2 = line1(lines1)
    origFrameBin = erode(dilate(origFrameBin))
    a, b, c = select_roi(frame, origFrameBin, a1, b1, a2, b2)
    
    #suma = SUM(b, c, d)
    
    suma09 += c
    
   # print(d)
    
    
    

    cv2.imshow("frame", frame)
    cv2.imshow("edges1", edges1)
    cv2.imshow("edges", edges)
 
 
    key = cv2.waitKey(25)
    if key == 27:
        break

print(suma09)


video.release()
cv2.destroyAllWindows()

file = open("out.txt","w")
file.write("RA 236/2015 Nemanja Vujovic\r")
file.write("file	sum\r")
file.write('video-0.avi\t' + str(suma0) +'\r')
file.write('video-1.avi\t' + str(suma01) +'\r')
file.write('video-2.avi\t' + str(suma02) +'\r')
file.write('video-3.avi\t' + str(suma03) +'\r')
file.write('video-4.avi\t' + str(suma04) +'\r')
file.write('video-5.avi\t' + str(suma05) +'\r')
file.write('video-6.avi\t' + str(suma06) +'\r')
file.write('video-7.avi\t' + str(suma07) +'\r')
file.write('video-8.avi\t' + str(suma08) +'\r')
file.write('video-9.avi\t' + str(suma09) +'\r')
file.close()