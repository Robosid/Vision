######################################### 
#Author : Sidney						#
# Last Modified : March 7 0400			#
# Purpose: KU Leuven CV Project 1		#
# Sidney's Quote: "I'm a fucking genius"# 
#                 "Deal with it"		#
# Quote on Sidney:"He sucks at bed"		#
#                 "Deal with it"		#
#########################################

import cv2
import numpy as np

#---------------Wrappers-------------------------------------------

def houghCT(frame_t, md, p1, p2, mnr, mxr):

	gray = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY) 
	gray_blurred = cv2.medianBlur(gray, 5) 
	#rows = gray_blurred.shape[0]
	# Apply Hough transform on the blurred image. 
	detected_circles = cv2.HoughCircles(gray_blurred,
    	cv2.HOUGH_GRADIENT, 1, md, param1 = p1, 
    	param2 = p2, minRadius = mnr, maxRadius = mxr) 

	if detected_circles is not None: 
		detected_circles = np.uint16(np.around(detected_circles)) 
		for pt in detected_circles[0, :]: 
			a, b, r = pt[0], pt[1], pt[2]
			cv2.circle(frame_t, (a, b), r, (0, 255, 0), 2) 
			#cv2.circle(frame_t, (a, b), 2, (0, 0, 255), 3)

	return frame_t 

def sobel(frame_o, size_k, scale_o, delta_o):

	src = cv2.GaussianBlur(frame_o, (3, 3), 0)
	gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	gray = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)

	'''
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5) #Horizontal Sobel
	sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5) #Vertical Sobel
	'''
	'''
	src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
	src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
	src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
	src.depth() = CV_64F, ddepth = -1/CV_64F
	KSIZE = 1 or 3 or 5 or 7
	'''

	grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=size_k, scale=scale_o, delta=delta_o, borderType=cv2.BORDER_DEFAULT) #CV_16S to avoid overflow.
	grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=size_k, scale=scale_o, delta=delta_o, borderType=cv2.BORDER_DEFAULT)
	
	abs_grad_x = cv2.convertScaleAbs(grad_x)
	abs_grad_y = cv2.convertScaleAbs(grad_y)
	grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

	return grad
#------------------------------------------------------------------------

#----------------Begin---------------------------------------------------
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('/home/Robosid/cv1/trial.mp4')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
# Define the fps. Also frame size is passed.

out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS), (frame_width,frame_height))

print(cap.get(cv2.CAP_PROP_FPS))

count = 0
v = 1
val = 75
fgbg2 = cv2.createBackgroundSubtractorKNN()
fgbg = cv2.createBackgroundSubtractorMOG2()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#------------------Trackbars for HSV---------------------------------
def nothing(x):
    pass
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
#--------------------------------------------------------------------

# Read until video is completed
while(cap.isOpened()):

  # Capture frame-by-frame
  ret, frame = cap.read()
  #frame = cv2.flip(frame, -1)
  retval = cap.get(cv2.CAP_PROP_POS_MSEC)
  #print(retval)
  grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  grayframe = cv2.cvtColor(grayFrame, cv2.COLOR_GRAY2BGR)
  
  if ret == True:

    if retval < 1000 or (retval >= 3000 and retval < 5000):
    	# Display the resulting frame
    	cv2.imshow('Frame',grayFrame)
    	out.write(grayframe)
    	# Press Q on keyboard to exit
    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    elif (retval >= 1000 and retval < 3000):
    	# Display the resulting frame
    	cv2.imshow('Frame',frame)
    	out.write(frame)
    	# Press Q on keyboard to  exit
    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break
    
    elif (retval >= 5000 and retval <12000):
    	
    	'''Sigma values: For simplicity, set the 2 sigma values to be the same. 
    	If they are small (< 10), the filter will not have much effect, whereas if they 
    	are large (more than 150), they will have a very strong effect, making the image look cartoonish.
    	sigmaColor : Filter sigma in the color space. A larger value of the parameter means
    	that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed
    	together, resulting in larger areas of semi-equal color.

    	sigmaSpace : Filter sigma in the coordinate space. A larger value of the parameter 
    	means that farther pixels will influence each other as long as their colors are close
    	enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of
    	sigmaSpace . Otherwise, d is proportional to sigmaSpace .

    	Filter size: Large filters (d > 5) are very slow, so it is recommended to use d=5 for 
    	real-time applications, and perhaps d=9 for offline applications that need heavy noise filtering.
    	'''
    	#gb = cv2.GaussianBlur(frame, (v, v), 0)
    	#blf = cv2.bilateralFilter(frame,9,val,val) #ALTER SIGMAS
    	#v = v + 2
    	#val = val + 2	
    	#image = cv2.resize(gb, (0, 0), None, .50, 1)
    	#image2 = cv2.resize(blf, (0, 0), None, .50, 1)
    	#horizontal = np.hstack((image, image2))
    	##horizontal = np.concatenate((image, image2), axis=1)
    	cv2.imshow('Frame',frame)
    	out.write(frame)
    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    elif (retval >= 12000 and retval <16000):

        #-----------HSV----SECTION----------------------
    	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    	    	
    	#-----------HSV Selector------------------------
    	l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    	l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    	l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    	u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    	u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    	u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    	#lower = np.array([l_h, l_s, l_v])
    	#upper = np.array([u_h, u_s, u_v])
    	#-----------------------------------------------

    	# define range of specific color in HSV after obtaining values from selector
    	lower = np.array([15,100,20])
    	upper = np.array([25,190,190])    	
    	
    	# Threshold the HSV image to get only specific color
    	mask = cv2.inRange(hsv, lower, upper)

    	# Bitwise-AND mask(HSV) and white image
    	img0 = cv2.imread('/home/Robosid/cv1/white.jpg')
    	img = cv2.resize(img0,(frame_width,frame_height))
    	res = cv2.bitwise_and(img,img, mask= mask)
        
        #-----------RGB --------SECTION------------------
    	lowerbgr = np.array([20, 40, 60])
    	upperbgr = np.array([80, 160, 200])
    	maskbgr = cv2.inRange(frame, lowerbgr, upperbgr)
    	resbgr = cv2.bitwise_and(img,img, mask= maskbgr)

    	#------------------RESIZE for HStack------------------------
    	res1 = cv2.resize(res, (0, 0), None, .50, 1)
    	res2 = cv2.resize(resbgr, (0, 0), None, .50, 1)
    	#img2 = cv2.resize(img, (0, 0), None, .x, 1)
    
    	horizontal2 = np.hstack((res1, res2))  #HStack [HSV | BGR]
    	#horizontal2 = np.concatenate((res1, res2), axis=1)
    	cv2.imshow('Frame', horizontal2)
    	out.write(horizontal2)

    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    elif (retval >= 16000 and retval <20000):

        #-----------HSV----SECTION----------------------
    	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    	    	
    	#-----------HSV Selector------------------------
    	l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    	l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    	l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    	u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    	u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    	u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    	#lower = np.array([l_h, l_s, l_v])
    	#upper = np.array([u_h, u_s, u_v])
    	#-----------------------------------------------

    	# define range of specific color in HSV after obtaining values from selector
    	lower = np.array([15,100,20])
    	upper = np.array([25,190,190])    	
    	
    	# Threshold the HSV image to get only specific color
    	mask = cv2.inRange(hsv, lower, upper)

    	# Bitwise-AND mask(HSV) and white image
    	img0 = cv2.imread('/home/Robosid/cv1/white.jpg')
    	img = cv2.resize(img0,(frame_width,frame_height))
    	res = cv2.bitwise_and(img,img, mask= mask)

    	#-------Erosion / Dilation / Opening / Closing--------------
    	kernel = np.ones((5,5),np.uint8)
    	erosion = cv2.erode(mask,kernel,iterations = 2)
    	dilation = cv2.dilate(mask,kernel,iterations = 1)
    	opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    	closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    	res0 = cv2.bitwise_and(img,img, mask= erosion) # mask= erosion / dilation / opening / closing

    	#------------------RESIZE for HStack------------------------
    	res1 = cv2.resize(res, (0, 0), None, .50, 1)
    	res2 = cv2.resize(res0, (0, 0), None, .50, 1)
    	#img2 = cv2.resize(img, (0, 0), None, .x, 1)
    
    	horizontal2 = np.hstack((res1, res2))  #HStack [HSV | Morpho]
    	#horizontal2 = np.concatenate((res1, res2), axis=1)
    	cv2.imshow('Frame', horizontal2)
    	out.write(horizontal2)

    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    #--------------------------Sobel Detection---------------------

    elif (retval >= 20000 and retval < 22000):

    	sob = sobel(frame, 3, 1, 0) # Kernal Size, Scale, Delta
    	cv2.imshow('Frame', sob)
    	out.write(sob)

    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    elif (retval >= 22000 and retval < 24000):

    	sob = sobel(frame, 5, 1, 0)
    	cv2.imshow('Frame', sob)
    	out.write(sob)

    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    elif (retval >= 24000 and retval < 26000):

    	sob = sobel(frame, 3, 4, 8)
    	cv2.imshow('Frame', sob)
    	out.write(sob)

    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

#-----------------Hough Circle Detection------------------------
    elif (retval >= 26000 and retval < 28000):

    	all_circle = houghCT(frame, 800, 200, 50, 1, 300)
    	cv2.imshow("Frame", all_circle)
    	out.write(all_circle)

    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    elif (retval >= 28000 and retval < 30000):

    	all_circle = houghCT(frame, 80, 200, 50, 130, 300)
    	cv2.imshow("Frame", all_circle)
    	out.write(all_circle)

    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    elif (retval >= 30000 and retval < 33000):

    	all_circle = houghCT(frame, 1, 200, 50, 130, 300)
    	cv2.imshow("Frame", all_circle)
    	out.write(all_circle)

    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    elif (retval >= 33000 and retval < 36000):

    	all_circle = houghCT(frame, 1, 200, 40, 100, 300)
    	cv2.imshow("Frame", all_circle)
    	out.write(all_circle)

    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    #-----------Template Matching-------------------------------
    elif(retval >= 36000 and retval <41000):
    	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	img2 = img.copy()
    	template1 = cv2.imread('disc.png')
    	template = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
    	#template.astype(np.uint8)
    	w, h = template.shape[::-1]
    	methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    	'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    	img = img2.copy()
    	method = eval(methods[3])
    	# Apply template Matching
    	res = cv2.matchTemplate(img,template,method)
    	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    	if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    		top_left = min_loc
    	else:
    		top_left = max_loc
    	bottom_right = (top_left[0] + w, top_left[1] + h)
    	cv2.rectangle(frame,top_left, bottom_right, 255, 2)
    	#main = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    	#flip = cv2.flip(frame, -1)
    	#flip2 = cv2.flip(main, -1)
    	cv2.imshow("Frame", frame)
    	out.write(frame)
    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    #-----------------CARTE BLANCHE--------------------------------

    elif(retval >= 41000 and retval <46000): # MOG
    	
    	fgmask = fgbg.apply(frame)
    	#fgmask = fgbg2.apply(frame)    	
    	fgmask_write = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    	cv2.imshow("Frame", fgmask)
    	out.write(fgmask_write)
    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    elif(retval >= 51000 and retval <56000): # MOG
    	
    	fgmask = fgbg.apply(frame)
    	#fgmask = fgbg2.apply(frame)    	
    	fgmask_write = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    	cv2.imshow("Frame", fgmask)
    	out.write(fgmask_write)
    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break    		

    elif(retval >= 56000 and retval <61000): #Cascade
    	
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    	for (x,y,w,h) in faces:
    		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    		roi_gray = gray[y:y+h, x:x+w]
    		roi_color = img[y:y+h, x:x+w]
    		eyes = eye_cascade.detectMultiScale(roi_gray)
    		for (ex,ey,ew,eh) in eyes:
    			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    	
    	cv2.imshow('Frame',frame)
    	out.write(frame)
    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    elif(retval >= 46000 and retval <51000): #Camshift
    	
    	if(count == 0):
    		x, y, w, h = 100, 100, 100, 50 
    		track_window = (x, y, w, h)
    		roi = frame[y:y+h, x:x+w]
    		hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    		mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    		roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    		cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    		term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1 )
    		count = 10
    	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    	dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    	# apply camshift to get the new location
    	ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    	# Draw it on image
    	pts = cv2.boxPoints(ret)
    	pts = np.int0(pts)
    	img2 = cv2.polylines(frame,[pts],True, 255,2)
    	cv2.imshow('Frame',img2)
    	out.write(img2)
    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break

    else:
    	cv2.imshow('Frame',frame)
    	out.write(frame)
    	if cv2.waitKey(25) & 0xFF == ord('q'):
    		break


  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
