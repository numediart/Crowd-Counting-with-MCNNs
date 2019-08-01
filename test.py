'''
Crowd-Counting using Multi-Column Convolutional Neural Networks.
'''

from network import MCNN
import cv2
#import sys
                
'''
# For accepting the input from the prompt.
if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    print('Usage: python3 test.py A(or B)')
    exit()
'''

dataset='A'    
type_input = 'image'

'''
# For the B dataset.
dataset = 'B'
'''                        
mcnn = MCNN(dataset)

if type_input == 'image':
	# image path
	img_path = '.\\data\\original\\shanghaitech\\part_'+ dataset +'_final\\test_data\\images\\5.jpg'
	img_path = 'C:\\Users\\sohai\\Documents\\codes\\paperswithcode\\CROWD\\Crowd-Counting-with-MCNNs\\7.jpg'
	# img_path = '.\\data\\original\\shanghaitech\\part_'+ dataset +'_final\\train_data\\images\\IMG_262.jpg'

	# For predicting the count of people in one Image.
	numoppl, den_sum = mcnn.predict(img_path)
	img = cv2.imread(img_path)
	cv2.putText(img, 'Prediction : ' + str(int(numoppl)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	# cv2.putText(img, 'Ground Tth : ' + str(int(den_sum)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
	cv2.putText(img, 'Ground Tth : ' + str(2197), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
	cv2.imshow('output', img)
	cv2.waitKey()

elif type_input == 'dataset':
	'''
	# For predicting the count of people in all Images in the Test Dataset.
	mcnn.test()
	'''
	mcnn.test()

elif type_input == 'video':
	# cap = cv2.VideoCapture(0)
	cap = cv2.VideoCapture('data/lille.mp4')
	while(cap.isOpened()):
		ret, frame = cap.read()
		cv2.imwrite('data/tmp.jpg', frame)
		numoppl, den_sum = mcnn.predict('data/tmp.jpg', isloaded=False)
		cv2.putText(frame, 'Prediction : ' + str(int(numoppl)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		cv2.imshow('output', frame)
		cv2.waitKey(1)
