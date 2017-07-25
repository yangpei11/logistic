from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
	dataMat = []; labelMat = []
	f = open('testSet.txt')
	for line in f.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([1, float(lineArr[0]), float(lineArr[1]) ])
		labelMat.append( int(lineArr[2]) )
	return dataMat, labelMat
	#z = w0 + w1*x1 + w2*x2
def sigmoid(inX):
	return 1.0/(1+exp(-inX))

def gradAscent(dataMat, labelMdat):
	dataMat = mat(dataMat)
	labelMat = mat(labelMat).transpose()
	alpha = 0.001
	maxCycles = 500
	m, n = shape(dataMat)
	w = ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMat * w)
		error = labelMat - h
		w = w + alpha*dataMat.transpose()*error
	return w

def plotBestFit(method):
	dataMat, labelMat = loadDataSet()
	dataMat = array(dataMat)
	if method == 'gradAscent':
		w = gradAscent(dataMat, labelMat)
		w = array(w).ravel()
	elif method == 'stocGradAscent0':
		w = stocGradAscent0(dataMat, labelMat)
	else:
		w = stocGradAscent1(dataMat, labelMat)
	n = shape(dataMat)[0]
	xcord1 = []; ycord1 = []
	xcord2 = []; ycord2 = []
	for i in range(n):
		if labelMat[i] == 1:
			xcord1.append(dataMat[i, 1]); ycord1.append(dataMat[i, 2])
		else:
			xcord2.append(dataMat[i, 1]); ycord2.append(dataMat[i, 2])
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xcord1, ycord1, s=30, c = 'red', marker = 's')
	ax.scatter(xcord2, ycord2, s = 30, c = 'green')
	x = arange(-3.0, 3.0, 0.1)
	y = (-w[0]-w[1]*x)/w[2]
	plt.plot(x,y)
	plt.xlabel('X1'); plt.ylabel('X2')
	plt.show()

def stocGradAscent0(dataMatrix, classLabels):
	m, n = shape(dataMatrix)
	alpha = 0.01
	w = ones(n)
	for i in range(m):
		h = sigmoid( sum(dataMatrix[i]*w) )
		error = classLabels[i] - h
		w = w + error*alpha*dataMatrix[i]
	return w

def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
	m, n = shape(dataMatrix)
	w = ones(n)
	for j in range(numIter):
		dataIndex = range(m)
		for i in range(m):
			alpha = 4/(1+i+j) + 0.01
			randIndex = int( random.uniform(0, len(dataIndex) ) )
			h = sigmoid(sum(dataMatrix[randIndex]*w) )
			error = classLabels[randIndex] - h
			w = w + alpha * error * dataMatrix[randIndex]
			del( dataIndex[randIndex] )
	return w

def classifyVector(inX, w):
	h = sigmoid( sum(inX*w) )
	if h > 0.5:
		return 1
	else:
		return 0

def colicTest():
	frTrain  = open('horseColicTraining.txt')
	frTest = open('horseColicTest.txt')
	trainingSet = []; trainingLabels = []
	for line in frTrain.readlines():
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append( float(currLine[i]) )
		trainingSet.append(lineArr)
		trainingLabels.append( float(currLine[21]) )
	w = stocGradAscent1( array(trainingSet), trainingLabels, 500)
	errorCount = 0; numTestVec = 0.0
	for line in frTest.readlines():
		numTestVec += 1.0
		currLine = line.strip().split('\t')
		lineArr = []
		for i in range(21):
			lineArr.append( float(currLine[i]) )
		if int( classifyVector(array(lineArr), w) ) != int(currLine[21]):
			errorCount += 1
	errorRate = float(errorCount)/numTestVec
	print "the error rate of this test is: %f" %errorRate
	return errorRate

def multiTest():
	numTests = 10; errorSum = 0.0
	for k in range(numTests):
		errorSum += colicTest()
	print "after %d iterations the average error rate is: %f" %(numTests, errorSum/float(numTests))


multiTest()






