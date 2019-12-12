

import numpy as np

melodie = np.array([0,0,4,2])
notes = ['A','B','C','D','E','F','G','H']
loss=np.array([])
def noteToNumber(note):
  return notes.index(note)
def NumberToNote(number):
  return notes[number]
# X = (note 1, note 2, note 3, note 4), y = note 5
xAllNotes = np.array(['A' ,'E' ,'C' ,'B' ,'C' ,'B','B','A','A' ,'G' ,'A' ,'B' ,'C' ,'A' ,'B' ,'C' ,'D' ,'C' ,'B' ,'A' ,'C' ,'B' ,'C', 'A' ,'B' ,'C' ,'B'  ,'C' ,'D' ,'C' ,'G' ,'A' ,'F' ,'G' ,'A' ,'B','A' ,'C' ,'B' ,'C' ,'D', 'F' ,'G' ,'A' ,'G' ,'D','C' ,'B' ,'C','C' ,'F' ,'A' ,'B' ,'C' ,'D' ,'A' ,'C' ,'B' ,'C' ,'D' ,'C' ,'F' ,'G' ,'A' ,'G' ,'D' ,'C' ,'B' ,'C' ,'A','B' ,'B','C'])
xAllDuration = np.array([2 , 2 , 2 , 2 , 2 , 4 , 2 , 1 , 1 , 4 , 2 , 2 , 6 , 2 , 2 , 2 , 2 , 2 , 4 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 2 , 6 , 2 , 3 , 1 , 2 , 1 , 1 , 4 , 2 , 2 , 6 , 2 , 6 , 2 , 2 , 3 , 1 , 2 , 2 , 4 , 2 , 1 , 1 , 4 , 4 , 4 , 4 , 3 , 1 , 2 , 1 , 1 , 4 , 2 , 1 , 1 , 2 , 2 , 4 , 4 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 4 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 1 , 1 , 2 , 2 , 2 , 2 , 2 , 4 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 4 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 4 , 2 , 2 , 2 , 2 , 2 , 2 , 4 , 2 , 1 , 1 , 4 , 2 , 2 , 4 , 2 , 2 , 2 , 2 , 2 , 2 , 4 , 4 , 4])

sortieSimplifie = np.array([[100,1,1,1,1,1,1,1],[1,100,1,1,1,1,1,1], [1,1,100,1,1,1,1,1],[1,1,1,100,1,1,1,1],[1,1,1,1,100,1,1,1],[1,1,1,1,1,100,1,1],[1,1,1,1,1,1,100,1],[1,1,1,1,1,1,1,100]])

# X = (four notes example of melody), y = the fihft note expected

#CREATION DE LINPUT
xAll=np.array([noteToNumber(note) for note in xAllNotes])
#print(xAll)
#print(xAll.shape)
X=[]
y=[]
for it in range(len(xAll)-1): # trains the NN len(Xall) times
  X = np.append(X,sortieSimplifie[int(xAll[it])])
X = X.reshape(24,24) #
#CREATION DE LOUTPUT POUR LENTRAINEMENT
i=0
while i<len(xAll)-1:
  y = np.append(y,sortieSimplifie[int(xAll[i+1])])
  i=i+3
y = y.reshape(24,8)
#print(X)
#print(y)
#print(X.shape)
#print(y.shape)
xAll=X

# scale units
xAll = xAll/np.amax(xAll, axis=0) # scaling input data
y = y/100 # scaling output data (max test score is 100)

# split data
X = np.split(xAll, [24])[0] # training data
xPredicted = np.split(xAll, [24])[1] # testing data

class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 24 #2 
    self.outputSize =  8 #1
    self.hiddenSize =  22  #3

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train(self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self,xPredicted):
    #print ("taille xpredicted")
    #print (xPredicted.shape)
    #print ("Input (scaled): \n" + str(xPredicted))
    xPredictedTD = np.append(sortieSimplifie[int(xPredicted[0])],sortieSimplifie[int(xPredicted[1])])
    xPredictedTD=np.append(xPredictedTD,sortieSimplifie[int(xPredicted[2])])
    xPredictedTD=xPredictedTD/100
    rep = self.forward(xPredictedTD)
    #print (rep)
    print ("nouvelle Note choisi par le reseauxN: ")
    posNote = np.argmax(rep)
    #print(notes[posNote])
    #print ("Nouvelle note compléte : ")
    #print(notes[posNote])
    print (posNote)
    return posNote

NN = Neural_Network()
for i in range(1000): # trains the NN 1,000 times
  print ("# " + str(i) + "\n")
  #print ("Input (scaled): \n" + str(X))
  #print ("Actual Output: \n" + str(y))
  #print ("Predicted Output: \n" + str(NN.forward(X)))
  print ("Loss: \n" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
  print ("\n")
  NN.train(X, y)
i=0
for i in range(30):
  xPredicted=melodie[i:i+3]
  NoteToAdd = NN.predict(xPredicted)
  melodie = np.append(melodie,NoteToAdd)
print('la mélodie Finale :')
print(melodie)
print('la mélodie Finale :')
melodieLettre = np.array([NumberToNote(number) for number in melodie])
#here is the final melody :
print(melodieLettre)

NN.saveWeights()



