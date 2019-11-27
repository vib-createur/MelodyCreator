
import numpy as np

melodie = np.array([0,3,4,0])
notes = ['A','B','C','D','E','F','G','H']

def noteToNumber(note):
  return notes.index(note)
def NumberToNote(number):
  return notes[number]
# X = (note 1, note 2, note 3, note 4), y = note 5
xAllNotes = np.array(['A' ,'E' ,'C' ,'B' ,'C' ,'B' ,'B' ,'A' ,'G' ,'A' ,'A' ,'B' ,'C' ,'A' ,'B' ,'B' ,'C' ,'D' ,'C' ,'C' ,'C' ,'C' ,'C' ,'B' ,'B' ,'A' ,'A' ,'A' ,'A' ,'C' ,'C' ,'C' ,'C' ,'C' ,'B' ,'B' ,'B' ,'C' ,'C' ,'C' ,'C' ,'C' ,'A' ,'A' ,'B' ,'C' ,'B' ,'B' ,'C' ,'D' ,'C' ,'G' ,'A' ,'A' ,'F' ,'F' ,'F' ,'G' ,'A' ,'B' ,'B' ,'B' ,'B' ,'B' ,'B' ,'B' ,'A' ,'A' ,'A' ,'C' ,'B' ,'B' ,'C' ,'D' ,'C' ,'C' ,'C' ,'F' ,'G' ,'A' ,'G' ,'G' ,'G' ,'G' ,'G' ,'G' ,'D' ,'D' ,'C' ,'B' ,'C' ,'C' ,'C' ,'F' ,'A' ,'B' ,'C' ,'D' ,'D' ,'D' ,'D' ,'A' ,'C' ,'B' ,'B' ,'C' ,'D' ,'C' ,'C' ,'C' ,'F' ,'G' ,'A' ,'G' ,'G' ,'G' ,'G' ,'G' ,'D' ,'C' ,'B' ,'C' ,'C' ,'C' ,'A' ,'A' ,'A' ,'B' ,'C','C'])
xAllDuration = np.array([2 , 2 , 2 , 2 , 2 , 4 , 2 , 1 , 1 , 4 , 2 , 2 , 6 , 2 , 2 , 2 , 2 , 2 , 4 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 2 , 6 , 2 , 3 , 1 , 2 , 1 , 1 , 4 , 2 , 2 , 6 , 2 , 6 , 2 , 2 , 3 , 1 , 2 , 2 , 4 , 2 , 1 , 1 , 4 , 4 , 4 , 4 , 3 , 1 , 2 , 1 , 1 , 4 , 2 , 1 , 1 , 2 , 2 , 4 , 4 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 4 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 1 , 1 , 2 , 2 , 2 , 2 , 2 , 4 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 4 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 4 , 2 , 2 , 2 , 2 , 2 , 2 , 4 , 2 , 1 , 1 , 4 , 2 , 2 , 4 , 2 , 2 , 2 , 2 , 2 , 2 , 4 , 4 , 4])

xAll=np.array([noteToNumber(note) for note in xAllNotes])

sortieSimplifie = np.array([[100,1,1,1,1,1,1,1],[1,100,1,1,1,1,1,1], [1,1,100,1,1,1,1,1],[1,1,1,100,1,1,1,1],[1,1,1,1,100,1,1,1],[1,1,1,1,1,100,1,1],[1,1,1,1,1,1,100,1],[1,1,1,1,1,1,1,100]])

#y = np.array((sortieSimplifié[4],[1,1,1,1,1,1,1,100], [1,1,1,1,1,100,1,1],[1,1,1,1,1,1,100,1],[1,100,1,1,1,1,1,1],[100,1,1,1,1,1,1,1],[100,1,1,1,1,100,1,1]), dtype=float) # output
# scale units
#xAll = xAll/np.amax(xAll, axis=0) # scaling input data par rapport aux plus grand x ou y ici 5 ou 10
#y = y/100 # scaling output data (max test score is 100)

# split data
#X = np.split(xAll, [xAll.len])[0] # training data

# xPredicted = np.split(xAll, [7])[1] # testing data

class Neural_Network(object):
  def __init__(self):
  #parameters
    self.inputSize = 4
    self.outputSize = 8
    self.hiddenSize = 8

  #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize) # (3x1) weight matrix from hidden to output layer
    self.W3= np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, X):
    #forward propagation through our network
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    self.z4 = self.sigmoid(self.z3)
    self.z5 = np.dot(self.z4, self.W3)
    o = self.sigmoid(self.z5) # final activation function
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
    self.z4_error = self.z2_delta.dot(self.W3.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z4_delta = self.z4_error*self.sigmoidPrime(self.z4) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.z4_delta) # adjusting second set (hidden -->hidden weights
    self.W3 += self.z4.T.dot(self.o_delta) # adjusting third set (hidden --> output) weights

  def train(self, X, y):
      print('on va printer X et Y puis o')
      print(X)
      print(y)
      o = self.forward(X)
      print(o)
      self.backward(X, y, o)

  def saveWeights(self):
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")
    np.savetxt("w3.txt", self.W3, fmt="%s")


  def predict(self):
    i = len(melodie) - 4
    xPredicted=melodie[i:i+4]
    print ("Predicted data based on trained weights: ")
    print ("Input (scaled): \n" + str(xPredicted))
    rep = self.forward(xPredicted)
    print (rep)
    print ("nouvelle Note choisi par le reseauxN: ")
    posNote = np.argmax(rep)
    print(notes[posNote])
    print ("Nouvelle note compléte : ")
    print(notes[posNote])
    return notes[posNote]
  

   # print ("Rep algo: \n" + str(self.forward(xPredicted)))

NN = Neural_Network()
for i in range(2): # trains the NN 1,000 times
  print ("# " + str(i) + "\n")
  print ("Input (scaled): \n" + str(xAll[0]))
  print ("Actual Output: \n" + str(sortieSimplifie[int(xAll[1])]))
  print ("Predicted Output: \n" + str(NN.forward(xAll[0])))
  print ("Loss: \n" + str(np.mean(np.square(sortieSimplifie[int(xAll[5]+1)] - NN.forward(xAll[0]))))) # mean sum squared loss
  print ("\n")
  X=np.array([])
  y=np.array([])

  for it in range(len(xAll)-5):
    print('a verifier i :')
    print(it)
    print('a verifier xAll[it:it+4] :')
    print(xAll[it:it+4])
    print('a verifier y :')
    X = xAll[it:it+4]/7
    y = (sortieSimplifie[int(xAll[it+5]+1)])/100
    print(y)
    NN.train(X, y)

NN.saveWeights()

for i in range(30):
  NoteToAdd = NN.predict()
  melodie = np.append(NoteToAdd,melodie)
print('la mélodie Finale :')
print(melodie)
print('la mélodie Finale :')
melodieLettre = np.array([NumberToNote(number) for number in melodie])
print(melodieLettre)
