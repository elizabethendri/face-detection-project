import numpy
from scipy.spatial import distance
from sklearn import neighbors,preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix


# Folders 1-5 are Male and 6-10 are Women

sample_size = 10
feature_num = 7
point_num = 22

# Array of all training samples
participants = [1,2,3,4,5,6,7,8,9,10]

trainData = numpy.zeros((sample_size, feature_num))
testData = numpy.zeros((sample_size, feature_num))

# Created For Loop to iterate through training images (every other file)

for x in range(sample_size):


    file = open("./FaceDatabase/" + str(x+1) + "/1.pts", "r")
    point = file.readlines()
    point = point[3:25]

    for p in range(point_num):
        point[p] = point[p].rstrip().split()
        point[p][0] = float(point[p][0])
        point[p][1] = float(point[p][1])

    #every point has an index = its number

    # Feature 1 -- Eye length Ration: Length of Eye (Max of 2)
    trainData[x][0] = max(distance.euclidean(point[9], point[10]), distance.euclidean(point[11], point[12]))/distance.euclidean(point[8], point[13])

    #Feature 2 - Eye Distance Ratio (Distance between center of two eyes)
    trainData[x][1] = distance.euclidean(point[0], point[1]) / distance.euclidean(point[8], point[13])

    #Nose ratio -- Feature 3
    trainData[x][2] = distance.euclidean(point[15], point[16])/distance.euclidean(point[20], point[21])

    #Lip size ratio --4
    trainData[x][3] = distance.euclidean(point[2], point[3]) / distance.euclidean(point[17], point[18])

    # Lip length Ratio --5
    trainData[x][4] = distance.euclidean(point[2], point[3]) / distance.euclidean(point[20], point[21])

    # Eye-brow length ratio  -- 6
    trainData[x][5] = max(distance.euclidean(point[4], point[5]),distance.euclidean(point[6], point[7])) / distance.euclidean(point[8], point[13])

    # Aggressive Ratio -- 7
    trainData[x][6] = distance.euclidean(point[10], point[19]) / distance.euclidean(point[20], point[21])



#for loop for testing
for x in range(sample_size):

    file = open("./FaceDatabase/" + str(x+1) + "/2.pts", "r")
    point = file.readlines()
    point = point[3:25]

    for p in range(point_num):
        point[p] = point[p].rstrip().split()
        point[p][0] = float(point[p][0])
        point[p][1] = float(point[p][1])

    #print(lines)


    # Feature 1 -- Eye length Ration: Length of Eye (Max of 2)
    testData[x][0] = max(distance.euclidean(point[9], point[10]),distance.euclidean(point[11], point[12])) / distance.euclidean(point[8], point[13])

    # Feature 2 - Eye Distance Ratio (Distance between center of two eyes)
    testData[x][1] = distance.euclidean(point[0], point[1]) / distance.euclidean(point[8], point[13])

    # Nose ratio -- 3
    testData[x][2] = distance.euclidean(point[15], point[16]) / distance.euclidean(point[20], point[21])

    # Lip size Ratio -- 4
    testData[x][3] = distance.euclidean(point[2], point[3]) / distance.euclidean(point[17], point[18])

    # Lip length Ratio -- 5
    testData[x][4] = distance.euclidean(point[2], point[3]) / distance.euclidean(point[20], point[21])

    # Eye-brow length Ratio  -- 6
    testData[x][5] = max(distance.euclidean(point[4], point[5]), distance.euclidean(point[6], point[7])) / distance.euclidean(point[8], point[13])

    # Aggressive Ratio -- 7
    testData[x][6] = distance.euclidean(point[10], point[19]) / distance.euclidean(point[20], point[21])



nn = neighbors.KNeighborsClassifier(n_neighbors = int(1))
nn.fit(trainData,participants)

decisions = nn.predict(testData)

le = preprocessing.LabelEncoder()
le.fit(["?", "m1", "m2", "m3", "m4", "m5", "w1", "w2", "w3", "w4", "w5"])

results = le.inverse_transform(decisions)

print(results)

print(confusion_matrix(participants, decisions))