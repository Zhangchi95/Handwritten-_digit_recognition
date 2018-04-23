import numpy as np
def ReadAndDivide():
    from mnist import MNIST
    mndata = MNIST('original_data')
    images, labels = mndata.load_training()
    images_1 = []
    images_2 = []
    images_3 = []
    labels_1 = []
    labels_2 = []
    labels_3 = []
    # Distribute the original data into three dataset evenly
    # First creat the serial number 
    serial_num1 = []
    serial_num2 = []
    serial_num3 = []
    for i in range (0,2000):
        serial_num1.append(i*3)
    for i in serial_num1:
        serial_num2.append(i+1)
    for i in serial_num1:
        serial_num3.append(i+2)
    for i in serial_num1:
        images_1.append(images[i])
        labels_1.append(labels[i])
    for i in serial_num2:
        images_2.append(images[i])
        labels_2.append(labels[i])
    for i in serial_num3:
        images_3.append(images[i])
        labels_3.append(labels[i])
    return images_1,labels_1,images_2,labels_2,images_3,labels_3
def SaveNewData():
    import csv
    images_1,labels_1,images_2,labels_2,images_3,labels_3 = ReadAndDivide()
    with open('new_data/images_1.csv','wb') as myfile:
        wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
        wr.writerow(images_1)
    with open('new_data/images_2.csv','wb') as myfile:
        wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
        wr.writerow(images_2)
    with open('new_data/images_3.csv','wb') as myfile:
        wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
        wr.writerow(images_3)
    with open('new_data/labels_1.csv','wb') as myfile:
        wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
        wr.writerow(labels_1)
    with open('new_data/labels_2.csv','wb') as myfile:
        wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
        wr.writerow(labels_1)
    with open('new_data/labels_3.csv','wb') as myfile:
        wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
        wr.writerow(labels_3)
def TurnToNP():
    images_1,labels_1,images_2,labels_2,images_3,labels_3 = ReadAndDivide()
    #print labels_1
    images_1 = np.array(images_1)
    images_2 = np.array(images_2)
    images_3 = np.array(images_3)
    labels_1 = np.array(labels_1)
    labels_2 = np.array(labels_2)
    labels_3 = np.array(labels_3)
   # print labels_1
    return images_1,labels_1,images_2,labels_2,images_3,labels_3
def OneVsOneSVM():
    # Training S1 Validating S2,S3
    from sklearn.multiclass import OneVsOneClassifier
    from sklearn.svm import LinearSVC
    images_1,labels_1,images_2,labels_2,images_3,labels_3 = TurnToNP()
    #print labels_2
    # Training SVM using X1 and Test X2
    outcome_2 = OneVsOneClassifier(LinearSVC(random_state=0)).fit(images_1, labels_1).predict(images_2)
    from sklearn.metrics import accuracy_score
    accuracy_2 = accuracy_score(labels_2,outcome_2)
    # Training SVM using X1 and Test X3
    outcome_3 = OneVsOneClassifier(LinearSVC(random_state=0)).fit(images_1, labels_1).predict(images_3)
    from sklearn.metrics import accuracy_score
    accuracy_3 = accuracy_score(labels_3,outcome_3)
    #print accuracy_2, accuracy_3
    return labels_1,outcome_2, outcome_3
def NeuralNetwork(): 
    # Training S2, Validating S1,S3
    import numpy as np
    import tensorflow as tf
    # import data
    val_x1,val_y1,train_x,train_y,val_x2,val_y2 = TurnToNP()
    #helper function
    def dense_to_one_hot(labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors"""
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def preproc(unclean_batch_x):
        """Convert values to range 0-1"""
        temp_batch = unclean_batch_x / unclean_batch_x.max()
        return temp_batch
    def batch_creator(batch_size, dataset_length, dataset_name):
        """Create batch with random samples and return appropriate format"""
        batch_mask = rng.choice(dataset_length, batch_size)
        batch_x = train_x[[batch_mask]].reshape(-1, input_num_units)
        batch_x = preproc(batch_x)
        batch_y = train_y[[batch_mask]].reshape(-1, 1)
        #if dataset_name == 'train':
        #    batch_y = train_x.ix[batch_mask, 'label'].values
        batch_y = dense_to_one_hot(batch_y)
        return batch_x, batch_y
    ### set all variables
    seed = 128
    rng = np.random.RandomState(seed)
    # number of neurons in each layer
    input_num_units = 28*28
    hidden_num_units = 500
    output_num_units = 10

    # define placeholders
    x = tf.placeholder(tf.float32, [None, input_num_units])
    y = tf.placeholder(tf.float32, [None, output_num_units])

    # set remaining variables
    #epochs = epochs      ## input study ranging around 5 - 20 
    #batch_size = batch_size   ## input  study ranging around 64 - 512
    #learning_rate = learning_rate ## input study ranging around 0.01 - 0.1
    
    ## Original setting 
    epochs = 35
    batch_size = 128
    learning_rate = 0.05

    ### define weights and biases of the neural network (refer this article if you don't understand the terminologies)

    weights = {
            'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
            'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
            }

    biases = {
            'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
            'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
            }
    hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)
    output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_layer, labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        # create initialized variables
        sess.run(init)
        ### for each epoch, do:
        ###   for each batch, do:
        ###     create pre-processed batch
        ###     run optimizer by feeding batch
        ###     find cost and reiterate to minimize
    
        for epoch in range(epochs):
            avg_cost = 0
            train_num = 2000 # Related to line 17 -- Range
            total_batch = int( train_num /batch_size) 
            for i in range(total_batch):
                batch_x, batch_y = batch_creator(batch_size, train_num, 'train')
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            
     #       print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)
    
    #    print "\nTraining complete!"
    # find predictions on val set
        pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        S1_accu = accuracy.eval({x: val_x1.reshape(-1, input_num_units), y: dense_to_one_hot(val_y1)})
        print "S1 Accuracy:", S1_accu
        pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
        S3_accu = accuracy.eval({x: val_x2.reshape(-1, input_num_units), y: dense_to_one_hot(val_y2)})
        print "S3 Accuracy:", S3_accu
        prediction = tf.argmax(output_layer, 1)
        outcome1 = prediction.eval(feed_dict = {x: val_x1},session = sess)
        outcome3 = prediction.eval(feed_dict = {x: val_x2},session = sess)
        return outcome1,train_y,outcome3
        #return pred_temp
def BinaryTreeCLF():
    from sklearn import tree
    # Training S3, Validating S1,S2
    images_1,labels_1,images_2,labels_2,images_3,labels_3 = TurnToNP()
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=64, max_features=784)
    clf = clf.fit(images_3,labels_3)
    S1_accu = clf.score(images_1,labels_1)
    S2_accu = clf.score(images_2,labels_2)
    outcome1 = clf.predict(images_1)
    outcome2 = clf.predict(images_2)
    #print S1_accu,S2_accu
    return outcome1,outcome2,labels_3
def MajorityVote():    
    #import data
    labels_1SVM,labels_2SVM,labels_3SVM = OneVsOneSVM()
    labels_1NN, labels_2NN, labels_3NN  = NeuralNetwork()
    labels_1BTC,labels_2BTC,labels_3BTC = BinaryTreeCLF()
    Labels_1 = []
    Labels_2 = []
    Labels_3 = []
    def Vote(SVM,NN,BTC):
        if (SVM == NN):
            return SVM
        elif (SVM == BTC):
            return SVM
        elif (NN == BTC):
            return NN
        else:
            return SVM # Since SVM has high general accurracy
    for i in range(0,2000): # Link to line 17
        Labels_1.append(Vote(labels_1SVM[i],labels_1NN[i],labels_1BTC[i]))
        Labels_2.append(Vote(labels_2SVM[i],labels_2NN[i],labels_2BTC[i]))
        Labels_3.append(Vote(labels_3SVM[i],labels_3NN[i],labels_3BTC[i]))
    return Labels_1,Labels_2,Labels_3
def SaveFinalData():
    import csv
    labels_1,labels_2,labels_3 = MajorityVote()
    with open('outcome/labels_1.csv','wb') as myfile:
        wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
        wr.writerow(labels_1)
    with open('outcome/labels_2.csv','wb') as myfile:
        wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
        wr.writerow(labels_2)
    with open('outcome/labels_3.csv','wb') as myfile:
        wr = csv.writer(myfile, quoting = csv.QUOTE_ALL)
        wr.writerow(labels_3)
def ShowPic():
    from PIL import Image
    images_1,labels_1,images_2,labels_2,images_3,labels_3 = ReadAndDivide()
    new_img = Image.new("L", (28, 28), "white")
    new_img.putdata(images_1[3])
    print labels_1[3]
    new_img.save('out.tif')
def Main():
    ReadAndDivide()
    SaveNewData()
    TurnToNP()
    NeuralNetwork()
    OneVsOneSVM()
    BinaryTreeCLF(max_depth)
    MajorityVote()
    SaveFinalData()
    #ShowPic()
Main()