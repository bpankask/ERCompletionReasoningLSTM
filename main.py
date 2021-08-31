import argparse
import math
import tensorflow.compat.v1 as tf
import os
from functools import partial
import json
import random
import numpy
import io
import numpy as np
from keras import models
from keras import layers

# Tested
def precision(TP, FP):
    """Calculates precision from number of true positives and false positives."""
    return 0 if TP == 0 and FP == 0 else TP / (TP + FP)


# Tested
def recall(TP, FN):
    """Calculates recall using number of true positive and false negative."""
    return 0 if TP == 0 and FN == 0 else TP / (TP + FN)


# Tested
def F1(precision, recall):
    """Calculates F1 score from precision and recall numbers."""
    return 0 if precision == 0 and recall == 0 else 2 * (precision * recall) / (precision + recall)


def distance_Evaluations(log, shape, newPredictions, trueLabels):
    """Brings together and returns the results of multiple distance calculations. Currently is only returns custom."""
    custTN, custNT, countTrue, countNew, newEvalInfoCustDist = custom_distance(shape, newPredictions, trueLabels)

    log.write(
        "\nCustom Label Distance:\nCustom Distance From True to Predicted Data,{}\nCustom Distance From Predicted "
        "to True Data,{}".format(custTN, custNT))

    log.write(
        "\nAverage Custom Distance From True to Predicted Statement,{}\nAverage Custom Distance From Prediction to "
        "True Statement,{}\n".format(custTN / countTrue, 0 if countNew == 0 else custNT / countNew))

    c = write_evaluation_measures(newEvalInfoCustDist, log)

    return np.array(np.array([custTN, custNT, countTrue, countNew, c]))


def custom_distance(shape, newPred, trueLabels):
    """Finds the number of true and false positives and false negatives.  Also collects distance data which is recorded."""

    # Combines all the timestep lists together so can compare whole samples.
    flatTrue = [[item for sublist in x for item in sublist] for x in trueLabels]
    flatNew = [[item for sublist in x for item in sublist] for x in newPred]

    # [0] = True Positives, [1] = False Positives, [2] = False Negatives
    evalInfoNew = np.array([0, 0, 0])

    custTN = 0 # True to New
    custNT = 0

    countTrue = 0
    countNew = 0

    # Loops through the output tensor of a model.
    for sampleNum in range(shape[0]):
        for timestepNum in range(shape[1]):
            for tripleNum in range(shape[2]):

                if len(trueLabels) > sampleNum and len(trueLabels[sampleNum]) > timestepNum and len(trueLabels[sampleNum][timestepNum]) > tripleNum:
                    countTrue = countTrue + 1
                    if len(newPred) > sampleNum and len(newPred[sampleNum]) > 0:
                        custTN = custTN + find_best_prediction_custom(trueLabels[sampleNum][timestepNum][tripleNum], flatNew[sampleNum])
                    else:
                        custTN = custTN + custom(trueLabels[sampleNum][timestepNum][tripleNum], [])

                if len(newPred) > sampleNum and len(newPred[sampleNum]) > timestepNum and len(newPred[sampleNum][timestepNum]) > tripleNum:  # Out of bounds check
                    countNew = countNew + 1
                    if len(trueLabels) > sampleNum and len(trueLabels[sampleNum]) > 0 and len(flatTrue[sampleNum]) != 0:
                        best = find_best_prediction_custom(newPred[sampleNum][timestepNum][tripleNum], flatTrue[sampleNum])
                        if best == 0:
                            evalInfoNew[0] = evalInfoNew[0] + 1
                        custNT = custNT + best
                    else:
                        custNT = custNT + custom(newPred[sampleNum][timestepNum][tripleNum], [])


    # Calculating False positives.
    evalInfoNew[1] = countNew - evalInfoNew[0]
    evalInfoNew[2] = countTrue - evalInfoNew[0]
    return custTN, custNT, countTrue, countNew, evalInfoNew


def find_best_prediction_custom(statement, otherKB):
    return min(map(partial(custom, statement), otherKB))


# Tested
def custom(tupleTriple1, tupleTriple2):
    if len(tupleTriple1) < len(tupleTriple2): return custom(tupleTriple2, tupleTriple1)

    dist = 0

    if tupleTriple1 == tupleTriple2:
        return 0
    else:
        for k in range(len(tupleTriple1)):
            string1 = tupleTriple1[k]
            string2 = tupleTriple2[k] if len(tupleTriple2) > k else ""
            if string2 == "":
                dist = dist + int(
                    ''.join(x for x in string1 if x.isdigit()))  # + (conceptSpace if string1[0] == 'C' else roleSpace)
            else:
                if (string1[0] == 'C' and string2[0] == 'R') or (string1[0] == 'R' and string2[0] == 'C'):
                    dist = dist + abs(
                        int(''.join(x for x in string1 if x.isdigit())) + int(''.join(x for x in string2 if x.isdigit())))
                else:
                    dist = dist + abs(
                        int(''.join(x for x in string1 if x.isdigit())) - int(''.join(x for x in string2 if x.isdigit())))

    return dist


# $$


def write_vector_file(filename, vector):
    file = io.open(filename, "w", encoding='utf-8')
    for i in range(len(vector)):
        file.write("Trial: {}\n".format(i))
        for j in range(len(vector[i])):
            file.write("\tStep: {}\n".format(j))
            for k in range(len(vector[i][j])):
                file.write("\t\t{}\n".format(vector[i][j][k]))
        file.write("\n")
    file.close()


# Tested
def training_stats(log, mseNew, mse0, mseL):
    """Calculates and logs training data passed from a learning model."""
    log.write(
        "Training Statistics\nPrediction Mean Squared Error,{}\nLearned Reduction MSE,{}\nIncrease (+) or Decrease (-) "
        "MSE on Test,{}\n" "Training Percent Change MSE,{}%\n".format(
            np.float32(mseNew), round(mse0 - mseL, 5), round(np.float32(mseNew) - mseL, 5), round(((mseL - mse0) / mse0 * 100)), 4))


# Tested
def write_evaluation_measures(F, log):
    TPs, FPs, FNs = F
    pre = precision(TPs, FPs)
    rec = recall(TPs, FNs)
    F = F1(pre, rec)

    log.write(
        "\nPrediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs, FPs, FNs, pre, rec, F))

    x = np.array([TPs, FPs, FNs, pre, rec, F])

    return x


# Tested barely
def write_final_average_data(result, log):
    custTN, custNT, countTrue, countNew, (TPs1, FPs1, FNs1, pre1, rec1, F1) = result

    log.write(
        "\nCustom Label Distance:\nAverage Custom Distance From True to Predicted Data,{}\nAverage Custom Distance "
        "From Predicted to True Data,{}".format(custTN, custNT))

    log.write(
        "\nAverage Custom Distance From True to Predicted Statement,{}\nAverage Custom Distance From Prediction to "
        "True Statement,{}\n".format(custTN / countTrue, custNT / countNew))

    log.write(
        "\nAverage Prediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse "
        "Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(TPs1, FPs1, FNs1, pre1, rec1, F1))


# $$


def get_rdf_data(file):
    """Gets RDF data from json file specified."""
    with open(file) as f:
        data = json.load(f)
        return data


def pad_kb(_list):
    """Pads a list of lists so that each list within the main list is equal sizes."""
    targetPadNum = 0
    for i in range(len(_list)):
        # Gets the max padding length
        if(len(_list[i]) > targetPadNum):
            targetPadNum = len(_list[i])

    # Pads each list within the main list
    for j in range(len(_list)):
        while len(_list[j]) < targetPadNum:
            _list[j].append(0.0)

    return _list


# Tested
def pad_list_of_lists(_list):
    """Pads a list of lists so that each list within the main list is equal sizes."""
    for i in range(len(_list)):
        element = _list[i]
        targetPadNum = 0

        # Gets the max padding length
        for j in range(len(element)):
            temp = element[j]
            if(len(temp) > targetPadNum):
                targetPadNum = len(temp)

        # Pads each list within the main list
        for j in range(len(element)):
            temp = element[j]
            while len(temp) < targetPadNum:
                temp.append(0.0)
    return _list


def convert_data_to_arrays(data):
    """Takes data and makes sure they are arrays."""
    kb, supp, outs, numToStmMap = data['kB'], data['supports'], data['outputs'], data['vectorMap']

    supp = pad_list_of_lists(supp)
    outs = pad_list_of_lists(outs)
    kb = pad_kb(kb)

    KB = numpy.array(kb)
    Supp = numpy.zeros((len(supp)), dtype=numpy.ndarray)
    Outs = numpy.zeros((len(outs)), dtype=numpy.ndarray)

    for i in range(len(supp)):
        Supp[i] = numpy.array(supp[i])

    for i in range(len(outs)):
        Outs[i] = numpy.array(outs[i])

    numToStmMap = numpy.array(numToStmMap)

    return KB, Supp, Outs, numToStmMap, data['concepts'], data['roles']


# Tested
def cross_validation_split_all_data(n, KBs, supports, outputs, encodedMap):
    # Potentially calculates size of the 3D tensor which will be padded and passed to the LSTM.
    fileShapes1 = [len(supports[0]), len(max(supports, key=lambda coll: len(coll[0]))[0]),
                   len(max(outputs, key=lambda coll: len(coll[0]))[0])]

    print("Repeating KBs")
    # Creates a new 3D tensor where the original KB sample is copied once per timestep.
    newKBs = numpy.zeros([KBs.shape[0], fileShapes1[0], KBs.shape[1]], dtype=float)
    # Goes through for every line of the KB
    for crossNum in range(len(newKBs)):
        for sampleNum in range(fileShapes1[0]):
            # The same KB line is copied timestep times in the sampleNum's place.
            newKBs[crossNum][sampleNum] = KBs[crossNum]

    KBs = newKBs

    print("Shuffling Split Indices")
    # Creates a list of test_indices up to the length of the # of Batch size?
    test_indices = list(range(len(KBs)))
    random.shuffle(test_indices)
    # k is the quotient and m is the remainder
    k, m = divmod(len(test_indices), n)
    # Creating num of cross validations of lists and assigning which test_indices are going to each one.
    test_indices = list(test_indices[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    # Creating place holding arrays of correct shape to be filled later.
    crossKBsTest = numpy.zeros((len(test_indices), len(test_indices[0]), len(KBs[0]), len(KBs[0][0])), dtype=float)
    crossSupportsTest = numpy.zeros(
        (len(test_indices), len(test_indices[0]), len(supports[0]), fileShapes1[1]), dtype=float)
    crossOutputsTest = numpy.zeros(
        (len(test_indices), len(test_indices[0]), len(outputs[0]), fileShapes1[2]), dtype=float)

    # Probably don't need yet.
    # crossErrTest = None if not isinstance(mKBs, numpy.ndarray) else numpy.zeros(
    #     (len(test_indices), len(test_indices[0]), len(mouts[0]), maxout), dtype=float)

    # If localMap is provided then create empty crossLabels array.
    if isinstance(encodedMap, numpy.ndarray):
        crossLabels = numpy.empty((len(test_indices), len(test_indices[0])), dtype=dict)
    else:
        crossLabels = None

    print("Extracting Test Sets")
    for crossNum in range(len(test_indices)): # crossNum is each cross validation fold.
        KBns = []
        for sampleNum in range(len(test_indices[crossNum])): # sampleNum is each index in the crossNum validation fold.
            crossKBsTest[crossNum][sampleNum] = KBs[test_indices[crossNum][sampleNum]] # assigning whole (timestep x numbers) arrays

            # I think the hstack is padding each support and output with zeros to match the max len.
            crossSupportsTest[crossNum][sampleNum] = numpy.hstack([supports[test_indices[crossNum][sampleNum]], numpy.zeros(
                [fileShapes1[0], fileShapes1[1] - len(supports[test_indices[crossNum][sampleNum]][0])])])
            crossOutputsTest[crossNum][sampleNum] = numpy.hstack([outputs[test_indices[crossNum][sampleNum]], numpy.zeros(
                [fileShapes1[0], fileShapes1[2] - len(outputs[test_indices[crossNum][sampleNum]][0])])])

        write_vector_file("crossValidationFolds/output/originalKBsIn[{}].txt".format(crossNum), numpy.array(KBns))

    crossKBsTrain = numpy.zeros((n, len(KBs) - len(crossKBsTest[0]), len(KBs[0]), len(KBs[0][0])), dtype=float)
    crossOutputsTrain = numpy.zeros((n, len(KBs) - len(crossKBsTest[0]), len(outputs[0]), fileShapes1[2]), dtype=float)
    crossSupportsTrain = numpy.zeros((n, len(KBs) - len(crossKBsTest[0]), len(supports[0]), fileShapes1[1]), dtype=float)

    print("Extracting Train Sets")
    for crossNum in range(len(test_indices)):

        all_indices = set(range(len(KBs)))
        trainIndices = list(all_indices.difference(set(test_indices[crossNum])))
        random.shuffle(trainIndices)

        for sampleNum in range(len(crossKBsTrain[crossNum])):

            crossKBsTrain[crossNum][sampleNum] = KBs[trainIndices[sampleNum]]

            crossSupportsTrain[crossNum][sampleNum] = numpy.hstack([supports[trainIndices[sampleNum]], numpy.zeros(
                [fileShapes1[0], fileShapes1[1] - len(supports[trainIndices[sampleNum]][0])])])

            crossOutputsTrain[crossNum][sampleNum] = numpy.hstack([outputs[trainIndices[sampleNum]],
                                                                   numpy.zeros([fileShapes1[0], fileShapes1[2] - len(
                                                                       outputs[trainIndices[sampleNum]][0])])])

    return crossKBsTest, crossKBsTrain, crossSupportsTrain, crossSupportsTest, crossOutputsTrain, crossOutputsTest,\
           crossLabels


def get_labels_from_encoding(trueValues, predValues, decodeNonProperty, decodeProperty):
    """Creates and returns the label representation of a models true and predicted encoded array."""

    trueArray = np.zeros(shape=(trueValues.shape[0], trueValues.shape[1]), dtype=tuple)
    predArray = np.zeros(shape=(predValues.shape[0], predValues.shape[1]), dtype=tuple)

    for sampleNum in range(len(trueValues)):
        for timeStepNum in range(len(trueValues[sampleNum])):
            trueTriple = []
            predTriple = []
            timeStepListTripleTrue = []
            timeStepListTriplePred = []
            for item in range(len(trueValues[sampleNum][timeStepNum])):
                trueTriple.append(trueValues[sampleNum][timeStepNum][item])
                predTriple.append(predValues[sampleNum][timeStepNum][item])
                if len(trueTriple) == 3:
                    label1 = convert_encoded_float_to_label(trueTriple[0], decodeNonProperty, decodeProperty)
                    label2 = convert_encoded_float_to_label(trueTriple[1], decodeNonProperty, decodeProperty)
                    label3 = convert_encoded_float_to_label(trueTriple[2], decodeNonProperty, decodeProperty)
                    if label1 == '0' or label2 == '0' or label3 == '0':
                        pass
                    else:
                        timeStepListTripleTrue.append(tuple((label1, label2, label3)))
                    trueTriple = []
                if len(predTriple) == 3:
                    label1 = convert_encoded_float_to_label(predTriple[0], decodeNonProperty, decodeProperty)
                    label2 = convert_encoded_float_to_label(predTriple[1], decodeNonProperty, decodeProperty)
                    label3 = convert_encoded_float_to_label(predTriple[2], decodeNonProperty, decodeProperty)
                    if label1 == '0' or label2 == '0' or label3 == '0':
                        pass
                    else:
                        timeStepListTriplePred.append(tuple((label1, label2, label3)))
                    predTriple = []
            trueArray[sampleNum][timeStepNum] = timeStepListTripleTrue
            predArray[sampleNum][timeStepNum] = timeStepListTriplePred

    return trueArray, predArray


def convert_encoded_float_to_label(float, decodeNonProperty, decodeProperty):
    """Converts one floating encoded number to its appropriet label based on the decoding numbers."""
    if float > 0:
        labelNum = round(float * decodeNonProperty)
        if labelNum == 0:
            return '0'
        if labelNum > decodeNonProperty:
            labelNum = decodeNonProperty
        return 'C' + str(labelNum)
    elif float < 0:
        labelNum = round(float * decodeProperty) * -1
        if labelNum == 0:
            return '0'
        if labelNum > decodeProperty:
            labelNum = decodeProperty
        return 'R' + str(labelNum)
    else:
        return '0'

# $$


def flat_system(n_epochs2, learning_rate2, trainlog, evallog, allTheData, n):
    """"The base line recurrent model for comparison with other rnn models."""
    KBs_test, KBs_train, X_train, X_test, y_train, y_test, iriMap, decodeNonProperty, decodeProperty = allTheData

    trainlog.write("Flat LSTM\nEpoch,Mean Squared Error,Root Mean Squared Error\n")
    evallog.write("\nFlat LSTM\n\n")
    print("")

    X0 = tf.compat.v1.placeholder(tf.float32, shape=[None, KBs_train.shape[1], KBs_train.shape[2]])
    y1 = tf.compat.v1.placeholder(tf.float32, shape=[None, y_train.shape[1], y_train.shape[2]])

    outputs2, states2 = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(num_units=y_train.shape[2]), X0, dtype=tf.float32)

    loss2 = tf.losses.mean_squared_error(y1, outputs2)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate2)
    training_op2 = optimizer2.minimize(loss2)

    # saver = tf.train.Saver()

    init2 = tf.global_variables_initializer()

    with tf.Session() as sess:
        init2.run()
        mse0 = 0
        mseL = 0
        for epoch in range(n_epochs2):
            print("Flat System\t\tEpoch: {}".format(epoch))
            ynew, a = sess.run([outputs2, training_op2], feed_dict={X0: KBs_train, y1: y_train})
            mse = loss2.eval(feed_dict={outputs2: ynew, y1: y_train})
            if epoch == 0: mse0 = mse
            if epoch == n_epochs2 - 1: mseL = mse
            trainlog.write("{},{},{}\n".format(epoch, mse, math.sqrt(mse)))
            if mse < 0.0001:
                mseL = mse
                break

        print("\nEvaluating Result\n")

        y_pred = sess.run(outputs2, feed_dict={X0: KBs_test})
        mseNew = loss2.eval(feed_dict={outputs2: y_pred, y1: y_test})

        training_stats(evallog, mseNew, mse0, mseL)

        evallog.write("\nTest Data Evaluation\n")

        trueLabels, labelPredictions = get_labels_from_encoding(y_test, y_pred, decodeNonProperty, decodeProperty)

        # write_vector_file("crossValidationFolds/output/predictionFlatArchitecture[{}].txt".format(n), newStrIRI)

        return distance_Evaluations(evallog, y_pred.shape, labelPredictions, trueLabels)


def deep_system(n_epochs2, learning_rate2, trainlog, evallog, allTheData, n):
    KBs_test, KBs_train, X_train, X_test, y_train, y_test, iriMap, decodeNonProperty, decodeProperty = allTheData

    trainlog.write("Deep LSTM\nEpoch,Mean Squared Error,Root Mean Squared Error\n")
    evallog.write("\nDeep LSTM\n\n")
    print("")

    X0 = tf.placeholder(tf.float32, shape=[None, KBs_train.shape[1], KBs_train.shape[2]])
    y1 = tf.placeholder(tf.float32, shape=[None, y_train.shape[1], y_train.shape[2]])

    outputs2, states2 = tf.nn.dynamic_rnn(tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.LSTMCell(num_units=X_train.shape[2]), tf.nn.rnn_cell.LSTMCell(num_units=y_train.shape[2])]), X0,
                                          dtype=tf.float32)

    loss2 = tf.losses.mean_squared_error(y1, outputs2)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate2)
    training_op2 = optimizer2.minimize(loss2)

    init2 = tf.global_variables_initializer()

    with tf.Session() as sess:
        init2.run()
        mse0 = 0
        mseL = 0
        for epoch in range(n_epochs2):
            print("Deep System\t\tEpoch: {}".format(epoch))
            ynew, a = sess.run([outputs2, training_op2], feed_dict={X0: KBs_train, y1: y_train})
            mse = loss2.eval(feed_dict={outputs2: ynew, y1: y_train})
            if epoch == 0: mse0 = mse
            if epoch == n_epochs2 - 1: mseL = mse
            trainlog.write("{},{},{}\n".format(epoch, mse, math.sqrt(mse)))
            if mse < 0.0001:
                mseL = mse
                break

        print("\nEvaluating Result\n")

        y_pred = sess.run(outputs2, feed_dict={X0: KBs_test})
        mseNew = loss2.eval(feed_dict={outputs2: y_pred, y1: y_test})

        training_stats(evallog, mseNew, mse0, mseL)

        evallog.write("\nTest Data Evaluation\n")

        trueLabels, labelPredictions = get_labels_from_encoding(y_test, y_pred, decodeNonProperty, decodeProperty)

        # write_vector_file("crossValidationFolds/output/predictionDeepArchitecture[{}].txt".format(n), newStrIRI)

        return distance_Evaluations(evallog, y_pred.shape, labelPredictions, trueLabels)


def piecewise_system(n_epochs0, learning_rate0, trainlog, evallog, allTheData, n):
    KBs_test, KBs_train, X_train, X_test, y_train, y_test, iriMap, decodeNonProperty, decodeProperty = allTheData

    trainlog.write("Piecewise LSTM Part One\nEpoch,Mean Squared Error,Root Mean Squared Error\n")
    evallog.write("Piecewise LSTM Part One\n")
    print("")

    n_neurons0 = X_train.shape[2]

    X0 = tf.compat.v1.placeholder(tf.float32, shape=[None, KBs_train.shape[1], KBs_train.shape[2]])
    y0 = tf.compat.v1.placeholder(tf.float32, shape=[None, X_train.shape[1], X_train.shape[2]])

    outputs0, states0 = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(num_units=n_neurons0), X0, dtype=tf.float32)

    loss0 = tf.losses.mean_squared_error(y0, outputs0)
    optimizer0 = tf.train.AdamOptimizer(learning_rate=learning_rate0)
    training_op0 = optimizer0.minimize(loss0)

    init0 = tf.global_variables_initializer()

    with tf.Session() as sess:
        init0.run()
        mse0 = 0
        mseL = 0
        for epoch in range(n_epochs0):
            print("Piecewise System\tEpoch: {}".format(epoch))
            ynew, a = sess.run([outputs0, training_op0], feed_dict={X0: KBs_train, y0: X_train})
            mse = loss0.eval(feed_dict={outputs0: ynew, y0: X_train})
            if epoch == 0: mse0 = mse
            if epoch == n_epochs0 - 1: mseL = mse
            trainlog.write("{},{},{}\n".format(epoch, mse, math.sqrt(mse)))
            if mse < 0.00001:
                mseL = mse
                break

        print("\nTesting first half\n")

        y_pred = sess.run(outputs0, feed_dict={X0: KBs_test, y0: X_test})
        mseNew = loss0.eval(feed_dict={outputs0: y_pred, y0: X_test})

        training_stats(evallog, mseNew, mse0, mseL)

        # write_vector_file("crossValidationFolds/output/learnedSupportsP[{}].txt".format(n),newStrIRI)

        numpy.savez("crossValidationFolds/saves/halfwayData[{}]".format(n), y_pred)

    tf.reset_default_graph()

    trainlog.write("Piecewise LSTM Part Two\nEpoch,Mean Squared Error,Root Mean Squared Error\n")
    evallog.write("\nPiecewise LSTM Part Two\n")

    n_neurons1 = y_train.shape[2]

    X1 = tf.placeholder(tf.float32, shape=[None, X_train.shape[1], X_train.shape[2]])
    y1 = tf.placeholder(tf.float32, shape=[None, y_train.shape[1], y_train.shape[2]])

    outputs1, states1 = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(num_units=n_neurons1), X1, dtype=tf.float32)

    loss1 = tf.losses.mean_squared_error(y1, outputs1)
    optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate0)
    training_op1 = optimizer1.minimize(loss1)

    init1 = tf.global_variables_initializer()

    with tf.Session() as sess:
        init1.run()
        mse0 = 0
        mseL = 0
        for epoch in range(n_epochs0):
            print("Piecewise System\tEpoch: {}".format(epoch + n_epochs0))
            ynew, a = sess.run([outputs1, training_op1], feed_dict={X1: X_train, y1: y_train})
            mse = loss1.eval(feed_dict={outputs1: ynew, y1: y_train})
            if epoch == 0: mse0 = mse
            if epoch == n_epochs0 - 1: mseL = mse
            trainlog.write("{},{},{}\n".format(epoch, mse, math.sqrt(mse)))
            if mse < 0.00001:
                mseL = mse
                break

        print("\nTesting second half")
        y_pred = sess.run(outputs1, feed_dict={X1: X_test})
        mseNew = loss1.eval(feed_dict={outputs1: y_pred, y1: y_test})

        training_stats(evallog, mseNew, mse0, mseL)

        print("\nEvaluating Result")

        evallog.write("\nReasoner Support Test Data Evaluation\n")

        # write_vector_file("crossValidationFolds/output/predictedOutLeftOverSupportTest[{}].txt".format(n), newStrIRI)

        data = numpy.load("crossValidationFolds/saves/halfwayData[{}].npz".format(n), allow_pickle=True)
        data = data['arr_0']

        evallog.write("\nSaved Test Data From Previous LSTM Evaluation\n")

        y_pred = sess.run(outputs1, feed_dict={X1: data})
        mseNew = loss1.eval(feed_dict={outputs1: y_pred, y1: y_test})

        evallog.write("\nTesting Statistic\nIncrease MSE on Saved,{}\n".format(numpy.float32(mseNew) - mseL))

        trueLabels, labelPredictions = get_labels_from_encoding(y_test, y_pred, decodeNonProperty, decodeProperty)

        # write_vector_file("crossValidationFolds/output/predictionSavedKBPipeline[{}].txt".format(n), newStrIRI)

        return distance_Evaluations(evallog, y_pred.shape, labelPredictions, trueLabels)


def run_nth_time(trainlog, evallog, epochs, learningRate, nthData, n):
    """Runs and collects results from shallow, deep, and flat models for one cycle of the cross validation."""
    evals1 = piecewise_system(int(epochs / 2), learningRate, trainlog, evallog, nthData, n)

    tf.reset_default_graph()

    evals2 = deep_system(epochs, learningRate / 2, trainlog, evallog, nthData, n)

    tf.reset_default_graph()

    evals3 = flat_system(epochs, learningRate / 2, trainlog, evallog, nthData, n)

    tf.reset_default_graph()

    trainlog.close()
    evallog.close()

    return evals1, evals2, evals3


def n_times_cross_validate(n, epochs, learningRate, decodingNonProperty, decodingProperty, dataFile):
    # Sets up logging.
    if not os.path.isdir("crossValidationFolds"): os.mkdir("crossValidationFolds")
    if not os.path.isdir("crossValidationFolds/training"): os.mkdir("crossValidationFolds/training")
    if not os.path.isdir("crossValidationFolds/evals"): os.mkdir("crossValidationFolds/evals")
    if not os.path.isdir("crossValidationFolds/saves"): os.mkdir("crossValidationFolds/saves")
    if not os.path.isdir("crossValidationFolds/output"): os.mkdir("crossValidationFolds/output")

    # Gets raw data.
    KB, supports, outputs, encodingMap, numConcepts, numRoles = convert_data_to_arrays(get_rdf_data(dataFile))

    # Processes data.

    allTheData = cross_validation_split_all_data(n, KB, supports, outputs, encodingMap)

    KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, labelss = allTheData

    if isinstance(labelss, numpy.ndarray):
        if (labelss.ndim and labelss.size) == 0:
            labelss = None

    evals = numpy.zeros((1, 5), dtype=numpy.float64)
    for i in range(n):
        print("\nCross Validation Fold {}\n\nTraining With {} Data\nTesting With {} Data\n".format(i, "RDF", "RDF"))
        x = run_nth_time(open("crossValidationFolds/training/trainFold[{}].csv".format(i), "w"),
                         open("crossValidationFolds/evals/evalFold[{}].csv".format(i), "w"), epochs, learningRate,
                         (KBs_tests[i], KBs_trains[i], X_trains[i], X_tests[i], y_trains[i], y_tests[i],
                          (labelss[i] if isinstance(labelss, numpy.ndarray) else None), decodingNonProperty,
                          decodingProperty), i)
        evals = evals + x

    evals = evals / n

    print("Summarizing All Results")

    avgResult = evals.tolist()

    log = open("crossValidationFolds/evalAllFolds[avg].csv", "w")

    log.write("Piecewise System\n")

    write_final_average_data(avgResult[0], log)

    log.write("\nDeep System\n")

    write_final_average_data(avgResult[1], log)

    log.write("\nFlat System\n")

    write_final_average_data(avgResult[2], log)

    log.close()

    print("\nDone")

    return avgResult


def read_inputs():
    """Collects arguments to be passed into the model."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", help="number of epochs for each system", type=int, default=20000)  # 20000
    parser.add_argument("-l", "--learningRate", help="learning rate of each system", type=float, default=0.0001)
    parser.add_argument("-c", "--cross", help="cross validation k", type=int, default=5)  # 10

    args = parser.parse_args()

    if args.epochs and args.epochs < 2:
        raise ValueError("Try a bigger number maybe!")

    if args.cross:
        if args.cross < 1:
            raise ValueError("K fold Cross Validation works better with k greater than 1")

    return args


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    tf.disable_v2_behavior()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    args = read_inputs()

    n_times_cross_validate(n=args.cross, epochs=args.epochs, learningRate=args.learningRate, decodingNonProperty=28,
                           decodingProperty=14, dataFile="rdfData/schemaorg.json")