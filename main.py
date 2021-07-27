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


'''
#Logging functions------------------------------------------------------------------------------------------------------

def writeFinalAverageDataMess(result, log):
    levTR, levRT, levTN, levNT, sizeTrue, sizeNew, sizeRan, (
    (TPs, FPs, FNs, pre, rec, F), (rTPs, rFPs, rFNs, rpre, rrec, rF)), (
    h, i, j, k, (mTPs, mFPs, mFNs, mpre, mrec, mF)) = result[0]
    levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, (
    (TPs1, FPs1, FNs1, pre1, rec1, F1), (rTPs1, rFPs1, rFNs1, rpre1, rrec1, rF1)), (
    h1, i1, j1, k1, (mTPs1, mFPs1, mFNs1, mpre1, mrec1, mF1)) = result[1]
    custTR, custRT, custTN, custNT, countTrue, countNew, countRan, (
    (TPs2, FPs2, FNs2, pre2, rec2, F2), (rTPs2, rFPs2, rFNs2, rpre2, rrec2, rF2)), (
    h2, i2, j2, k2, (mTPs2, mFPs2, mFNs2, mpre2, mrec2, mF2)) = result[2]
    log.write(
        "\nNo Nums\nAverage Levenshtein Distance From Reasoner to Random Data,{}\nAverage Levenshtein Distance From Random to Reasoner Data,{}\nAverage Levenshtein Distance From Reasoner to Predicted Data,{}\nAverage Levenshtein Distance From Prediction to Reasoner Data,{}\n".format(
            levTR, levRT, levTN, levNT))
    log.write(
        "Average Levenshtein Distance From Reasoner to Error Data,{}\nAverage Levenshtein Distance From Error to Reasoner Data,{}\n".format(
            h, i))
    log.write(
        "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
            levTR / sizeTrue, levRT / sizeRan, levTN / sizeTrue, levNT / sizeNew))
    log.write(
        "Average Levenshtein Distance From Reasoner to Error Statement,{}\nAverage Levenshtein Distance From Error to Reasoner Statement,{}\n".format(
            h / j, 0 if k == 0 else i / k))

    log.write(
        "\nAverage Prediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs, FPs, FNs, pre, rec, F))
    log.write(
        "\nAverage Random Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            rTPs, rFPs, rFNs, rpre, rrec, rF))
    log.write(
        "\nAverage Error Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            mTPs, mFPs, mFNs, mpre, mrec, mF))

    log.write(
        "\nNums\nAverage Levenshtein Distance From Reasoner to Random Data,{}\nAverage Levenshtein Distance From Random to Reasoner Data,{}\nAverage Levenshtein Distance From Reasoner to Predicted Data,{}\nAverage Levenshtein Distance From Prediction to Reasoner Data,{}\n".format(
            levTR2, levRT2, levTN2, levNT2))
    log.write(
        "Average Levenshtein Distance From Reasoner to Error Data,{}\nAverage Levenshtein Distance From Error to Reasoner Data,{}\n".format(
            h1, i1))
    log.write(
        "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
            levTR2 / sizeTrue2, levRT2 / sizeRan2, levTN2 / sizeTrue2, levNT2 / sizeNew2))
    log.write(
        "Average Levenshtein Distance From Reasoner to Error Statement,{}\nAverage Levenshtein Distance From Error to Reasoner Statement,{}\n".format(
            h1 / j1, 0 if k1 == 0 else i1 / k1))

    log.write(
        "\nAverage Prediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs1, FPs1, FNs1, pre1, rec1, F1))
    log.write(
        "\nAverage Random Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            rTPs1, rFPs1, rFNs1, rpre1, rrec1, rF1))
    log.write(
        "\nAverage Error Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            mTPs1, mFPs1, mFNs1, mpre1, mrec1, mF1))

    log.write(
        "\nCustom\nAverage Custom Distance From Reasoner to Random Data,{}\nAverage Custom Distance From Random to Reasoner Data,{}\nAverage Custom Distance From Reasoner to Predicted Data,{}\nAverage Custom Distance From Predicted to Reasoner Data,{}\n".format(
            custTR, custRT, custTN, custNT))
    log.write(
        "Average Custom Distance From Reasoner to Error Data,{}\nAverage Custom Distance From Error to Reasoner Data,{}\n".format(
            h2, i2))
    log.write(
        "Average Custom Distance From Reasoner to Random Statement,{}\nAverage Custom Distance From Random to Reasoner Statement,{}\nAverage Custom Distance From Reasoner to Predicted Statement,{}\nAverage Custom Distance From Prediction to Reasoner Statement,{}\n".format(
            custTR / countTrue, custRT / countRan, custTN / countTrue, custNT / countNew))
    log.write(
        "Average Custom Distance From Reasoner to Error Statement,{}\nAverage Custom Distance From Error to Reasoner Statement,{}\n".format(
            h2 / j2, 0 if k2 == 0 else i2 / k2))

    log.write(
        "\nAverage Prediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs2, FPs2, FNs2, pre2, rec2, F2))
    log.write(
        "\nAverage Random Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            rTPs2, rFPs2, rFNs2, rpre2, rrec2, rF2))
    log.write(
        "\nAverage Error Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            mTPs2, mFPs2, mFNs2, mpre2, mrec2, mF2))

'''

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


def training_stats(log, mseNew, mse0, mseL):
    log.write(
        "Training Statistics\nPrediction Mean Squared Error,{}\nLearned Reduction MSE,{}\nIncrease MSE on Test,{}\nTraining Percent Change MSE,{}\n".format(
            np.float32(mseNew), mse0 - mseL, np.float32(mseNew) - mseL, (mseL - mse0) / mse0 * 100))


def write_evaluation_measures(F, rF, log):
    TPs, FPs, FNs = F
    pre = precision(TPs, FPs)
    rec = recall(TPs, FNs)
    F = F1(pre, rec)

    log.write(
        "\nPrediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs, FPs, FNs, pre, rec, F))

    x = np.array([TPs, FPs, FNs, pre, rec, F])

    TPs, FPs, FNs = rF
    pre = precision(TPs, FPs)
    rec = recall(TPs, FNs)
    F = F1(pre, rec)

    log.write(
        "\nRandom Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs, FPs, FNs, pre, rec, F))

    return np.array([x, np.array([TPs, FPs, FNs, pre, rec, F])])


def write_final_average_data(result, log):
    levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, (
    (TPs1, FPs1, FNs1, pre1, rec1, F1), (rTPs1, rFPs1, rFNs1, rpre1, rrec1, rF1)) = result[0]
    custTR, custRT, custTN, custNT, countTrue, countNew, countRan, (
    (TPs2, FPs2, FNs2, pre2, rec2, F2), (rTPs2, rFPs2, rFNs2, rpre2, rrec2, rF2)) = result[1]

    log.write(
        "\nString Distance:\nAverage Levenshtein Distance From Reasoner to Random Data,{}\nAverage Levenshtein Distance From Random to Reasoner Data,{}\nAverage Levenshtein Distance From Reasoner to Predicted Data,{}\nAverage Levenshtein Distance From Prediction to Reasoner Data,{}\n".format(
            levTR2, levRT2, levTN2, levNT2))
    log.write(
        "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
            levTR2 / sizeTrue2, levRT2 / sizeRan2, levTN2 / sizeTrue2, levNT2 / sizeNew2))

    log.write(
        "\nAverage Prediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs1, FPs1, FNs1, pre1, rec1, F1))
    log.write(
        "\nAverage Random Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            rTPs1, rFPs1, rFNs1, rpre1, rrec1, rF1))

    log.write(
        "\nCustom Label Distance:\nAverage Custom Distance From Reasoner to Random Data,{}\nAverage Custom Distance From Random to Reasoner Data,{}\nAverage Custom Distance From Reasoner to Predicted Data,{}\nAverage Custom Distance From Predicted to Reasoner Data,{}\n".format(
            custTR, custRT, custTN, custNT))
    log.write(
        "Average Custom Distance From Reasoner to Random Statement,{}\nAverage Custom Distance From Random to Reasoner Statement,{}\nAverage Custom Distance From Reasoner to Predicted Statement,{}\nAverage Custom Distance From Prediction to Reasoner Statement,{}\n".format(
            custTR / countTrue, custRT / countRan, custTN / countTrue, custNT / countNew))

    log.write(
        "\nAverage Prediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs2, FPs2, FNs2, pre2, rec2, F2))
    log.write(
        "\nAverage Random Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            rTPs2, rFPs2, rFNs2, rpre2, rrec2, rF2))


def flat_system(n_epochs2, learning_rate2, trainlog, evallog, allTheData, n):
    """"The base line recurrent model for comparison with other rnn models."""
    KBs_test, KBs_train, X_train, X_test, y_train, y_test, iriMap, trueLabels, trueIRIs, labels, numConcepts,\
        numRoles = allTheData

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

        newPredictions, newStrIRI = get_label_and_iri_from_encoding(y_pred, labels, numConcepts, numRoles)

        write_vector_file("crossValidationFolds/output/predictionFlatArchitecture[{}].txt".format(n), newStrIRI)

        return distanceEvaluations(evallog, y_pred.shape, newPredictions, trueLabels, newStrIRI, trueIRIs,
                                   numConcepts, numRoles, labels)


def deep_system(n_epochs2, learning_rate2, trainlog, evallog, allTheData, n):
    KBs_test, KBs_train, X_train, X_test, y_train, y_test, iriMap, trueLabels, trueIRIs, labels, numConcepts, numRoles = allTheData

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

        newPredictions, newStrIRI = get_label_and_iri_from_encoding(y_pred, labels, numConcepts, numRoles)

        write_vector_file(
            "crossValidationFolds/output/predictionDeepArchitecture[{}].txt".format(n),
            newStrIRI)

        return distanceEvaluations(evallog, y_pred.shape, newPredictions, trueLabels, newStrIRI, trueIRIs,
                                   numConcepts, numRoles, labels)


def shallowSystem(n_epochs0, learning_rate0, trainlog, evallog, allTheData, n):
    KBs_test, KBs_train, X_train, X_test, y_train, y_test, iriMap, trueLabels, trueIRIs, labels, numConcepts, numRoles = allTheData

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
        newPredictions, newStrIRI = get_label_and_iri_from_encoding(y_pred, labels, numConcepts, numRoles)

        training_stats(evallog, mseNew, mse0, mseL)

        write_vector_file("crossValidationFolds/output/learnedSupportsP[{}].txt".format(n),newStrIRI)

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

        newPredictions, newStrIRI = get_label_and_iri_from_encoding(y_pred, labels, numConcepts, numRoles)

        write_vector_file(
            "crossValidationFolds/output/predictedOutLeftOverSupportTest[{}].txt".format(n), newStrIRI)

        data = numpy.load("crossValidationFolds/saves/halfwayData[{}].npz".format(n), allow_pickle=True)
        data = data['arr_0']

        evallog.write("\nSaved Test Data From Previous LSTM Evaluation\n")

        y_pred = sess.run(outputs1, feed_dict={X1: data})
        mseNew = loss1.eval(feed_dict={outputs1: y_pred, y1: y_test})

        evallog.write("\nTesting Statistic\nIncrease MSE on Saved,{}\n".format(numpy.float32(mseNew) - mseL))

        newPredictions, newStrIRI = get_label_and_iri_from_encoding(y_pred, labels, numConcepts, numRoles)

        write_vector_file(
            "crossValidationFolds/output/predictionSavedKBPipeline[{}].txt".format(n), newStrIRI)

        return distanceEvaluations(evallog, y_pred.shape, newPredictions, trueLabels, newStrIRI, trueIRIs,
                                   numConcepts, numRoles, labels)


def run_nth_time(trainlog, evallog, epochs, learningRate, nthData, n):
    """Runs and collects results from shallow, deep, and flat models for one cycle of the cross validation."""
    evals1 = shallowSystem(int(epochs / 2), learningRate, trainlog, evallog, nthData, n)

    tf.reset_default_graph()

    evals2 = deep_system(epochs, learningRate / 2, trainlog, evallog, nthData, n)

    tf.reset_default_graph()

    evals3 = flat_system(epochs, learningRate / 2, trainlog, evallog, nthData, n)

    tf.reset_default_graph()

    trainlog.close()
    evallog.close()

    return evals1, evals2, evals3


def n_times_cross_validate(n, epochs, learningRate, dataFile):
    # Sets up logging.
    if not os.path.isdir("crossValidationFolds"): os.mkdir("crossValidationFolds")
    if not os.path.isdir("crossValidationFolds/training"): os.mkdir("crossValidationFolds/training")
    if not os.path.isdir("crossValidationFolds/evals"): os.mkdir("crossValidationFolds/evals")
    if not os.path.isdir("crossValidationFolds/saves"): os.mkdir("crossValidationFolds/saves")
    if not os.path.isdir("crossValidationFolds/output"): os.mkdir("crossValidationFolds/output")

    # Gets raw data.
    KB, supports, outputs, encodingMap, labels = convert_data_to_arrays(get_rdf_data(dataFile))

    numConcepts, numRoles = get_concept_and_role_count(labels)

    # Processes data.
    allTheData = cross_validation_split_all_data(n, KB, supports, outputs, encodingMap, labels, numConcepts, numRoles)

    KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, trueLabels, trueIRIs, labelss = allTheData

    # Saves all the data in a file for some reason?
    numpy.savez("saves/{}foldData.npz".format(n), KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, labelss)

    if isinstance(labelss, numpy.ndarray):
        if (labelss.ndim and labelss.size) == 0:
            labelss = None

    evals = numpy.zeros((2, 8), dtype=numpy.float64)
    for i in range(n):
        print("\nCross Validation Fold {}\n\nTraining With {} Data\nTesting With {} Data\n".format(i, "RDF", "RDF"))
        x = run_nth_time(open("crossValidationFolds/training/trainFold[{}].csv".format(i), "w"),
                         open("crossValidationFolds/evals/evalFold[{}].csv".format(i), "w"), epochs, learningRate,
                         (KBs_tests[i], KBs_trains[i], X_trains[i], X_tests[i], y_trains[i], y_tests[i],
                          (labelss[i] if isinstance(labelss, numpy.ndarray) else None), trueLabels[i], trueIRIs[i], labels,
                          numConcepts, numRoles), i)
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


def distanceEvaluations(log, shape, newPredictions, trueLabels, newStrIRI, trueIRIs, numConcepts, numRoles, map):
    levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, F12, F121 = lev_distance(shape, newStrIRI, trueIRIs,
                                                                                            numConcepts, numRoles, map)

    log.write(
        "\nString Distance:\nLevenshtein Distance From Reasoner to Random Data,{}\nLevenshtein Distance From Random to Reasoner Data,{}\nLevenshtein Distance From Reasoner to Predicted Data,{}\nLevenshtein Distance From Prediction to Reasoner Data,{}\n".format(
            levTR2, levRT2, levTN2, levNT2))
    log.write(
        "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
            levTR2 / sizeTrue2, levRT2 / sizeRan2, levTN2 / sizeTrue2, 0 if sizeNew2 == 0 else levNT2 / sizeNew2))

    b = write_evaluation_measures(F12, F121, log)

    custTR, custRT, custTN, custNT, countTrue, countNew, countRan, newEvalInfo, randEvalInfo = custom_distance(shape,
                                                                                                               newPredictions,
                                                                                                               trueLabels,
                                                                                                               numConcepts,
                                                                                                               numRoles)

    log.write(
        "\nCustom Label Distance:\nCustom Distance From Reasoner to Random Data,{}\nCustom Distance From Random to Reasoner Data,{}\nCustom Distance From Reasoner to Predicted Data,{}\nCustom Distance From Predicted to Reasoner Data,{}\n".format(
            custTR, custRT, custTN, custNT))
    log.write(
        "Average Custom Distance From Reasoner to Random Statement,{}\nAverage Custom Distance From Random to Reasoner Statement,{}\nAverage Custom Distance From Reasoner to Predicted Statement,{}\nAverage Custom Distance From Prediction to Reasoner Statement,{}\n".format(
            custTR / countTrue, custRT / countRan, custTN / countTrue, 0 if countNew == 0 else custNT / countNew))

    c = write_evaluation_measures(newEvalInfo, randEvalInfo, log)

    return np.array([np.array([levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, b]),
                    np.array([custTR, custRT, custTN, custNT, countTrue, countNew, countRan, c])])


def custom_distance(sampleShape, newPred, trueLabels, conceptSpace, roleSpace):
    """Finds the number of true and false positives and false negatives.  Also collects distance data which is recorded."""
    rando = create_random_label_predictions(sampleShape, conceptSpace, roleSpace)

    # Combines all the timestep lists together so can compare whole samples.
    flatTrue = [[item for sublist in x for item in sublist] for x in trueLabels]
    flatRand = [[item for sublist in x for item in sublist] for x in rando]
    flatNew = [[item for sublist in x for item in sublist] for x in newPred]

    # [0] = True Positives, [1] = False Positives, [2] = False Negatives
    evalInfoNew = np.array([0, 0, 0])
    evalInfoRandom = np.array([0, 0, 0])

    custTR = 0 # True to Reasoner
    custRT = 0
    custTN = 0 # True to New
    custNT = 0

    countRan = 0
    countTrue = 0
    countNew = 0

    for i in range(len(rando)):  # KB
        for j in range(len(rando[i])):  # Step
            for k in range(len(rando[i][j])):  # Statement
                countRan = countRan + 1

                if len(trueLabels) > i and len(trueLabels[i]) > j and len(trueLabels[i][j]) > k:
                    countTrue = countTrue + 1
                    # Seems to take a predicted axiom and find the one with the smallest distance in the actual
                    # predictions. Uses custom distance formula for the labels. Seems to add all the distances together.
                    custTR = custTR + find_best_pred_match(trueLabels[i][j][k], flatRand[i], conceptSpace, roleSpace)
                    best = find_best_pred_match(rando[i][j][k], flatTrue[i], conceptSpace, roleSpace)
                    if best == 0:
                        evalInfoRandom[0] = evalInfoRandom[0] + 1
                    custRT = custRT + best
                    if len(newPred) > i and len(newPred[i]) > 0:
                        custTN = custTN + find_best_pred_match(trueLabels[i][j][k], flatNew[i], conceptSpace, roleSpace)
                    else:
                        custTN = custTN + custom(conceptSpace, roleSpace, trueLabels[i][j][k], [])

                if len(newPred) > i and len(newPred[i]) > j and len(newPred[i][j]) > k:  # Out of bounds check
                    countNew = countNew + 1
                    if len(trueLabels) > i and len(trueLabels[i]) > 0:
                        best = find_best_pred_match(newPred[i][j][k], flatTrue[i], conceptSpace, roleSpace)
                        if best == 0:
                            evalInfoNew[0] = evalInfoNew[0] + 1
                        custNT = custNT + best
                    else:
                        custNT = custNT + custom(conceptSpace, roleSpace, newPred[i][j][k], [])

    # Calculating False positives.
    evalInfoNew[1] = countNew - evalInfoNew[0]
    evalInfoNew[2] = countTrue - evalInfoNew[0]
    evalInfoRandom[1] = countRan - evalInfoRandom[0]
    evalInfoRandom[2] = countTrue - evalInfoRandom[0]
    return custTR, custRT, custTN, custNT, countTrue, countNew, countRan, evalInfoNew, evalInfoRandom


def lev_distance(shape, newIRIs, trueIRIs, numConcepts, numRoles, map):
    """Calculates levenshtein distance between iri strings."""

    rando = create_random_IRI_predictions(shape, map, numConcepts, numRoles)

    flatTrue = [[item for sublist in x for item in sublist] for x in trueIRIs]
    flatRand = [[item for sublist in x for item in sublist] for x in rando]
    flatNew = [[item for sublist in x for item in sublist] for x in newIRIs]

    evalInfoNew = np.array([0, 0, 0])
    evalInfoRandom = np.array([0, 0, 0])

    levTR = 0
    levRT = 0
    levTN = 0
    levNT = 0

    countRan = 0
    countTrue = 0
    countNew = 0

    for i in range(len(rando)):
        for j in range(len(rando[i])):
            for k in range(len(rando[i][j])):
                countRan = countRan + 1
                if len(trueIRIs) > i and len(trueIRIs[i]) > j and len(trueIRIs[i][j]) > k:  # FOR VERY TRUE STATEMENT
                    countTrue = countTrue + 1
                    levTR = levTR + find_best_match(trueIRIs[i][j][k],
                                                    flatRand[i])  # compare to best match in random and vice versa
                    best = find_best_match(rando[i][j][k], flatTrue[i])
                    if best == 0: evalInfoRandom[0] = evalInfoRandom[0] + 1
                    levRT = levRT + best

                    if (len(newIRIs) > i and len(newIRIs[i]) > 0):
                        levTN = levTN + find_best_match(trueIRIs[i][j][k], flatNew[
                            i])  # if there are predictions for this KB, compare to best match in there
                    else:
                        levTN = levTN + levenshtein(trueIRIs[i][j][k], '')  # otherwise compare with no prediction

                if len(newIRIs) > i and len(newIRIs[i]) > j and len(
                        newIRIs[i][j]) > k:  # FOR EVERY PREDICTION
                    countNew = countNew + 1
                    if (len(trueIRIs) > i and len(trueIRIs[i]) > 0):
                        best = find_best_match(newIRIs[i][j][k], flatTrue[i])
                        if best == 0: evalInfoNew[0] = evalInfoNew[0] + 1
                        levNT = levNT + best  # if there are true values for this KB, compare to best match in there
                    else:
                        levNT = levNT + levenshtein(newIRIs[i][j][k], '')  # otherwise compare with no true value

    evalInfoNew[1] = countNew - evalInfoNew[0]
    evalInfoNew[2] = countTrue - evalInfoNew[0]
    evalInfoRandom[1] = countRan - evalInfoRandom[0]
    evalInfoRandom[2] = countTrue - evalInfoRandom[0]
    return levTR, levRT, levTN, levNT, countTrue, countNew, countRan, evalInfoNew, evalInfoRandom


def find_best_match(statement, reasonerSteps):
    return min(map(partial(levenshtein, statement), reasonerSteps))


def find_best_pred_match(statement, otherKB, conceptSpace, roleSpace):
    return min(map(partial(custom, conceptSpace, roleSpace, statement), otherKB))


# Tested
def levenshtein(s1, s2):
    # https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """Calculates basic Levenshtein edit distance between the given strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# Tested
def custom(numConcepts, numRoles, tupleTriple1, tupleTriple2):
    if len(tupleTriple1) < len(tupleTriple2): return custom(numConcepts, numRoles, tupleTriple2, tupleTriple1)

    dist = 0

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


def get_rdf_data(file):
    """Gets RDF data from json file specified."""
    with open(file) as f:
        data = json.load(f)
        return data


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
    kb, supp, outs, numToStmMap, labelMap = data['kB'], data['supports'], data['outputs'], data['vectorMap'],\
                                            data['labelMap']
    Kb = numpy.array(kb)

    supp = pad_list_of_lists(supp)
    outs = pad_list_of_lists(outs)

    Supp = numpy.zeros((len(supp)), dtype=numpy.ndarray)
    Outs = numpy.zeros((len(outs)), dtype=numpy.ndarray)

    for i in range(len(supp)):
        Supp[i] = numpy.array(supp[i])

    for i in range(len(outs)):
        Outs[i] = numpy.array(outs[i])

    numToStmMap = numpy.array(numToStmMap)

    return Kb, Supp, Outs, numToStmMap, labelMap


def get_concept_and_role_count(labelMap):
    """Finds the number of concepts and roles in a given labelMap.  Roles have negative keys and concepts positive."""
    numConcepts = 0
    numRoles = 0

    for key in labelMap:
        if int(key) > 0:
            numConcepts = numConcepts + 1
        else:
            numRoles = numRoles + 1

    return numConcepts, numRoles


def cross_validation_split_all_data(n, KBs, supports, outputs, encodedMap, labels, numConcepts, numRoles):
    # maxout = None if not isinstance(mouts, numpy.ndarray) else len(max(mouts, key=lambda coll: len(coll))[0])

    # Potentially calculates size of the 3D tensor which will be padded and passed to the LSTM.
    fileShapes1 = [len(supports[0]), len(max(supports, key=lambda coll: len(coll[0]))[0]),
                   len(max(outputs, key=lambda coll: len(coll[0]))[0])]

    print("Repeating KBs")
    # Creates a new 3D tensor where the original KB is copied once per timestep.
    newKBs = numpy.empty([KBs.shape[0], fileShapes1[0], KBs.shape[1]], dtype=float)
    # Goes through for every line of the KB
    for i in range(len(newKBs)):
        for j in range(fileShapes1[0]):
            # The same KB line is copied timestep times in the j's place.
            newKBs[i][j] = KBs[i]

    KBs = newKBs

    print("Shuffling Split Indices")
    # Creates a list of indices up to the length of the # of Batch size?
    indices = list(range(len(KBs)))
    random.shuffle(indices)
    # k is the quotient and m is the remainder
    k, m = divmod(len(indices), n)
    # Setting indices to be equal to a list where each element is a list from the previous indices elements.
    # I don't know why yet.
    indices = list(indices[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    # Shape (numberOfCross, numberInEachCross, numberOfTimeStep, ?, ?)
    # Basically just splitting KBs n times.
    crossKBsTest = numpy.zeros((len(indices), len(indices[0]), len(KBs[0]), len(KBs[0][0])), dtype=float)
    crossSupportsTest = numpy.zeros(
        (len(indices), len(indices[0]), len(supports[0]), fileShapes1[1]), dtype=float)
    crossOutputsTest = numpy.zeros(
        (len(indices), len(indices[0]), len(outputs[0]), fileShapes1[2]), dtype=float)

    # Probably don't need yet.
    # crossErrTest = None if not isinstance(mKBs, numpy.ndarray) else numpy.zeros(
    #     (len(indices), len(indices[0]), len(mouts[0]), maxout), dtype=float)

    # If localMap is provided then create empty crossLabels array.
    if isinstance(encodedMap, numpy.ndarray):
        crossLabels = numpy.empty((len(indices), len(indices[0])), dtype=dict)
    else:
        crossLabels = None

    print("Extracting Test Sets")
    for i in range(len(indices)):
        KBns = []
        for j in range(len(indices[i])):
            crossKBsTest[i][j] = KBs[indices[i][j]]
            crossSupportsTest[i][j] = numpy.hstack([supports[indices[i][j]], numpy.zeros(
                [fileShapes1[0], fileShapes1[1] - len(supports[indices[i][j]][0])])])
            crossOutputsTest[i][j] = numpy.hstack([outputs[indices[i][j]], numpy.zeros(
                [fileShapes1[0], fileShapes1[2] - len(outputs[indices[i][j]][0])])])
            if isinstance(encodedMap, numpy.ndarray):
                crossLabels[i][j] = encodedMap[indices[i][j]]
        write_vector_file("crossValidationFolds/output/originalKBsIn[{}].txt".format(i), numpy.array(KBns))

    crossKBsTrain = numpy.zeros((n, len(KBs) - len(crossKBsTest[0]), len(KBs[0]), len(KBs[0][0])), dtype=float)
    crossOutputsTrain = numpy.zeros((n, len(KBs) - len(crossKBsTest[0]), len(outputs[0]), fileShapes1[2]), dtype=float)
    crossSupportsTrain = numpy.zeros((n, len(KBs) - len(crossKBsTest[0]), len(supports[0]), fileShapes1[1]), dtype=float)

    print("Extracting Train Sets")
    for i in range(len(indices)):
        for j in range(len(crossKBsTrain[i])):
            train = list(set(range(len(KBs))).difference(set(indices[i])))
            random.shuffle(train)
            for k in range(len(train)):
                crossKBsTrain[i][j] = KBs[train[k]]
                crossSupportsTrain[i][j] = numpy.hstack(
                    [supports[train[k]], numpy.zeros([fileShapes1[0], fileShapes1[1] - len(supports[train[k]][0])])])
                crossOutputsTrain[i][j] = numpy.hstack(
                    [outputs[train[k]], numpy.zeros([fileShapes1[0], fileShapes1[2] - len(outputs[train[k]][0])])])

    print("Saving Reasoner Answers")
    nTrueLabels = numpy.empty(n, dtype=numpy.ndarray)
    nTrueIRIs = numpy.empty(n, dtype=numpy.ndarray)
    for i in range(n):
        placeholder, KBn = get_label_and_iri_from_encoding(crossKBsTest[i], labels, numConcepts, numRoles)

        nTrueLabels[i], nTrueIRIs[i] = get_label_and_iri_from_encoding(crossOutputsTest[i], labels, numConcepts, numRoles)

        placeholder, inputs = get_label_and_iri_from_encoding(crossSupportsTest[i], labels, numConcepts, numRoles)

        write_vector_file("crossValidationFolds/output/reasonerCompletion[{}].txt".format(i), nTrueIRIs[i])

        write_vector_file("crossValidationFolds/output/KBsIn[{}].txt".format(i), KBn)

    write_vector_file("crossValidationFolds/output/supports[{}].txt".format(i),inputs)

    return crossKBsTest, crossKBsTrain, crossSupportsTrain, crossSupportsTest, crossOutputsTrain, crossOutputsTest,\
        nTrueLabels, nTrueIRIs, crossLabels


def get_label_and_iri_from_encoding(encodedPredictions, labelMap, numConcepts, numRoles):
    """Gets closest label and iri for a model prediction."""

    labelPredictions = np.zeros((encodedPredictions.shape[0], encodedPredictions.shape[1]), dtype=tuple)
    stringPredictions = np.zeros((encodedPredictions.shape[0], encodedPredictions.shape[1]), dtype=tuple)

    sampleBatchIndex = 0
    for sampleBatch in encodedPredictions:
        timeStepIndex = 0
        for timeStep in sampleBatch:
            labelsForTimeStep = []
            strForTimeStep = []
            tempLabels = []
            tempStr = []

            for item in timeStep:
                intLabel, strIri = convert_encoding_to_label_and_iri(item, labelMap, numConcepts, numRoles)
                tempLabels.append(intLabel)
                tempStr.append(strIri)

                if len(tempLabels) == 3:
                    if len(tempStr) == 3:
                        labelsForTimeStep.append(tuple((tempLabels[0], tempLabels[1], tempLabels[2])))
                        strForTimeStep.append(tuple((tempStr[0], tempStr[1], tempStr[2])))
                        tempLabels = []
                        tempStr = []
                    else:
                        print("Error in get_predicted_label_and_iri_from_encoding: ???")

            labelPredictions[sampleBatchIndex][timeStepIndex] = labelsForTimeStep
            stringPredictions[sampleBatchIndex][timeStepIndex] = strForTimeStep
            timeStepIndex += 1
        sampleBatchIndex += 1
    return labelPredictions, stringPredictions


def convert_encoding_to_label_and_iri(enc, labelMap, numConcepts, numRoles):
    """Converts a float representing an encoding into an int label and its string iri."""
    if (enc > 0):
        label = int(enc * numConcepts)

        # Makes sure it is in range
        if label > numConcepts:
            label = label - 1

        iriStr = labelMap.get(str(label))

        return "C" + str(label), iriStr

    else:
        label = int(enc * numRoles)

        # Makes sure it is in range
        if label < (-1 * numRoles):
            label = label + 1

        iriStr = labelMap.get(str(label))

        return "R" + str(abs(label)), iriStr


def create_random_label_predictions(shape, numConcepts, numRoles):
    """Creates and returns random data integer label data in the same shape as the sample."""
    randLabels = np.zeros(shape=(shape[0], shape[1]), dtype=tuple)

    for sample in range(shape[0]):
        for timeStep in range(shape[1]):
            tempTimeStepLabels = []
            for tup in range(shape[2]):
                tempTimeStepLabels.append(("C" + str(random.randint(1, numConcepts)), "R" +
                                           str(random.randint(1, numRoles)),
                                           "C" + str(random.randint(1, numConcepts))))
            randLabels[sample][timeStep] = tempTimeStepLabels

    return randLabels


def create_random_IRI_predictions(shape, map, numConcepts, numRoles):
    """Creates and returns random string data in the same shape as the sample."""
    randIRI = np.zeros(shape=(shape[0], shape[1]), dtype=tuple)

    for sample in range(shape[0]):
        for timeStep in range(shape[1]):
            tempTimeStepLabels = []
            for tup in range(shape[2]):
                tempTimeStepLabels.append((map.get(str(random.randint(1, numConcepts))), map.get(str(random.randint(1, numRoles))),
                                          map.get(str(random.randint(1, numConcepts)))))
            randIRI[sample][timeStep] = tempTimeStepLabels

    return randIRI


def read_inputs():
    """Collects arguments to be passed into the model."""
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", help="number of epochs for each system", type=int, default=10000)  # 20000
    parser.add_argument("-l", "--learningRate", help="learning rate of each system", type=float, default=0.0001)
    parser.add_argument("-c", "--cross", help="cross validation k", type=int, default=2)  # 10

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

    n_times_cross_validate(n=args.cross, epochs=args.epochs, learningRate=args.learningRate,
                           dataFile="rdfData/gfo.json")