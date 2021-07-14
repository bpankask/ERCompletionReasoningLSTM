import os, sys, shutil, argparse, random, numpy, math
import io
import json
from numpy import array
from functools import partial
import tensorflow as tf
# from tensorflow.contrib import rnn

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Measuring Metrics------------------------------------------------------------------------------------------------------
'''
def precision(TP, FP):
    return 0 if TP == 0 and FP == 0 else TP / (TP + FP)


def recall(TP, FN):
    return 0 if TP == 0 and FN == 0 else TP / (TP + FN)


def F1(precision, recall):
    return 0 if precision == 0 and recall == 0 else 2 * (precision * recall) / (precision + recall)


def writeAccMeasures(F, rF, log):
    TPs, FPs, FNs = F
    pre = precision(TPs, FPs)
    rec = recall(TPs, FNs)
    F = F1(pre, rec)

    log.write(
        "\nPrediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs, FPs, FNs, pre, rec, F))

    x = array([TPs, FPs, FNs, pre, rec, F])

    TPs, FPs, FNs = rF
    pre = precision(TPs, FPs)
    rec = recall(TPs, FNs)
    F = F1(pre, rec)

    log.write(
        "\nRandom Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs, FPs, FNs, pre, rec, F))

    return array([x, array([TPs, FPs, FNs, pre, rec, F])])


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


def writeFinalAverageData(result, log):
    """Writes all final data to a file after it has been averaged."""
    levTR, levRT, levTN, levNT, sizeTrue, sizeNew, sizeRan, (
    (TPs, FPs, FNs, pre, rec, F), (rTPs, rFPs, rFNs, rpre, rrec, rF)) = result[0]
    levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, (
    (TPs1, FPs1, FNs1, pre1, rec1, F1), (rTPs1, rFPs1, rFNs1, rpre1, rrec1, rF1)) = result[1]
    custTR, custRT, custTN, custNT, countTrue, countNew, countRan, (
    (TPs2, FPs2, FNs2, pre2, rec2, F2), (rTPs2, rFPs2, rFNs2, rpre2, rrec2, rF2)) = result[2]
    log.write(
        "\nNo Nums\nAverage Levenshtein Distance From Reasoner to Random Data,{}\nAverage Levenshtein Distance From Random to Reasoner Data,{}\nAverage Levenshtein Distance From Reasoner to Predicted Data,{}\nAverage Levenshtein Distance From Prediction to Reasoner Data,{}\n".format(
            levTR, levRT, levTN, levNT))
    log.write(
        "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
            levTR / sizeTrue, levRT / sizeRan, levTN / sizeTrue, levNT / sizeNew))

    log.write(
        "\nAverage Prediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs, FPs, FNs, pre, rec, F))
    log.write(
        "\nAverage Random Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            rTPs, rFPs, rFNs, rpre, rrec, rF))

    log.write(
        "\nNums\nAverage Levenshtein Distance From Reasoner to Random Data,{}\nAverage Levenshtein Distance From Random to Reasoner Data,{}\nAverage Levenshtein Distance From Reasoner to Predicted Data,{}\nAverage Levenshtein Distance From Prediction to Reasoner Data,{}\n".format(
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
        "\nCustom\nAverage Custom Distance From Reasoner to Random Data,{}\nAverage Custom Distance From Random to Reasoner Data,{}\nAverage Custom Distance From Reasoner to Predicted Data,{}\nAverage Custom Distance From Predicted to Reasoner Data,{}\n".format(
            custTR, custRT, custTN, custNT))
    log.write(
        "Average Custom Distance From Reasoner to Random Statement,{}\nAverage Custom Distance From Random to Reasoner Statement,{}\nAverage Custom Distance From Reasoner to Predicted Statement,{}\nAverage Custom Distance From Prediction to Reasoner Statement,{}\n".format(
            custTR / countTrue, custRT / countRan, custTN / countTrue, custNT / countNew))

    log.write(
        "\nAverage Prediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs2, FPs2, FNs2, pre2, rec2, F2))
    log.write(
        "\nAverage Random Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(rTPs2, rFPs2, rFNs2, rpre2, rrec2, rF2))


def writeVectorFileWithMap(filename, vector, mapping):
    file = open(filename, "w")
    for i in range(len(vector)):
        print(mapping[i])
        file.write("Trial: {}\n".format(i))
        for j in range(len(vector[i])):
            file.write("\tStep: {}\n".format(j))
            for k in range(len(vector[i][j])):
                file.write("\t\t{}\n".format(vector[i][j][k]))
        file.write("\n")
    file.close()


def writeVectorFile(filename, vector):
    file = io.open(filename, "w", encoding='utf-8')
    for i in range(len(vector)):
        file.write("Trial: {}\n".format(i))
        for j in range(len(vector[i])):
            file.write("\tStep: {}\n".format(j))
            for k in range(len(vector[i][j])):
                file.write("\t\t{}\n".format(vector[i][j][k]))
        file.write("\n")
    file.close()


def trainingStats(log, mseNew, mse0, mseL):
    log.write(
        "Training Statistics\nPrediction Mean Squared Error,{}\nLearned Reduction MSE,{}\nIncrease MSE on Test,{}\nTraining Percent Change MSE,{}\n".format(
            numpy.float32(mseNew), mse0 - mseL, numpy.float32(mseNew) - mseL, (mseL - mse0) / mse0 * 100))


#Helpers for learning models--------------------------------------------------------------------------------------------

def pad(arr, maxlen1=0, maxlen2=0):
    for i in range(0, len(arr)):
        if len(arr[i]) > maxlen1: maxlen1 = len(arr[i])
        for j in range(0, len(arr[i])):
            if len(arr[i][j]) > maxlen2: maxlen2 = len(arr[i][j])

    newarr = numpy.zeros(shape=(len(arr), maxlen1, maxlen2), dtype=float)
    for i in range(0, len(arr)):
        for j in range(0, len(arr[i])):
            for k in range(0, len(arr[i][j])):
                newarr[i][j][k] = arr[i][j][k]

    return newarr


def splitTensors(inputs, outputs, size):
    inTest, inTrain = numpy.split(inputs, [int(len(inputs) * size)])
    outTest, outTrain = numpy.split(outputs, [int(len(outputs) * size)])
    return inTrain, inTest, outTrain, outTest


def repeatAndSplitKBs(kbs, steps, splitSize):
    newKBs = numpy.empty([kbs.shape[0], steps, kbs.shape[1]], dtype=numpy.float32)
    for i in range(len(newKBs)):
        for j in range(steps):
            newKBs[i][j] = kbs[i]
    return numpy.split(newKBs, [int(len(newKBs) * splitSize)])

#LSTM Models------------------------------------------------------------------------------------------------------------

# def shallowSystem(n_epochs0, learning_rate0, trainlog, evallog, conceptSpace, roleSpace, allTheData, syn, mix, n):
#     KBs_test, KBs_train, X_train, X_test, y_train, y_test, truePreds, trueStatements, labels, errPreds, errStatements = allTheData
# 
#     trainlog.write("Piecewise LSTM Part One\nEpoch,Mean Squared Error,Root Mean Squared Error\n")
#     evallog.write("Piecewise LSTM Part One\n")
#     print("")
# 
#     n_neurons0 = X_train.shape[2]
# 
#     X0 = tf.compat.v1.placeholder(tf.float32, shape=[None, KBs_train.shape[1], KBs_train.shape[2]])
#     y0 = tf.compat.v1.placeholder(tf.float32, shape=[None, X_train.shape[1], X_train.shape[2]])
# 
#     outputs0, states0 = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(num_units=n_neurons0), X0, dtype=tf.float32)
# 
#     loss0 = tf.losses.mean_squared_error(y0, outputs0)
#     optimizer0 = tf.train.AdamOptimizer(learning_rate=learning_rate0)
#     training_op0 = optimizer0.minimize(loss0)
# 
#     # saver = tf.train.Saver()
# 
#     init0 = tf.global_variables_initializer()
# 
#     with tf.Session() as sess:
#         init0.run()
#         mse0 = 0
#         mseL = 0
#         for epoch in range(n_epochs0):
#             print("Piecewise System\tEpoch: {}".format(epoch))
#             ynew, a = sess.run([outputs0, training_op0], feed_dict={X0: KBs_train, y0: X_train})
#             mse = loss0.eval(feed_dict={outputs0: ynew, y0: X_train})
#             if epoch == 0: mse0 = mse
#             if epoch == n_epochs0 - 1: mseL = mse
#             trainlog.write("{},{},{}\n".format(epoch, mse, math.sqrt(mse)))
#             if mse < 0.00001:
#                 mseL = mse
#                 break
# 
#         print("\nTesting first half\n")
# 
#         y_pred = sess.run(outputs0, feed_dict={X0: KBs_test, y0: X_test})
#         mseNew = loss0.eval(feed_dict={outputs0: y_pred, y0: X_test})
#         newPreds, newStatements = vecToStatements(y_pred, conceptSpace, roleSpace)
# 
#         trainingStats(evallog, mseNew, mse0, mseL)
# 
#         writeVectorFile("crossValidationFolds/{}output/learnedSupportsP[{}].txt".format("" if syn else "sn", n),
#                         newStatements)
# 
#         numpy.savez("crossValidationFolds/{}saves/halfwayData[{}]".format("" if syn else "s", n), y_pred)
# 
#         # saver.save(sess,"{}{}saves/firstHalfModel[{}]".format("" if n == 1 else "crossValidationFolds/","" if syn else "s",n))
# 
#     tf.reset_default_graph()
# 
#     trainlog.write("Piecewise LSTM Part Two\nEpoch,Mean Squared Error,Root Mean Squared Error\n")
#     evallog.write("\nPiecewise LSTM Part Two\n")
# 
#     n_neurons1 = y_train.shape[2]
# 
#     X1 = tf.placeholder(tf.float32, shape=[None, X_train.shape[1], X_train.shape[2]])
#     y1 = tf.placeholder(tf.float32, shape=[None, y_train.shape[1], y_train.shape[2]])
# 
#     outputs1, states1 = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(num_units=n_neurons1), X1, dtype=tf.float32)
# 
#     loss1 = tf.losses.mean_squared_error(y1, outputs1)
#     optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate0)
#     training_op1 = optimizer1.minimize(loss1)
# 
#     # saver = tf.train.Saver()
# 
#     init1 = tf.global_variables_initializer()
# 
#     with tf.Session() as sess:
#         init1.run()
#         mse0 = 0
#         mseL = 0
#         for epoch in range(n_epochs0):
#             print("Piecewise System\tEpoch: {}".format(epoch + n_epochs0))
#             ynew, a = sess.run([outputs1, training_op1], feed_dict={X1: X_train, y1: y_train})
#             mse = loss1.eval(feed_dict={outputs1: ynew, y1: y_train})
#             if epoch == 0: mse0 = mse
#             if epoch == n_epochs0 - 1: mseL = mse
#             trainlog.write("{},{},{}\n".format(epoch, mse, math.sqrt(mse)))
#             if mse < 0.00001:
#                 mseL = mse
#                 break
# 
#         print("\nTesting second half")
#         y_pred = sess.run(outputs1, feed_dict={X1: X_test})
#         mseNew = loss1.eval(feed_dict={outputs1: y_pred, y1: y_test})
# 
#         trainingStats(evallog, mseNew, mse0, mseL)
# 
#         print("\nEvaluating Result")
# 
#         evallog.write("\nReasoner Support Test Data Evaluation\n")
# 
#         newPreds, newStatements = vecToStatements(y_pred, conceptSpace, roleSpace)
# 
#         writeVectorFile(
#             "crossValidationFolds/{}output/predictedOutLeftOverSupportTest[{}].txt".format("" if syn else "sn", n),
#             newStatements)
# 
#         evals = distanceEvaluations(evallog, y_pred.shape, newPreds, truePreds, newStatements, trueStatements,
#                                     conceptSpace, roleSpace, syn, mix, False, False)
# 
#         data = numpy.load("crossValidationFolds/{}saves/halfwayData[{}].npz".format("" if syn else "s", n),
#                           allow_pickle=True)
#         data = data['arr_0']
# 
#         evallog.write("\nSaved Test Data From Previous LSTM Evaluation\n")
# 
#         y_pred = sess.run(outputs1, feed_dict={X1: data})
#         mseNew = loss1.eval(feed_dict={outputs1: y_pred, y1: y_test})
# 
#         evallog.write("\nTesting Statistic\nIncrease MSE on Saved,{}\n".format(numpy.float32(mseNew) - mseL))
# 
#         newPreds, newStatements = vecToStatementsWithLabels(y_pred, conceptSpace, roleSpace, labels) if (
#                     not mix and not syn) else vecToStatements(y_pred, conceptSpace, roleSpace)
# 
#         writeVectorFile(
#             "crossValidationFolds/{}output/predictionSavedKBPipeline[{}].txt".format("" if syn else "sn", n),
#             newStatements)
# 
#         if (not mix and not syn):
#             newPreds, newStatements = vecToStatements(y_pred, conceptSpace, roleSpace)
# 
#         # saver.save(sess,"{}{}saves/secondHalfModel[{}]".format("" if n == 1 else "crossValidationFolds/","" if syn else "s",n))
# 
#         return distanceEvaluations(evallog, y_pred.shape, newPreds, truePreds, newStatements, trueStatements,
#                                    conceptSpace, roleSpace, syn, mix, errPreds, errStatements)
# 
# 
# def deepSystem(n_epochs2, learning_rate2, trainlog, evallog, conceptSpace, roleSpace, allTheData, syn, mix, n):
#     KBs_test, KBs_train, X_train, X_test, y_train, y_test, truePreds, trueStatements, labels, errPreds, errStatements = allTheData
# 
#     trainlog.write("Deep LSTM\nEpoch,Mean Squared Error,Root Mean Squared Error\n")
#     evallog.write("\nDeep LSTM\n\n")
#     print("")
# 
#     X0 = tf.placeholder(tf.float32, shape=[None, KBs_train.shape[1], KBs_train.shape[2]])
#     y1 = tf.placeholder(tf.float32, shape=[None, y_train.shape[1], y_train.shape[2]])
# 
#     outputs2, states2 = tf.nn.dynamic_rnn(tf.contrib.rnn.MultiRNNCell(
#         [tf.contrib.rnn.LSTMCell(num_units=X_train.shape[2]), tf.contrib.rnn.LSTMCell(num_units=y_train.shape[2])]), X0,
#                                           dtype=tf.float32)
# 
#     loss2 = tf.losses.mean_squared_error(y1, outputs2)
#     optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate2)
#     training_op2 = optimizer2.minimize(loss2)
# 
#     # saver = tf.train.Saver()
# 
#     init2 = tf.global_variables_initializer()
# 
#     with tf.Session() as sess:
#         init2.run()
#         mse0 = 0
#         mseL = 0
#         for epoch in range(n_epochs2):
#             print("Deep System\t\tEpoch: {}".format(epoch))
#             ynew, a = sess.run([outputs2, training_op2], feed_dict={X0: KBs_train, y1: y_train})
#             mse = loss2.eval(feed_dict={outputs2: ynew, y1: y_train})
#             if epoch == 0: mse0 = mse
#             if epoch == n_epochs2 - 1: mseL = mse
#             trainlog.write("{},{},{}\n".format(epoch, mse, math.sqrt(mse)))
#             if mse < 0.0001:
#                 mseL = mse
#                 break
# 
#         print("\nEvaluating Result\n")
# 
#         y_pred = sess.run(outputs2, feed_dict={X0: KBs_test})
#         mseNew = loss2.eval(feed_dict={outputs2: y_pred, y1: y_test})
# 
#         trainingStats(evallog, mseNew, mse0, mseL)
# 
#         evallog.write("\nTest Data Evaluation\n")
# 
#         newPreds, newStatements = vecToStatementsWithLabels(y_pred, conceptSpace, roleSpace, labels) if (
#                     not mix and not syn) else vecToStatements(y_pred, conceptSpace, roleSpace)
# 
#         writeVectorFile(
#             "crossValidationFolds/{}output/predictionDeepArchitecture[{}].txt".format("" if syn else "sn", n),
#             newStatements)
# 
#         if (not mix and not syn):
#             newPreds, newStatements = vecToStatements(y_pred, conceptSpace, roleSpace)
# 
#         # saver.save(sess,"{}{}saves/deepModel[{}]".format("" if n == 1 else "crossValidationFolds/","" if syn else "s",n))
# 
#         return distanceEvaluations(evallog, y_pred.shape, newPreds, truePreds, newStatements, trueStatements,
#                                    conceptSpace, roleSpace, syn, mix, errPreds, errStatements)


def flatSystem(n_epochs2, learning_rate2, trainlog, evallog, conceptSpace, roleSpace, allTheData, syn, mix, n):
    """"The base line recurrent model for comparison with other rnn models."""
    KBs_test, KBs_train, X_train, X_test, y_train, y_test, truePreds, trueStatements, labels, errPreds, errStatements = allTheData

    trainlog.write("Flat LSTM\nEpoch,Mean Squared Error,Root Mean Squared Error\n")
    evallog.write("\nFlat LSTM\n\n")
    print("")

    X0 = tf.placeholder(tf.float32, shape=[None, KBs_train.shape[1], KBs_train.shape[2]])
    y1 = tf.placeholder(tf.float32, shape=[None, y_train.shape[1], y_train.shape[2]])

    outputs2, states2 = tf.nn.dynamic_rnn(tf.contrib.rnn.LSTMCell(num_units=y_train.shape[2]), X0, dtype=tf.float32)

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

        trainingStats(evallog, mseNew, mse0, mseL)

        evallog.write("\nTest Data Evaluation\n")

        newPreds, newStatements = vecToStatementsWithLabels(y_pred, conceptSpace, roleSpace, labels) if (
                    not mix and not syn) else vecToStatements(y_pred, conceptSpace, roleSpace)

        writeVectorFile(
            "crossValidationFolds/{}output/predictionFlatArchitecture[{}].txt".format("" if syn else "sn", n),
            newStatements)

        if (not mix and not syn):
            newPreds, newStatements = vecToStatements(y_pred, conceptSpace, roleSpace)

        # saver.save(sess,"{}{}saves/deepModel[{}]".format("" if n == 1 else "crossValidationFolds/","" if syn else "s",n))

        return distanceEvaluations(evallog, y_pred.shape, newPreds, truePreds, newStatements, trueStatements,
                                   conceptSpace, roleSpace, syn, mix, errPreds, errStatements)


#Important for Running--------------------------------------------------------------------------------------------------

def runOnce(trainlog, evallog, epochs, learningRate, conceptSpace, roleSpace, syn, mix):
    if syn:
        if not os.path.isdir("output"): os.mkdir("output")
        KBs, supports, output = getSynDataFromFile('saves/data.npz')
        if mix:
            sKBs, ssupports, soutput, localMaps, stats = getSnoDataFromFile('ssaves/data.npz')
            allTheData = formatDataSyn2Sno(trainlog, conceptSpace, roleSpace, KBs, supports, output, sKBs, ssupports,
                                           soutput, localMaps, stats)
        else:
            allTheData = formatDataSynth(trainlog, conceptSpace, roleSpace, KBs, supports, output)
    else:
        if not os.path.isdir("snoutput"): os.mkdir("snoutput")
        KBs, supports, output, localMaps, stats = getSnoDataFromFile('ssaves/data.npz')
        if mix:
            sKBs, ssupports, soutput = getSynDataFromFile('saves/data.npz')
            allTheData = formatDataSno2Syn(trainlog, conceptSpace, roleSpace, KBs, supports, output, sKBs, ssupports,
                                           soutput, localMaps, stats)
        else:
            allTheData = formatDataSno(trainlog, conceptSpace, roleSpace, KBs, supports, output, localMaps, stats)

    shallowSystem(int(epochs / 2), learningRate, trainlog, evallog, conceptSpace, roleSpace, allTheData, syn, mix, 1)

    tf.reset_default_graph()

    deepSystem(epochs, learningRate / 2, trainlog, evallog, conceptSpace, roleSpace, allTheData, syn, mix, 1)

    tf.reset_default_graph()

    flatSystem(epochs, learningRate / 2, trainlog, evallog, conceptSpace, roleSpace, allTheData, syn, mix, 1)

    trainlog.close()
    evallog.close()

    print("\nDone")


def runNthTime(trainlog, evallog, epochs, learningRate, conceptSpace, roleSpace, nthData, syn, mix, n):
    """Runs and collects results from shallow, deep, and flat models for one cycle of the cross validation."""
    evals1 = shallowSystem(int(epochs / 2), learningRate, trainlog, evallog, conceptSpace, roleSpace, nthData, syn, mix,
                           n)

    tf.reset_default_graph()

    evals2 = deepSystem(epochs, learningRate / 2, trainlog, evallog, conceptSpace, roleSpace, nthData, syn, mix, n)

    tf.reset_default_graph()

    evals3 = flatSystem(epochs, learningRate / 2, trainlog, evallog, conceptSpace, roleSpace, nthData, syn, mix, n)

    tf.reset_default_graph()

    trainlog.close()
    evallog.close()

    if mix:
        return evals1[0], evals1[1], evals2[0], evals2[1], evals3[0], evals3[1]
    else:
        return evals1, evals2, evals3
'''

def nTimesCrossValidate(n, epochs, learningRate):
    if not os.path.isdir("crossValidationFolds"): os.mkdir("crossValidationFolds")
    if not os.path.isdir("crossValidationFolds/training"): os.mkdir("crossValidationFolds/training")
    if not os.path.isdir("crossValidationFolds/evals"): os.mkdir("crossValidationFolds/evals")
    if not os.path.isdir("crossValidationFolds/saves"): os.mkdir("crossValidationFolds/saves")

# If else statements to decide where to pull the data from. Synthetic vs actual
    if os.path.isfile("saves/{}foldData.npz".format(n):
        data = numpy.load("{}saves/{}foldData{}{}.npz".format("" if syn else "s", n, "Mixed" if mix else "",
                                                              "Err[{}]".format(str(pert)) if pert >= 0 else ""), allow_pickle=True)
        
        allTheData = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'], data[
            'arr_6'], data['arr_7'], data['arr_8'], data['arr_9'], data['arr_10']
    else: # Actual Data
        if pert >= 0 and os.path.isfile("ssaves/messData{}.npz".format(pert)):
            data = numpy.load("ssaves/messData{}.npz".format(pert), allow_pickle=True)
            KBs, supports, outputs, localMaps, stats, mKBs, mouts = data['arr_0'], data['arr_1'], data['arr_2'], data[
                'arr_3'], data['arr_4'], data['arr_5'], data['arr_6']
        else:
            KBs, supports, outputs, localMaps, stats = getSnoDataFromFile(
                'ssaves/data.npz')  # !!!!!!!!!!!This is probably where I would put my stuff!!!!!!!!!!!!!!!!
            mKBs = None
            mouts = None
        # Creates labels for roles and concepts.
        labels = collapseLabelMap(localMaps, stats[0][2], stats[1][2], stats[4][1])
        if mix and pert == 0:
            sKBs, ssupports, soutputs = getSynDataFromFile('saves/data.npz')
            allTheData = crossValidationSplitAllData(n, KBs, supports, outputs, sKBs, ssupports, soutputs, labels, mKBs,
                                                     mouts, conceptSpace, roleSpace, syn, mix, pert)
        else:
            allTheData = crossValidationSplitAllData(n, KBs, supports, outputs, None, None, None, labels, mKBs, mouts,
                                                     conceptSpace, roleSpace, syn, mix, pert)

   #crossKBsTest, crossKBsTrain, crossSupportsTrain, crossSupportsTest, crossOutputsTrain, crossOutputsTest, nTruePreds, nTrueStatements, crossLabels, nErrsPreds, nErrStatements
    KBs_tests,    KBs_trains,    X_trains,           X_tests,           y_trains,          y_tests,          truePredss, trueStatementss, labelss,     nErrsPreds, nErrStatements = allTheData
    # Saves all the data in a file for some reason.
    numpy.savez("{}saves/{}foldData{}{}.npz".format("" if syn else "s", n, "Mixed" if mix else "",
                                                    "Err[{}]".format(str(pert)) if pert >= 0 else ""), KBs_tests,
                KBs_trains, X_trains, X_tests, y_trains, y_tests, truePredss, trueStatementss, labelss, nErrsPreds,
                nErrStatements)

    if isinstance(labelss, numpy.ndarray):
        if (labelss.ndim and labelss.size) == 0:
            labelss = None

    evals = numpy.zeros(((6, 3, 8) if mix else ((3, 3, 9) if pert >= 0 else (3, 3, 8))), dtype=numpy.float64)
    for i in range(n):
        print("\nCross Validation Fold {}\n\nTraining With {} Data\nTesting With {} Data\n".format(i,
                                                                                                   "Synthetic" if syn else "SNOMED",
                                                                                                   "Synthetic" if syn else "SNOMED"))
        x = runNthTime(open("crossValidationFolds/training/trainFold[{}].csv".format(i), "w"),
                       open("crossValidationFolds/evals/evalFold[{}].csv".format(i), "w"), epochs, learningRate,
                       conceptSpace, roleSpace, (
                       KBs_tests[i], KBs_trains[i], X_trains[i], X_tests[i], y_trains[i], y_tests[i], truePredss[i],
                       trueStatementss[i], (labelss[i] if isinstance(labelss, numpy.ndarray) else None), nErrsPreds[i],
                       nErrStatements[i]), syn, mix, i)
        evals = evals + x
    evals = evals / n

    print("Summarizing All Results")

    avgResult = evals.tolist()

    log = open("crossValidationFolds/evalAllFolds[avg].csv", "w")

    if mix:

        log.write("Trained With {} Data\nTested With {} Data\n\nRegular Distances\n\nPiecewise System\n".format(
            "Synthetic" if syn else "SNOMED", "Synthetic" if syn else "SNOMED"))

        writeFinalAverageData(avgResult[0], log)

        log.write("\nDeep System\n")

        writeFinalAverageData(avgResult[1], log)

        log.write("\nFlat System\n")

        writeFinalAverageData(avgResult[2], log)

        log.write("\n\nDistance Ignoring Prediction Gaps\n\nPiecewise System\n")

        writeFinalAverageData(avgResult[3], log)

        log.write("\nDeep System\n")

        writeFinalAverageData(avgResult[4], log)

        log.write("\nFlat System\n")

        writeFinalAverageData(avgResult[5], log)

    else:

        log.write("Piecewise System\n")

        # writeFinalAverageDataMess(avgResult[0],log)
        writeFinalAverageData(avgResult[0], log)

        log.write("\nDeep System\n")

        # writeFinalAverageDataMess(avgResult[1],log)
        writeFinalAverageData(avgResult[1], log)

        log.write("\nFlat System\n")

        # writeFinalAverageDataMess(avgResult[2],log)
        writeFinalAverageData(avgResult[2], log)

    log.close()

    print("\nDone")

    return avgResult

#Data manipulation------------------------------------------------------------------------------------------------------
def crossValidationSplitAllData(n, KBs, supports, outputs, sKBs, ssupports, soutputs, localMaps, mKBs, mouts,
                                conceptSpace, roleSpace, syn, mix, pert):
    maxout = None if not isinstance(mouts, numpy.ndarray) else len(max(mouts, key=lambda coll: len(coll))[0])

    # Potentially calculates size of the 3D tensor which will be padded and passed to the LSTM.
    fileShapes1 = [len(supports[0]), len(max(supports, key=lambda coll: len(coll[0]))[0]),
                   len(max(outputs, key=lambda coll: len(coll[0]))[0])]
    fileShapes2 = [len(ssupports[0]), len(max(ssupports, key=lambda coll: len(coll[0]))[0]),
                   len(max(soutputs, key=lambda coll: len(coll[0]))[0])] if mix else [0, 0, 0]

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
        (len(indices), len(indices[0]), len(supports[0]), fileShapes1[1] if not mix else fileShapes2[1]), dtype=float)
    crossOutputsTest = numpy.zeros(
        (len(indices), len(indices[0]), len(outputs[0]), fileShapes1[2] if not mix else fileShapes2[2]), dtype=float)
    crossErrTest = None if not isinstance(mKBs, numpy.ndarray) else numpy.zeros(
        (len(indices), len(indices[0]), len(mouts[0]), maxout), dtype=float)

    # If localMap is provided then create empty crossLabels array.
    if isinstance(localMaps, numpy.ndarray):
        crossLabels = numpy.empty((len(indices), len(indices[0])), dtype=dict)
    else:
        crossLabels = None

    print("Extracting Test Sets")
    for i in range(len(indices)):
        KBns = []
        for j in range(len(indices[i])):
            if not mix:
                if pert >= 0.0 and isinstance(mKBs, numpy.ndarray):
                    crossKBsTest[i][j] = mKBs[indices[i][j]]
                    placeholder, KBn = vecToStatement(KBs[indices[i][j]], conceptSpace, roleSpace)
                    KBns.append(KBn)
                    crossErrTest[i][j] = numpy.hstack(
                        [mouts[indices[i][j]], numpy.zeros([fileShapes1[0], maxout - len(mouts[indices[i][j]][0])])])
                else:
                    crossKBsTest[i][j] = KBs[indices[i][j]]
                crossSupportsTest[i][j] = numpy.hstack([supports[indices[i][j]], numpy.zeros(
                    [fileShapes1[0], fileShapes1[1] - len(supports[indices[i][j]][0])])])
                crossOutputsTest[i][j] = numpy.hstack([outputs[indices[i][j]], numpy.zeros(
                    [fileShapes1[0], fileShapes1[2] - len(outputs[indices[i][j]][0])])])
            else: # Mixed values
                crossKBsTest[i][j] = sKBs[indices[i][j]]
                crossSupportsTest[i][j] = numpy.hstack([ssupports[indices[i][j]], numpy.zeros(
                    [fileShapes2[0], fileShapes2[1] - len(ssupports[indices[i][j]][0])])])
                crossOutputsTest[i][j] = numpy.hstack([soutputs[indices[i][j]], numpy.zeros(
                    [fileShapes2[0], fileShapes2[2] - len(soutputs[indices[i][j]][0])])])
            if isinstance(localMaps, numpy.ndarray):
                crossLabels[i][j] = localMaps[indices[i][j]]
        writeVectorFile("crossValidationFolds/{}output/originalKBsIn[{}].txt".format("sn" if not syn else "", i),
                        array(KBns))

    crossKBsTrain = numpy.zeros((n, len(KBs) - len(crossKBsTest[0]), len(KBs[0]), len(KBs[0][0])), dtype=float)
    crossOutputsTrain = numpy.zeros((n, len(KBs) - len(crossKBsTest[0]), len(outputs[0]), fileShapes1[2]), dtype=float)
    crossSupportsTrain = numpy.zeros((n, len(KBs) - len(crossKBsTest[0]), len(supports[0]), fileShapes1[1]),
                                     dtype=float)

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
    nTruePreds = numpy.empty(n, dtype=numpy.ndarray)
    nTrueStatements = numpy.empty(n, dtype=numpy.ndarray)
    nErrsPreds = numpy.empty(n, dtype=numpy.ndarray)
    nErrStatements = numpy.empty(n, dtype=numpy.ndarray)
    for i in range(n):
        if not syn or (syn and mix):
            placeholder, KBn = vecToStatementsWithLabels(crossKBsTest[i], conceptSpace, roleSpace, crossLabels[i])
            placeholder, nTrueStatementsLabeled = vecToStatementsWithLabels(crossOutputsTest[i], conceptSpace,
                                                                            roleSpace, crossLabels[i])
            placeholder, inputs = vecToStatementsWithLabels(crossSupportsTest[i], conceptSpace, roleSpace,
                                                            crossLabels[i])

            writeVectorFile(
                "crossValidationFolds/{}output/reasonerCompletion[{}].txt".format("sn" if not syn else "", i),
                nTrueStatementsLabeled)

            nTruePreds[i], nTrueStatements[i] = vecToStatements(crossOutputsTest[i], conceptSpace, roleSpace)
            if pert >= 0:
                nErrsPreds[i], nErrStatements[i] = vecToStatements(crossErrTest[i], conceptSpace, roleSpace)
                writeVectorFile(
                    "crossValidationFolds/{}output/ruinedCompletion[{}].txt".format("sn" if not syn else "", i),
                    nErrStatements[i])
        else:
            placeholder, KBn = vecToStatements(crossKBsTest[i], conceptSpace, roleSpace)
            nTruePreds[i], nTrueStatements[i] = vecToStatements(crossOutputsTest[i], conceptSpace, roleSpace)
            if pert >= 0:
                nErrsPreds[i], nErrStatements[i] = vecToStatements(crossErrTest[i], conceptSpace, roleSpace)
                writeVectorFile(
                    "crossValidationFolds/{}output/ruinedCompletion[{}].txt".format("sn" if not syn else "", i),
                    nErrStatements[i])
            placeholder, inputs = vecToStatements(crossSupportsTest[i], conceptSpace, roleSpace)

            writeVectorFile(
                "crossValidationFolds/{}output/reasonerCompletion[{}].txt".format("sn" if not syn else "", i),
                nTrueStatements[i])

        writeVectorFile("crossValidationFolds/{}output/{}KBsIn[{}].txt".format("sn" if not syn else "",
                                                                               "Messed" if pert >= 0 else "", i), KBn)
        # writeVectorFile("crossValidationFolds/{}output/supports[{}].txt".format("sn" if not syn else "",i),inputs)

    return crossKBsTest, crossKBsTrain, crossSupportsTrain, crossSupportsTest, crossOutputsTrain, crossOutputsTest, nTruePreds, nTrueStatements, crossLabels, nErrsPreds, nErrStatements

def getRDFData(file):
    with open(file) as f:
        data = json.load(f)
        kb = data['kB']
        supp = data['supports']
        outs = data['outputs']
        map = data['vectorMap']
        return kb, supp, outs, map

def readInputs():
    """Collects arguments to be passed into the model"""
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", help="number of epochs for each system", type=int, default=10)  # 20000
    parser.add_argument("-l", "--learningRate", help="learning rate of each system", type=float, default=0.0001)
    parser.add_argument("-s", "--snomed", help="use SNOMED dataset", action="store_true", default=True)  # False
    parser.add_argument("-m", "--mix", help="use test set from different souce than train", action="store_true",
                        default=False)  # True
    parser.add_argument("-c", "--cross", help="cross validation k", type=int, default=2)  # 10
    parser.add_argument("-p", "--perturb", help="disturb each kb for comparison", type=float, default=-1.0)

    args = parser.parse_args()

    if args.mix and args.perturb >= 0:
        raise IOError("Can't use two test sets simultaneously")

    if args.epochs and args.epochs < 2:
        raise ValueError("Try a bigger number maybe!")

    if args.cross:
        if args.cross < 1:
            raise ValueError("K fold Cross Validation works better with k greater than 1")

    return args


if __name__ == "__main__":

    KB, Supp, Outs, Map = getRDFData("rdfData/test1.1.json")


    args = readInputs()

    tf.compat.v1.disable_eager_execution()

    nTimesCrossValidate(n=args.cross, epochs=args.epochs, learningRate=args.learningRate)
