# def create_random_label_predictions(shape, numConcepts, numRoles):
#     """Creates and returns random data integer label data in the same shape as the sample."""
#     randLabels = np.zeros(shape=(shape[0], shape[1]), dtype=tuple)
#
#     for sample in range(shape[0]):
#         for timeStep in range(shape[1]):
#             tempTimeStepLabels = []
#             for tup in range(shape[2]):
#                 tempTimeStepLabels.append((get_random_label(numConcepts, numRoles),
#                                            "R" + str(random.randint(0, numRoles)),
#                                            get_random_label(numConcepts, numRoles)))
#             randLabels[sample][timeStep] = tempTimeStepLabels
#
#     return randLabels


# def get_random_label(numConcepts, numRoles):
#     """Returns a random string label for use in the subject or object place in a triple."""
#     x = random.randint(0, numConcepts)
#     if x == 0:
#         return "R0"
#     elif x <= numRoles:
#         temp = random.randint(0, 1)
#         if temp == 0:
#             return "C" + str(x)
#         else:
#             return "R" + str(x)
#     else:
#         return "C" + str(x)


# def create_random_IRI_predictions(shape, map, numConcepts, numRoles):
#     """Creates and returns random string data in the same shape as the sample."""
#     randIRI = np.zeros(shape=(shape[0], shape[1]), dtype=tuple)
#
#     for sample in range(shape[0]):
#         for timeStep in range(shape[1]):
#             tempTimeStepLabels = []
#             for tup in range(shape[2]):
#                 tempTimeStepLabels.append((map.get(str(random.randint(1, numConcepts))), map.get(str(random.randint(1, numRoles))),
#                                           map.get(str(random.randint(1, numConcepts)))))
#             randIRI[sample][timeStep] = tempTimeStepLabels
#
#     return randIRI

# def get_concept_and_role_count(labelMap):
#     """Finds the number of concepts and roles in a given labelMap.  Roles have negative keys and concepts positive."""
#     numConcepts = 0
#     numRoles = 0
#
#     for key in labelMap:
#         if int(key) > 0:
#             numConcepts = numConcepts + 1
#         else:
#             numRoles = numRoles + 1
#
#     return numConcepts, numRoles

# Tested
# def levenshtein(s1, s2):
#     # https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
#     """Calculates basic Levenshtein edit distance between the given strings."""
#     if len(s1) < len(s2):
#         return levenshtein(s2, s1)
#
#     # len(s1) >= len(s2)
#     if len(s2) == 0:
#         return len(s1)
#
#     previous_row = range(len(s2) + 1)
#     for i, c1 in enumerate(s1):
#         current_row = [i + 1]
#         for j, c2 in enumerate(s2):
#             insertions = previous_row[
#                              j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
#             deletions = current_row[j] + 1  # than s2
#             substitutions = previous_row[j] + (c1 != c2)
#             current_row.append(min(insertions, deletions, substitutions))
#         previous_row = current_row
#
#     return previous_row[-1]


# def lev_distance(shape, newIRIs, trueIRIs, numConcepts, numRoles, map):
#     """Calculates levenshtein distance between iri strings."""
#
#     flatTrue = [[item for sublist in x for item in sublist] for x in trueIRIs]
#     flatNew = [[item for sublist in x for item in sublist] for x in newIRIs]
#
#     evalInfoNew = np.array([0, 0, 0])
#
#     levTN = 0
#     levNT = 0
#
#     countTrue = 0
#     countNew = 0
#
#     for sampleNum in range(shape[0]):
#         for timestepNum in range(shape[1]):
#             for tripleNum in range(shape[2]):
#
#                 if len(trueIRIs) > sampleNum and len(trueIRIs[sampleNum]) > timestepNum and len(trueIRIs[sampleNum][timestepNum]) > tripleNum:
#
#                     countTrue = countTrue + 1
#
#                     if (len(newIRIs) > sampleNum and len(newIRIs[sampleNum]) > 0):
#                         levTN = levTN + find_best_match(trueIRIs[sampleNum][timestepNum][tripleNum], flatNew[
#                             sampleNum])  # if there are predictions for this KB, compare to best match in there
#                     else:
#                         levTN = levTN + levenshtein(trueIRIs[sampleNum][timestepNum][tripleNum], '')  # otherwise compare with no prediction
#
#                 if len(newIRIs) > sampleNum and len(newIRIs[sampleNum]) > timestepNum and len(newIRIs[sampleNum][timestepNum]) > tripleNum:  # FOR EVERY PREDICTION
#                     countNew = countNew + 1
#                     if (len(trueIRIs) > sampleNum and len(trueIRIs[sampleNum]) > 0):
#                         best = find_best_match(newIRIs[sampleNum][timestepNum][tripleNum], flatTrue[sampleNum])
#                         if best == 0:
#                             print(newIRIs[sampleNum][timestepNum][tripleNum])
#                             evalInfoNew[0] = evalInfoNew[0] + 1
#                         levNT = levNT + best  # If there are true values for this KB, compare to best match in there
#                     else:
#                         levNT = levNT + levenshtein(newIRIs[sampleNum][timestepNum][tripleNum], '')  # otherwise compare with no true value
#
#     evalInfoNew[1] = countNew - evalInfoNew[0]
#     evalInfoNew[2] = countTrue - evalInfoNew[0]
#     return levTN, levNT, countTrue, countNew, evalInfoNew

#
# def find_best_match(statement, reasonerSteps):
#     return min(map(partial(levenshtein, statement), reasonerSteps))

# Distance evaluation with strings if Pascal wants us to do that.
# def distanceEvaluations(log, shape, newPredictions, trueLabels, newStrIRI, trueIRIs, numConcepts, numRoles, map):
#     custTN, custNT, countTrue, countNew, newEvalInfoCustDist = custom_distance(shape, newPredictions, trueLabels,
#                                                                                numConcepts, numRoles)
#
#     log.write(
#         "\nCustom Label Distance:\nCustom Distance From True to Predicted Data,{}\nCustom Distance From Predicted "
#         "to True Data,{}".format(custTN, custNT))
#
#     log.write(
#         "\nAverage Custom Distance From True to Predicted Statement,{}\nAverage Custom Distance From Prediction to "
#         "True Statement,{}\n".format(custTN / countTrue, 0 if countNew == 0 else custNT / countNew))
#
#     c = write_evaluation_measures(newEvalInfoCustDist, log)
#
#     levTN2, levNT2, sizeTrue2, sizeNew2, newEvalInfoLevDist = lev_distance(shape, newStrIRI, trueIRIs, numConcepts,
#                                                                            numRoles, map)
#
#     log.write(
#         "\nString Distance:\nLevenshtein Distance From True to Predicted Data,{}\nLevenshtein Distance From Prediction "
#         "to True Data,{}".format(levTN2, levNT2))
#
#     log.write(
#         "\nAverage Levenshtein Distance From True to Predicted Statement,{}\nAverage Levenshtein Distance From "
#         "Predicted to True Statement,{}\n".format(levTN2 / sizeTrue2, 0 if sizeNew2 == 0 else levNT2 / sizeNew2))
#
#     b = write_evaluation_measures(newEvalInfoLevDist, log)
#
#     return np.array([np.array([levTN2, levNT2, sizeTrue2, sizeNew2, b]),
#                     np.array([custTN, custNT, countTrue, countNew, c])])

# Tested
# def remove_padding(origData):
#     data = np.array(origData)
#
#     for sampleNum in range(len(data)):
#         for timeStepNum in range(len(data[sampleNum])):
#             tripleNum = 0
#             while tripleNum < len(data[sampleNum][timeStepNum]):
#                 if data[sampleNum][timeStepNum][tripleNum] == ('0', '0', '0'):
#                     data[sampleNum][timeStepNum].pop(tripleNum)
#                 else:
#                     tripleNum += 1
#     return data

# def write_final_average_data(result, log):
#     levTN2, levNT2, sizeTrue2, sizeNew2, (TPs1, FPs1, FNs1, pre1, rec1, F1) = result[1]
#     custTN, custNT, countTrue, countNew, (TPs2, FPs2, FNs2, pre2, rec2, F2) = result[0]
#
#     log.write(
#         "\nString Distance:\nAverage Levenshtein Distance From True to Predicted Data,{}\nAverage Levenshtein Distance "
#         "From Prediction to True Data,{}".format(levTN2, levNT2))
#
#     log.write(
#         "\nAverage Levenshtein Distance From True to Predicted Statement,{}\nAverage Levenshtein Distance From "
#         "Prediction to True Statement,{}\n".format(levTN2 / sizeTrue2, levNT2 / sizeNew2))
#
#     log.write(
#         "\nAverage Prediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse "
#         "Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(TPs1, FPs1, FNs1, pre1, rec1, F1))
#
#     log.write(
#         "\nCustom Label Distance:\nAverage Custom Distance From True to Predicted Data,{}\nAverage Custom Distance "
#         "From Predicted to True Data,{}".format(custTN, custNT))
#
#     log.write(
#         "\nAverage Custom Distance From True to Predicted Statement,{}\nAverage Custom Distance From Prediction to "
#         "True Statement,{}\n".format(custTN / countTrue, custNT / countNew))
#
#     log.write(
#         "\nAverage Prediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse "
#         "Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(TPs2, FPs2, FNs2, pre2, rec2, F2))
#
# def get_label_and_iri_from_encoding(encodedPredictions, labelMap, decodeNonProperty, decodeProperty):
#     """Gets closest label and iri for a model prediction."""
#
#     labelPredictions = np.zeros((encodedPredictions.shape[0], encodedPredictions.shape[1]), dtype=tuple)
#     stringPredictions = np.zeros((encodedPredictions.shape[0], encodedPredictions.shape[1]), dtype=tuple)
#
#     sampleBatchIndex = 0
#     for sampleBatch in encodedPredictions:
#         timeStepIndex = 0
#         for timeStep in sampleBatch:
#             labelsForTimeStep = []
#             strForTimeStep = []
#             tempLabels = []
#             tempStr = []
#
#             for item in timeStep:
#                 intLabel, strIri = convert_encoding_to_label_and_iri(item, labelMap, decodeNonProperty, decodeProperty)
#                 tempLabels.append(intLabel)
#                 tempStr.append(strIri)
#
#                 if len(tempLabels) == 3:
#                     # Prevents any incomplete or padding triples from being evaluated similar to how Aaron did his.
#                     if tempLabels.__contains__('0'):
#                         pass
#                     else:
#                         if len(tempStr) == 3:
#                             labelsForTimeStep.append(tuple((tempLabels[0], tempLabels[1], tempLabels[2])))
#                             strForTimeStep.append(tuple((tempStr[0], tempStr[1], tempStr[2])))
#                             tempLabels = []
#                             tempStr = []
#                         else:
#                             print("Error in get_predicted_label_and_iri_from_encoding: ???")
#
#             labelPredictions[sampleBatchIndex][timeStepIndex] = labelsForTimeStep
#             stringPredictions[sampleBatchIndex][timeStepIndex] = strForTimeStep
#             timeStepIndex += 1
#         sampleBatchIndex += 1
#     return labelPredictions, stringPredictions
#
#
# def convert_encoding_to_label_and_iri(enc, labelMap, numConcepts, numRoles):
#     """Converts a float representing an encoding into an int label and its string iri."""
#     if (enc > 0):
#         label = int(enc * numConcepts)
#
#         # Makes sure it is in range
#         if label > numConcepts:
#             label = label - 1
#         # This makes it possible for model to guess an empty label.
#         if label == 0:
#             return '0', '0'
#
#         iriStr = labelMap.get(str(label))
#
#         return "C" + str(label), iriStr
#
#     else:
#         label = int(enc * numRoles)
#
#         # Makes sure it is in range
#         if label < (-1 * numRoles):
#             label = label + 1
#         if label == 0:
#             return '0', '0'
#
#         iriStr = labelMap.get(str(label))
#
#         return "R" + str(abs(label)), iriStr