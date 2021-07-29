import io
import random, math

from numpy import array
from functools import partial

from Generator.GenERator import GenERator
from datasetGenerators import *

import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def makeRandomPredCompletions(shape, conceptSpace, roleSpace, syn):
    rando = []

    for i in range(shape[0]):
        kb = []
        for j in range(shape[1]):
            step = []
            inner = int(shape[2] / 4)
            split = random.randint(0, inner)
            gen = GenERator(numCType1=split, numCType2=0, numCType3=(inner - split), numCType4=0, numRoleSub=0,
                            numRoleChains=0, conceptNamespace=conceptSpace, roleNamespace=roleSpace, CTypeNull=[],
                            CType1=[], CType2=[], CType3=[], CType4=[], roleSubs=[], roleChains=[])
            gen.genERate()
            for k in range(0, len(gen.CType1)):
                step.append([gen.CType1[k].antecedent.toString(), gen.CType1[k].consequent.toString()])
            for k in range(0, len(gen.CType3)):
                step.append([gen.CType3[k].antecedent.toString(), gen.CType3[k].consequent.role.toString(),
                             gen.CType3[k].consequent.concept.toString()])
            kb.append(step)
        rando.append(kb)

    numpy.savez("saves/randoPred" if syn else "ssaves/randoPred", rando)

    return rando


def makeRandomStrCompletions(shape, conceptSpace, roleSpace, syn):
    rando = []

    for i in range(shape[0]):
        kb = []
        for j in range(shape[1]):
            step = []
            inner = int(shape[2] / 4)
            split = random.randint(0, inner)
            gen = GenERator(numCType1=split, numCType2=0, numCType3=(inner - split), numCType4=0, numRoleSub=0,
                            numRoleChains=0, conceptNamespace=conceptSpace + 1, roleNamespace=roleSpace + 1,
                            CTypeNull=[], CType1=[], CType2=[], CType3=[], CType4=[], roleSubs=[], roleChains=[])
            gen.genERate()
            for k in range(0, len(gen.CType1)):
                # print(gen.CType1[k].toString())
                if gen.CType1[k].antecedent.name == 0:
                    gen.CType1[k].antecedent.name = conceptSpace
                if gen.CType1[k].consequent.name == 0:
                    gen.CType1[k].consequent.name = conceptSpace
                step.append(gen.CType1[k].toString())
            for k in range(0, len(gen.CType3)):
                # print(gen.CType3[k].toString())
                if gen.CType3[k].antecedent.name == 0:
                    gen.CType3[k].antecedent.name = conceptSpace
                if gen.CType3[k].consequent.concept.name == 0:
                    gen.CType3[k].consequent.concept.name = conceptSpace
                if gen.CType3[k].consequent.role.name == 0:
                    gen.CType3[k].consequent.role.name = roleSpace
                step.append(gen.CType3[k].toString())
            kb.append(step)
        rando.append(kb)

    numpy.savez("saves/randoStr" if syn else "ssaves/randoStr", rando)

    return rando


def collapseLabelMap(localMap, classes, roles, labels):
    for mapping in localMap:
        for entry in mapping:
            mapping[entry] = labels[classes[mapping[entry]]] if mapping[entry] > 0 else labels[roles[mapping[entry]]]
    return localMap


def getSynDataFromFile(filename):
    print("Loading data from " + filename)
    data = numpy.load(filename, allow_pickle=True)
    return data['arr_0'], data['arr_1'], data['arr_2']


def getSnoDataFromFile(filename):
    print("Loading data from " + filename)
    data = numpy.load(filename, allow_pickle=True)
    return data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4']


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
    file = io.open(filename, "w", encoding="utf-8")
    for i in range(len(vector)):
        file.write("Trial: {}\n".format(i))
        for j in range(len(vector[i])):
            file.write("\tStep: {}\n".format(j))
            for k in range(len(vector[i][j])):
                file.write("\t\t{}\n".format(vector[i][j][k]))
        file.write("\n")
    file.close()


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


def vecToStatement(vec, conceptSpace, roleSpace):
    four = []
    statementStr = []
    statementPred = []

    for j in range(len(vec)):
        stepStr = []
        stepPred = []
        for k in range(len(vec[j])):
            if len(four) == 3:
                four.append(vec[j][k])
                pred, stri = convertToStatement(four, conceptSpace, roleSpace)
                if stri != None: stepStr.append(stri)
                if pred != None: stepPred.append(pred)
                four = []
            else:
                four.append(vec[j][k])
        if len(stepStr) > 0:
            statementStr.append(stepStr)
        if len(stepPred) > 0:
            statementPred.append(stepPred)

    return statementPred, statementStr


def vecToStatements(vec, conceptSpace, roleSpace):
    four = []
    statementStr = []
    statementPred = []

    for i in range(len(vec)):
        trialStr = []
        trialPred = []
        for j in range(len(vec[i])):
            stepStr = []
            stepPred = []
            for k in range(len(vec[i][j])):
                if len(four) == 3:
                    four.append(vec[i][j][k])
                    pred, stri = convertToStatement(four, conceptSpace, roleSpace)
                    if stri != None: stepStr.append(stri)
                    if pred != None: stepPred.append(pred)
                    four = []
                else:
                    four.append(vec[i][j][k])
            if len(stepStr) > 0:
                trialStr.append(stepStr)
            if len(stepPred) > 0:
                trialPred.append(stepPred)
        statementStr.append(trialStr)
        statementPred.append(trialPred)

    return statementPred, statementStr


def vecToStatementsWithLabels(vec, conceptSpace, roleSpace, labels):
    four = []
    statementStr = []
    statementPred = []

    # For each piece of sample data in the predicted y from model.
    for i in range(len(vec)):
        trialStr = []
        trialPred = []

        # For each timestep in the sample
        for j in range(len(vec[i])):
            stepStr = []
            stepPred = []

            for k in range(len(vec[i][j])):
                if len(four) == 3:
                    four.append(vec[i][j][k])
                    pred, stri = convertToStatementWithLabels(four, conceptSpace, roleSpace, labels[i])
                    if stri != None: stepStr.append(stri)
                    if pred != None: stepPred.append(pred)
                    four = []
                else:
                    four.append(vec[i][j][k])
            if len(stepStr) > 0:
                trialStr.append(stepStr)
            if len(stepPred) > 0:
                trialPred.append(stepPred)
        statementStr.append(trialStr)
        statementPred.append(trialPred)

    return statementPred, statementStr


def convertToStatementWithLabels(four, conceptSpace, roleSpace, labels):
    new = []
    text = []
    for number in four:
        if isinstance(number, numpy.float32):
            number = number.item()
        if number < 0 and number >= -1:
            if int(number * roleSpace * -1) == 0:
                pass
            else:
                number = int(number * roleSpace * -1)
                text.append(labels[-number]) if (-number) in labels.keys() else text.append(
                    "undefinedRelationTo{}".format(number))
                new.append("R{}".format(number))
        elif number > 0 and number <= 1:
            if int(number * conceptSpace) == 0:
                pass
            else:
                number = int(number * conceptSpace)
                text.append(labels[number]) if number in labels.keys() else text.append(
                    "UndefinedConcept{}".format(number))
                new.append("C{}".format(number))

    if len(new) == 0:
        return None, None
    elif len(new) == 1:
        return None, None  # new,new[0]
    elif len(new) == 2:  # and ((four[1] > 0 and four[2] > 0) or (four[1] < 0 and four[2] < 0)):
        return new, "{}\n\t\t\ta {}".format(" ⊑ ".join(new), " is a ".join(text))
    # elif len(new) == 2:
    # return new,None
    elif len(new) == 3:
        if four[1] > 0 and four[2] < 0 and four[3] > 0:
            return new, "{} ⊑ ∃{}.{}\n\t\t\tif something is a {} then there is a {} that it is {}".format(new[0],
                                                                                                          new[1],
                                                                                                          new[2],
                                                                                                          text[0],
                                                                                                          text[2],
                                                                                                          text[1])
        elif four[1] > 0 and four[0] < 0 and four[2] > 0:
            return new, "∃{}.{} ⊑ {}\n\t\t\tif there is a {} that is {} another thing then the other thing is a {}".format(
                new[0], new[1], new[2], text[1], text[0], text[2])
        elif four[1] > 0 and four[0] > 0 and four[2] > 0:
            return new, "{} ⊓ {} ⊑ {}\n\t\t\tif something is both a {} and a {}, then it is also a {}".format(new[0],
                                                                                                              new[1],
                                                                                                              new[2],
                                                                                                              text[0],
                                                                                                              text[1],
                                                                                                              text[2])
        elif four[1] < 0 and four[0] < 0 and four[2] < 0:
            return new, "{} ∘ {} ⊑ {}\n\t\t\tif a first thing is {} anything that is {} a third thing, then the first is {} the third".format(
                new[0], new[1], new[2], text[0], text[1], text[2])

    return None, None


def convertToStatement(four, conceptSpace, roleSpace):
    new = []
    for number in four:
        if isinstance(number, numpy.float32):
            number = number.item()
            # if number < 0: pass
        if number < 0 and number >= -1:
            if int(number * roleSpace * -1) == 0:
                pass
            else:
                new.append("R{}".format(int(number * roleSpace * -1)))
        elif number > 0 and number <= 1:
            if int(number * conceptSpace) == 0:
                pass
            else:
                new.append("C{}".format(int(number * conceptSpace)))

    if len(new) == 0:
        return None, None
    elif len(new) == 1:
        return None, None  # new,new[0]
    elif len(new) == 2:  # and ((four[1] > 0 and four[2] > 0) or (four[1] < 0 and four[2] < 0)):
        return new, " ⊑ ".join(new)
    # elif len(new) == 2:
    # return new,None
    elif len(new) == 3:
        if four[1] > 0 and four[2] < 0 and four[3] > 0:
            return new, "{} ⊑ ∃{}.{}".format(new[0], new[1], new[2])
        elif four[1] > 0 and four[0] < 0 and four[2] > 0:
            return new, "∃{}.{} ⊑ {}".format(new[0], new[1], new[2])
        elif four[1] > 0 and four[0] > 0 and four[2] > 0:
            return new, "{} ⊓ {} ⊑ {}".format(new[0], new[1], new[2])
        elif four[1] < 0 and four[0] < 0 and four[2] < 0:
            return new, "{} ∘ {} ⊑ {}".format(new[0], new[1], new[2])

    return None, None  # " ".join(new)


def splitTensors(inputs, outputs, size):
    inTest, inTrain = numpy.split(inputs, [int(len(inputs) * size)])
    outTest, outTrain = numpy.split(outputs, [int(len(outputs) * size)])
    return inTrain, inTest, outTrain, outTest


# https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
def levenshtein(s1, s2):
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


def levenshteinIgnoreNum(s1, s2):
    if len(s1) < len(s2):
        return levenshteinIgnoreNum(s2, s1)

    s1, s2 = convertAllNumsToAtoms(s1, s2)

    if len(s1) < len(s2):
        return levenshteinIgnoreNum(s2, s1)

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


def convertAllNumsToAtoms(s1, s2):
    dic = {}
    a = 'a'
    st = ""
    longer = False
    for i in range(len(s1)):
        if s1[i].isdigit() and not longer:
            st = s1[i]
            for j in range(i + 1, len(s1)):
                if s1[j].isdigit():
                    longer = True
                    st = st + s1[j]
                else:
                    break
            if not int(st) in dic.keys():
                dic[int(st)] = a
                a = chr(ord(a) + 1)
        elif not s1[i].isdigit():
            longer = False
    longer = False
    for i in range(len(s2)):
        if s2[i].isdigit() and not longer:
            st = s2[i]
            for j in range(i + 1, len(s2)):
                if s2[j].isdigit():
                    longer = True
                    st = st + s2[j]
                else:
                    break
            if not int(st) in dic.keys():
                dic[int(st)] = a
                a = chr(ord(a) + 1)
        elif not s1[i].isdigit():
            longer = False

    for key in sorted(dic, reverse=True):
        s1 = s1.replace(str(key), dic[key])
        s2 = s2.replace(str(key), dic[key])
    s1 = s1.replace(" ", "")
    s2 = s2.replace(" ", "")
    return s1, s2


def findBestMatchNoNums(statement, reasonerSteps):
    return min(map(partial(levenshteinIgnoreNum, statement), reasonerSteps))


def levDistanceNoNums(shape, newStatements, trueStatements, conceptSpace, roleSpace, syn, mix):
    if (syn and os.path.isfile("saves/randoStr.npz")) or os.path.isfile("ssaves/randoStr.npz"):
        rando = numpy.load(("saves/randoStr.npz" if syn else "ssaves/randoStr.npz"), allow_pickle=True)
        rando = rando['arr_0'].tolist()
    else:
        rando = makeRandomStrCompletions(shape, conceptSpace, roleSpace, syn)

    flatTrue = [[item for sublist in x for item in sublist] for x in trueStatements]
    flatRand = [[item for sublist in x for item in sublist] for x in rando]
    flatNew = [[item for sublist in x for item in sublist] for x in newStatements]

    F1s = array([0, sum([len(x) for x in flatNew]), sum([len(x) for x in flatTrue])])
    rF1s = array([0, 0, 0])

    levTR = 0
    levRT = 0
    levTN = 0
    levNT = 0

    sizeRan = 0
    sizeTrue = 0
    sizeNew = 0

    for i in range(len(rando)):
        for j in range(len(rando[i])):
            for k in range(len(rando[i][j])):
                sizeRan = sizeRan + 1
                if len(trueStatements) > i and len(trueStatements[i]) > j and len(
                        trueStatements[i][j]) > k:  # FOR VERY TRUE STATEMENT
                    sizeTrue = sizeTrue + 1  # if there is a true KB corresponding to this random data, compare the random statement to its best match in the true statements

                    levRT = levRT + findBestMatchNoNums(rando[i][j][k], flatTrue[i])
                    best = findBestMatchNoNums(trueStatements[i][j][k], flatRand[i])  # compare to best match in random
                    if best == 0: rF1s[0] = rF1s[0] + 1
                    levTR = levTR + best
                    if (len(newStatements) > i and len(newStatements[i]) > 0):
                        levTN = levTN + findBestMatchNoNums(trueStatements[i][j][k], flatNew[
                            i])  # if there are predictions for this KB, compare to best match in there
                    elif not mix:
                        levTN = levTN + levenshteinIgnoreNum(trueStatements[i][j][k],
                                                             '')  # otherwise compare with no prediction

                if len(newStatements) > i and len(newStatements[i]) > j and len(
                        newStatements[i][j]) > k:  # FOR EVERY PREDICTION
                    sizeNew = sizeNew + 1
                    if (len(trueStatements) > i and len(trueStatements[i]) > 0):
                        best = findBestMatchNoNums(newStatements[i][j][k], flatTrue[i])
                        if best == 0: F1s[0] = F1s[0] + 1
                        levNT = levNT + best  # if there are true values for this KB, compare to best match in there
                    elif not mix:
                        levNT = levNT + levenshteinIgnoreNum(newStatements[i][j][k],
                                                             '')  # otherwise compare with no true value

    F1s[1] = sizeNew - F1s[0]
    F1s[2] = sizeTrue - F1s[0]
    rF1s[1] = sizeRan - rF1s[0]
    rF1s[2] = sizeTrue - rF1s[0]
    return levTR, levRT, levTN, levNT, sizeTrue, sizeNew, sizeRan, F1s, rF1s


def levDistance(shape, newStatements, trueStatements, conceptSpace, roleSpace, syn, mix):
    if (syn and os.path.isfile("saves/randoStr.npz")) or os.path.isfile("ssaves/randoStr.npz"):
        rando = numpy.load("saves/randoStr.npz" if syn else "ssaves/randoStr.npz", allow_pickle=True)
        rando = rando['arr_0'].tolist()
    else:
        rando = makeRandomStrCompletions(shape, conceptSpace, roleSpace, syn)

    flatTrue = [[item for sublist in x for item in sublist] for x in trueStatements]
    flatRand = [[item for sublist in x for item in sublist] for x in rando]
    flatNew = [[item for sublist in x for item in sublist] for x in newStatements]

    F1s = array([0, 0, 0])
    rF1s = array([0, 0, 0])

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
                if len(trueStatements) > i and len(trueStatements[i]) > j and len(
                        trueStatements[i][j]) > k:  # FOR VERY TRUE STATEMENT
                    countTrue = countTrue + 1
                    levTR = levTR + findBestMatch(trueStatements[i][j][k],
                                                  flatRand[i])  # compare to best match in random and vice versa
                    best = findBestMatch(rando[i][j][k], flatTrue[i])
                    if best == 0: rF1s[0] = rF1s[0] + 1
                    levRT = levRT + best

                    if (len(newStatements) > i and len(newStatements[i]) > 0):
                        levTN = levTN + findBestMatch(trueStatements[i][j][k], flatNew[
                            i])  # if there are predictions for this KB, compare to best match in there
                    elif not mix:
                        levTN = levTN + levenshtein(trueStatements[i][j][k], '')  # otherwise compare with no prediction

                if len(newStatements) > i and len(newStatements[i]) > j and len(
                        newStatements[i][j]) > k:  # FOR EVERY PREDICTION
                    countNew = countNew + 1
                    if (len(trueStatements) > i and len(trueStatements[i]) > 0):
                        best = findBestMatch(newStatements[i][j][k], flatTrue[i])
                        if best == 0: F1s[0] = F1s[0] + 1
                        levNT = levNT + best  # if there are true values for this KB, compare to best match in there
                    elif not mix:
                        levNT = levNT + levenshtein(newStatements[i][j][k], '')  # otherwise compare with no true value

    F1s[1] = countNew - F1s[0]
    F1s[2] = countTrue - F1s[0]
    rF1s[1] = countRan - rF1s[0]
    rF1s[2] = countTrue - rF1s[0]
    return levTR, levRT, levTN, levNT, countTrue, countNew, countRan, F1s, rF1s


def findBestMatch(statement, reasonerSteps):
    return min(map(partial(levenshtein, statement), reasonerSteps))


def custom(conceptSpace, roleSpace, s1, s2):
    if len(s1) < len(s2): return custom(conceptSpace, roleSpace, s2, s1)

    dist = 0

    for k in range(len(s1)):
        string1 = s1[k]
        string2 = s2[k] if len(s2) > k else ""
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


def customDistance(shape, newPred, truePred, conceptSpace, roleSpace, syn, mix):
    if (syn and os.path.isfile("saves/randoPred.npz")) or os.path.isfile("ssaves/randoPred.npz"):
        rando = numpy.load("saves/randoPred.npz" if syn else "ssaves/randoPred.npz", allow_pickle=True)
        rando = rando['arr_0'].tolist()
    else:
        rando = makeRandomPredCompletions(shape, conceptSpace, roleSpace, syn)

    flatTrue = [[item for sublist in x for item in sublist] for x in truePred]
    flatRand = [[item for sublist in x for item in sublist] for x in rando]
    flatNew = [[item for sublist in x for item in sublist] for x in newPred]

    F1s = array([0, 0, 0])
    rF1s = array([0, 0, 0])

    custTR = 0
    custRT = 0
    custTN = 0
    custNT = 0

    countRan = 0
    countTrue = 0
    countNew = 0

    for i in range(len(rando)):  # KB
        for j in range(len(rando[i])):  # Step
            for k in range(len(rando[i][j])):  # Statement
                countRan = countRan + 1
                if (len(truePred) > i and len(truePred[i]) > j and len(truePred[i][j]) > k):
                    countTrue = countTrue + 1
                    custTR = custTR + findBestPredMatch(truePred[i][j][k], flatRand[i], conceptSpace, roleSpace)
                    best = findBestPredMatch(rando[i][j][k], flatTrue[i], conceptSpace, roleSpace)
                    if best == 0: rF1s[0] = rF1s[0] + 1
                    custRT = custRT + best
                    if (len(newPred) > i and len(newPred[i]) > 0):
                        custTN = custTN + findBestPredMatch(truePred[i][j][k], flatNew[i], conceptSpace, roleSpace)
                    elif not mix:
                        custTN = custTN + custom(conceptSpace, roleSpace, truePred[i][j][k], [])
                if (len(newPred) > i and len(newPred[i]) > j and len(newPred[i][j]) > k):
                    countNew = countNew + 1
                    if (len(truePred) > i and len(truePred[i]) > 0):
                        best = findBestPredMatch(newPred[i][j][k], flatTrue[i], conceptSpace, roleSpace)
                        if best == 0: F1s[0] = F1s[0] + 1
                        custNT = custNT + best
                    elif not mix:
                        custNT = custNT + custom(conceptSpace, roleSpace, newPred[i][j][k], [])
    F1s[1] = countNew - F1s[0]
    F1s[2] = countTrue - F1s[0]
    rF1s[1] = countRan - rF1s[0]
    rF1s[2] = countTrue - rF1s[0]
    return custTR, custRT, custTN, custNT, countTrue, countNew, countRan, F1s, rF1s


def findBestPredMatch(statement, otherKB, conceptSpace, roleSpace):
    return min(map(partial(custom, conceptSpace, roleSpace, statement), otherKB))


def repeatAndSplitKBs(kbs, steps, splitSize):
    newKBs = numpy.empty([kbs.shape[0], steps, kbs.shape[1]], dtype=numpy.float32)
    for i in range(len(newKBs)):
        for j in range(steps):
            newKBs[i][j] = kbs[i]
    return numpy.split(newKBs, [int(len(newKBs) * splitSize)])


def formatDataSynth(log, conceptSpace, roleSpace, KBs, supports, output):
    fileShapes1 = [max(len(max(supports, key=lambda coll: len(coll))), len(max(output, key=lambda coll: len(coll)))),
                   len(max(supports, key=lambda coll: len(coll[0]))[0]),
                   len(max(output, key=lambda coll: len(coll[0]))[0])]

    KBs_test, KBs_train = repeatAndSplitKBs(KBs, fileShapes1[0], 0.1)

    X_train, X_test, y_train, y_test = splitTensors(supports, output, 0.1)

    X_train = pad(X_train, maxlen1=fileShapes1[0], maxlen2=fileShapes1[1])
    X_test = pad(X_test, maxlen1=fileShapes1[0], maxlen2=fileShapes1[1])

    y_train = pad(y_train, maxlen1=fileShapes1[0], maxlen2=fileShapes1[2])
    y_test = pad(y_test, maxlen1=fileShapes1[0], maxlen2=fileShapes1[2])

    print("KBs shape:\t\t{}\nExtended KBs shape:\t{}{}\nDependencies shape:\t{}{}\nOutput shape:\t\t{}{}\n\n".format(
        KBs.shape, KBs_train.shape, KBs_test.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    log.write(
        "KBs shape,{}\nExtended KBs shape,{},{}\nDependencies shape,{},{}\nOutput shape,{},{}\n\n".format(KBs.shape,
                                                                                                          KBs_train.shape,
                                                                                                          KBs_test.shape,
                                                                                                          X_train.shape,
                                                                                                          X_test.shape,
                                                                                                          y_train.shape,
                                                                                                          y_test.shape))

    KBvec, KBstr = vecToStatements(KBs_test, conceptSpace, roleSpace)
    truePreds, trueStatements = vecToStatements(y_test, conceptSpace, roleSpace)
    placeholder, inputs = vecToStatements(X_test, conceptSpace, roleSpace)

    writeVectorFile("output/KBsIn.txt", KBstr)
    writeVectorFile("output/supports.txt", inputs)
    writeVectorFile("output/reasonerCompletion.txt", trueStatements)

    return KBs_test, KBs_train, X_train, X_test, y_train, y_test, truePreds, trueStatements, None


def formatDataSno(log, conceptSpace, roleSpace, KBs, supports, output, localMaps, stats):
    labels = collapseLabelMap(localMaps, stats[0][2], stats[1][2], stats[4][1])

    fileShapes1 = [max(len(max(supports, key=lambda coll: len(coll))), len(max(output, key=lambda coll: len(coll)))),
                   len(max(supports, key=lambda coll: len(coll[0]))[0]),
                   len(max(output, key=lambda coll: len(coll[0]))[0])]

    KBs_test, KBs_train = repeatAndSplitKBs(KBs, fileShapes1[0], 0.33)

    testLabels = labels[:len(KBs_test)]
    trainLabels = labels[len(KBs_test):]

    X_train, X_test, y_train, y_test = splitTensors(supports, output, 0.33)

    X_train = pad(X_train, maxlen1=fileShapes1[0], maxlen2=fileShapes1[1])
    X_test = pad(X_test, maxlen1=fileShapes1[0], maxlen2=fileShapes1[1])

    y_train = pad(y_train, maxlen1=fileShapes1[0], maxlen2=fileShapes1[2])
    y_test = pad(y_test, maxlen1=fileShapes1[0], maxlen2=fileShapes1[2])

    print("KBs shape:\t\t{}\nExtended KBs shape:\t{}{}\nDependencies shape:\t{}{}\nOutput shape:\t\t{}{}\n\n".format(
        KBs.shape, KBs_train.shape, KBs_test.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    log.write(
        "KBs shape,{}\nExtended KBs shape,{},{}\nDependencies shape,{},{}\nOutput shape,{},{}\n\n".format(KBs.shape,
                                                                                                          KBs_train.shape,
                                                                                                          KBs_test.shape,
                                                                                                          X_train.shape,
                                                                                                          X_test.shape,
                                                                                                          y_train.shape,
                                                                                                          y_test.shape))

    KBvec, KBstr = vecToStatementsWithLabels(KBs_test, conceptSpace, roleSpace, testLabels)
    preds, trueStatements = vecToStatementsWithLabels(y_test, conceptSpace, roleSpace, testLabels)
    placeholder, inputs = vecToStatementsWithLabels(X_test, conceptSpace, roleSpace, testLabels)

    writeVectorFile("snoutput/KBsIn.txt", KBstr)
    writeVectorFile("snoutput/supports.txt", inputs)
    writeVectorFile("snoutput/reasonerCompletion.txt", trueStatements)

    truePreds, trueStatements = vecToStatements(y_test, conceptSpace, roleSpace)

    return KBs_test, KBs_train, X_train, X_test, y_train, y_test, truePreds, trueStatements, labels


def formatDataSyn2Sno(log, conceptSpace, roleSpace, KBs, supports, output, sKBs, ssupports, soutput, localMaps, stats):
    labels = collapseLabelMap(localMaps, stats[0][2], stats[1][2], stats[4][1])

    fileShapes1 = [max(len(max(supports, key=lambda coll: len(coll))), len(max(output, key=lambda coll: len(coll)))),
                   len(max(supports, key=lambda coll: len(coll[0]))[0]),
                   len(max(output, key=lambda coll: len(coll[0]))[0])]

    KBs_test, KBs_train = repeatAndSplitKBs(KBs, fileShapes1[0], 0.33)
    KBs_test, a = repeatAndSplitKBs(sKBs, fileShapes1[0], 0.33)

    testLabels = labels[:len(KBs_test)]
    trainLabels = labels[len(KBs_test):]

    X_train, X_test, y_train, y_test = splitTensors(supports, output, 0.33)
    a, X_test, a, y_test = splitTensors(ssupports, soutput, 0.33)

    X_train = pad(X_train, maxlen1=fileShapes1[0], maxlen2=fileShapes1[1])
    X_test = pad(X_test, maxlen1=fileShapes1[0], maxlen2=fileShapes1[1])

    y_train = pad(y_train, maxlen1=fileShapes1[0], maxlen2=fileShapes1[2])
    y_test = pad(y_test, maxlen1=fileShapes1[0], maxlen2=fileShapes1[2])

    print("KBs shape:\t\t{}\nExtended KBs shape:\t{}{}\nDependencies shape:\t{}{}\nOutput shape:\t\t{}{}\n\n".format(
        KBs.shape, KBs_train.shape, KBs_test.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    log.write(
        "KBs shape,{}\nExtended KBs shape,{},{}\nDependencies shape,{},{}\nOutput shape,{},{}\n\n".format(KBs.shape,
                                                                                                          KBs_train.shape,
                                                                                                          KBs_test.shape,
                                                                                                          X_train.shape,
                                                                                                          X_test.shape,
                                                                                                          y_train.shape,
                                                                                                          y_test.shape))

    truePreds, trueStatements = vecToStatements(y_test, conceptSpace, roleSpace)

    return KBs_test, KBs_train, X_train, X_test, y_train, y_test, truePreds, trueStatements, testLabels


def formatDataSno2Syn(log, conceptSpace, roleSpace, KBs, supports, output, sKBs, ssupports, soutput, localMaps, stats):
    labels = collapseLabelMap(localMaps, stats[0][2], stats[1][2], stats[4][1])

    fileShapes1 = [max(len(max(supports, key=lambda coll: len(coll))), len(max(output, key=lambda coll: len(coll)))),
                   len(max(supports, key=lambda coll: len(coll[0]))[0]),
                   len(max(output, key=lambda coll: len(coll[0]))[0])]

    a, KBs_train = repeatAndSplitKBs(KBs, fileShapes1[0], 0.33)
    KBs_test, a = repeatAndSplitKBs(sKBs, fileShapes1[0], 0.33)

    testLabels = labels[:len(KBs_test)]
    trainLabels = labels[len(KBs_test):]

    X_train, X_test, y_train, y_test = splitTensors(supports, output, 0.33)
    a, X_test, a, y_test = splitTensors(ssupports, soutput, 0.33)

    X_train = pad(X_train, maxlen1=fileShapes1[0], maxlen2=fileShapes1[1])
    X_test = pad(X_test, maxlen1=fileShapes1[0], maxlen2=fileShapes1[1])

    y_train = pad(y_train, maxlen1=fileShapes1[0], maxlen2=fileShapes1[2])
    y_test = pad(y_test, maxlen1=fileShapes1[0], maxlen2=fileShapes1[2])

    print("KBs shape:\t\t{}\nExtended KBs shape:\t{}{}\nDependencies shape:\t{}{}\nOutput shape:\t\t{}{}\n\n".format(
        KBs.shape, KBs_train.shape, KBs_test.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    log.write(
        "KBs shape,{}\nExtended KBs shape,{},{}\nDependencies shape,{},{}\nOutput shape,{},{}\n\n".format(KBs.shape,
                                                                                                          KBs_train.shape,
                                                                                                          KBs_test.shape,
                                                                                                          X_train.shape,
                                                                                                          X_test.shape,
                                                                                                          y_train.shape,
                                                                                                          y_test.shape))

    truePreds, trueStatements = vecToStatements(y_test, conceptSpace, roleSpace)

    return KBs_test, KBs_train, X_train, X_test, y_train, y_test, truePreds, trueStatements, testLabels


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


def distanceEvaluations(log, shape, newPreds, truePreds, newStatements, trueStatements, conceptSpace, roleSpace, syn,
                        mix, newErrs, newErrStatements):
    if mix:
        levTR, levRT, levTN, levNT, sizeTrue, sizeNew, sizeRan, F11, F111 = levDistanceNoNums(shape, newStatements,
                                                                                              trueStatements,
                                                                                              conceptSpace, roleSpace,
                                                                                              syn, False)

        log.write(
            "\nRegular Distance\n\nNo Nums\nLevenshtein Distance From Reasoner to Random Data,{}\nLevenshtein Distance From Random to Reasoner Data,{}\nLevenshtein Distance From Reasoner to Predicted Data,{}\nLevenshtein Distance From Prediction to Reasoner Data,{}\n".format(
                levTR, levRT, levTN, levNT))
        log.write(
            "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
                levTR / sizeTrue, levRT / sizeRan, levTN / sizeTrue, 0 if sizeNew == 0 else levNT / sizeNew))

        a = writeAccMeasures(F11, F111, log)

        levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, F12, F121 = levDistance(shape, newStatements,
                                                                                               trueStatements,
                                                                                               conceptSpace, roleSpace,
                                                                                               syn, False)

        log.write(
            "\nNums\nLevenshtein Distance From Reasoner to Random Data,{}\nLevenshtein Distance From Random to Reasoner Data,{}\nLevenshtein Distance From Reasoner to Predicted Data,{}\nLevenshtein Distance From Prediction to Reasoner Data,{}\n".format(
                levTR2, levRT2, levTN2, levNT2))
        log.write(
            "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
                levTR2 / sizeTrue2, levRT2 / sizeRan2, levTN2 / sizeTrue2, 0 if sizeNew2 == 0 else levNT2 / sizeNew2))

        b = writeAccMeasures(F12, F121, log)

        custTR, custRT, custTN, custNT, countTrue, countNew, countRan, F13, F131 = customDistance(shape, newPreds,
                                                                                                  truePreds,
                                                                                                  conceptSpace,
                                                                                                  roleSpace, syn, False)

        log.write(
            "\nCustom\nCustom Distance From Reasoner to Random Data,{}\nCustom Distance From Random to Reasoner Data,{}\nCustom Distance From Reasoner to Predicted Data,{}\nCustom Distance From Predicted to Reasoner Data,{}\n".format(
                custTR, custRT, custTN, custNT))
        log.write(
            "Average Custom Distance From Reasoner to Random Statement,{}\nAverage Custom Distance From Random to Reasoner Statement,{}\nAverage Custom Distance From Reasoner to Predicted Statement,{}\nAverage Custom Distance From Prediction to Reasoner Statement,{}\n".format(
                custTR / countTrue, custRT / countRan, custTN / countTrue, 0 if countNew == 0 else custNT / countNew))

        c = writeAccMeasures(F13, F131, log)

        x = array([array([levTR, levRT, levTN, levNT, sizeTrue, sizeNew, sizeRan, a]),
                   array([levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, b]),
                   array([custTR, custRT, custTN, custNT, countTrue, countNew, countRan, c])])

        levTR, levRT, levTN, levNT, sizeTrue, sizeNew, sizeRan, F11, F111 = levDistanceNoNums(shape, newStatements,
                                                                                              trueStatements,
                                                                                              conceptSpace, roleSpace,
                                                                                              syn, mix)

        log.write(
            "\nDistance Ignoring Prediction Gaps\n\nNo Nums\nLevenshtein Distance From Reasoner to Random Data,{}\nLevenshtein Distance From Random to Reasoner Data,{}\nLevenshtein Distance From Reasoner to Predicted Data,{}\nLevenshtein Distance From Prediction to Reasoner Data,{}\n".format(
                levTR, levRT, levTN, levNT))
        log.write(
            "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
                levTR / sizeTrue, levRT / sizeRan, levTN / sizeTrue, 0 if sizeNew == 0 else levNT / sizeNew))

        a = writeAccMeasures(F11, F111, log)

        levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, F12, F121 = levDistance(shape, newStatements,
                                                                                               trueStatements,
                                                                                               conceptSpace, roleSpace,
                                                                                               syn, mix)

        log.write(
            "\nNums\nLevenshtein Distance From Reasoner to Random Data,{}\nLevenshtein Distance From Random to Reasoner Data,{}\nLevenshtein Distance From Reasoner to Predicted Data,{}\nLevenshtein Distance From Prediction to Reasoner Data,{}\n".format(
                levTR2, levRT2, levTN2, levNT2))
        log.write(
            "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
                levTR2 / sizeTrue2, levRT2 / sizeRan2, levTN2 / sizeTrue2, 0 if sizeNew2 == 0 else levNT2 / sizeNew2))

        b = writeAccMeasures(F12, F121, log)

        custTR, custRT, custTN, custNT, countTrue, countNew, countRan, F13, F131 = customDistance(shape, newPreds,
                                                                                                  truePreds,
                                                                                                  conceptSpace,
                                                                                                  roleSpace, syn, mix)

        log.write(
            "\nCustom\nCustom Distance From Reasoner to Random Data,{}\nCustom Distance From Random to Reasoner Data,{}\nCustom Distance From Reasoner to Predicted Data,{}\nCustom Distance From Predicted to Reasoner Data,{}\n".format(
                custTR, custRT, custTN, custNT))
        log.write(
            "Average Custom Distance From Reasoner to Random Statement,{}\nAverage Custom Distance From Random to Reasoner Statement,{}\nAverage Custom Distance From Reasoner to Predicted Statement,{}\nAverage Custom Distance From Prediction to Reasoner Statement,{}\n".format(
                custTR / countTrue, custRT / countRan, custTN / countTrue, 0 if countNew == 0 else custNT / countNew))

        c = writeAccMeasures(F13, F131, log)

        return x, array([array([levTR, levRT, levTN, levNT, sizeTrue, sizeNew, sizeRan, a]),
                         array([levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, b]),
                         array([custTR, custRT, custTN, custNT, countTrue, countNew, countRan, c])])
    elif newErrs:
        levTR, levRT, levTN, levNT, sizeTrue, sizeNew, sizeRan, F11, F111 = levDistanceNoNums(shape, newStatements,
                                                                                              trueStatements,
                                                                                              conceptSpace, roleSpace,
                                                                                              syn, mix)
        f, g, h, i, j, k, l, m, n = levDistanceNoNums(shape, newErrStatements, trueStatements, conceptSpace, roleSpace,
                                                      syn, mix)

        log.write(
            "\nNo Nums\nLevenshtein Distance From Reasoner to Random Data,{}\nLevenshtein Distance From Random to Reasoner Data,{}\nLevenshtein Distance From Reasoner to Error Data,{}\nLevenshtein Distance From Error to Reasoner Data,{}\n".format(
                levTR, levRT, levTN, levNT))
        log.write(
            "Levenshtein Distance From Reasoner to Error Data,{}\nLevenshtein Distance From Error to Reasoner Data,{}\n".format(
                h, i))
        log.write(
            "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
                levTR / sizeTrue, levRT / sizeRan, levTN / sizeTrue, 0 if sizeNew == 0 else levNT / sizeNew))
        log.write(
            "Average Levenshtein Distance From Reasoner to Error Statement,{}\nAverage Levenshtein Distance From Error to Reasoner Statement,{}\n".format(
                h / j, 0 if k == 0 else i / k))

        a = writeAccMeasures(F11, F111, log)

        TPs, FPs, FNs = m
        pre = precision(TPs, FPs)
        rec = recall(TPs, FNs)
        F = F1(pre, rec)

        log.write(
            "\nError Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
                TPs, FPs, FNs, pre, rec, F))
        x = array([h, i, j, k, array([TPs, FPs, FNs, pre, rec, F])])

        levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, F12, F121 = levDistance(shape, newStatements,
                                                                                               trueStatements,
                                                                                               conceptSpace, roleSpace,
                                                                                               syn, mix)
        f1, g1, h1, i1, j1, k1, l1, m1, n1 = levDistance(shape, newErrStatements, trueStatements, conceptSpace,
                                                         roleSpace, syn, mix)

        log.write(
            "\nNums\nLevenshtein Distance From Reasoner to Random Data,{}\nLevenshtein Distance From Random to Reasoner Data,{}\nLevenshtein Distance From Reasoner to Predicted Data,{}\nLevenshtein Distance From Prediction to Reasoner Data,{}\n".format(
                levTR2, levRT2, levTN2, levNT2))
        log.write(
            "Levenshtein Distance From Reasoner to Error Data,{}\nLevenshtein Distance From Error to Reasoner Data,{}\n".format(
                f1, g1))
        log.write(
            "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
                levTR2 / sizeTrue2, levRT2 / sizeRan2, levTN2 / sizeTrue2, 0 if sizeNew2 == 0 else levNT2 / sizeNew2))
        log.write(
            "Average Levenshtein Distance From Reasoner to Error Statement,{}\nAverage Levenshtein Distance From Error to Reasoner Statement,{}\n".format(
                h1 / j1, 0 if k1 == 0 else i1 / k1))

        b = writeAccMeasures(F12, F121, log)

        TPs, FPs, FNs = m1
        pre = precision(TPs, FPs)
        rec = recall(TPs, FNs)
        F = F1(pre, rec)

        log.write(
            "\nError Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
                TPs, FPs, FNs, pre, rec, F))
        x1 = array([h1, i1, j1, k1, array([TPs, FPs, FNs, pre, rec, F])])

        custTR, custRT, custTN, custNT, countTrue, countNew, countRan, F13, F131 = customDistance(shape, newPreds,
                                                                                                  truePreds,
                                                                                                  conceptSpace,
                                                                                                  roleSpace, syn, mix)
        f2, g2, h2, i2, j2, k2, l2, m2, n2 = customDistance(shape, newErrs, truePreds, conceptSpace, roleSpace, syn,
                                                            mix)

        log.write(
            "\nCustom\nCustom Distance From Reasoner to Random Data,{}\nCustom Distance From Random to Reasoner Data,{}\nCustom Distance From Reasoner to Predicted Data,{}\nCustom Distance From Predicted to Reasoner Data,{}\n".format(
                custTR, custRT, custTN, custNT))
        log.write(
            "Custom Distance From Reasoner to Error Data,{}\nCustom Distance From Error to Reasoner Data,{}\n".format(
                f2, g2))
        log.write(
            "Average Custom Distance From Reasoner to Random Statement,{}\nAverage Custom Distance From Random to Reasoner Statement,{}\nAverage Custom Distance From Reasoner to Predicted Statement,{}\nAverage Custom Distance From Prediction to Reasoner Statement,{}\n".format(
                custTR / countTrue, custRT / countRan, custTN / countTrue, 0 if countNew == 0 else custNT / countNew))
        log.write(
            "Average Custom Distance From Reasoner to Error Statement,{}\nAverage Custom Distance From Error to Reasoner Statement,{}\n".format(
                h2 / j2, 0 if k2 == 0 else i2 / k2))

        c = writeAccMeasures(F13, F131, log)

        TPs, FPs, FNs = m2
        pre = precision(TPs, FPs)
        rec = recall(TPs, FNs)
        F = F1(pre, rec)

        log.write(
            "\nError Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
                TPs, FPs, FNs, pre, rec, F))
        x2 = array([h2, i2, j2, k2, array([TPs, FPs, FNs, pre, rec, F])])

        return array([array([levTR, levRT, levTN, levNT, sizeTrue, sizeNew, sizeRan, a, x]),
                      array([levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, b, x1]),
                      array([custTR, custRT, custTN, custNT, countTrue, countNew, countRan, c, x2])])
    else:
        levTR, levRT, levTN, levNT, sizeTrue, sizeNew, sizeRan, F11, F111 = levDistanceNoNums(shape, newStatements,
                                                                                              trueStatements,
                                                                                              conceptSpace, roleSpace,
                                                                                              syn, mix)

        log.write(
            "\nNo Nums\nLevenshtein Distance From Reasoner to Random Data,{}\nLevenshtein Distance From Random to Reasoner Data,{}\nLevenshtein Distance From Reasoner to Predicted Data,{}\nLevenshtein Distance From Prediction to Reasoner Data,{}\n".format(
                levTR, levRT, levTN, levNT))
        log.write(
            "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
                levTR / sizeTrue, levRT / sizeRan, levTN / sizeTrue, 0 if sizeNew == 0 else levNT / sizeNew))

        a = writeAccMeasures(F11, F111, log)

        levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, F12, F121 = levDistance(shape, newStatements,
                                                                                               trueStatements,
                                                                                               conceptSpace, roleSpace,
                                                                                               syn, mix)

        log.write(
            "\nNums\nLevenshtein Distance From Reasoner to Random Data,{}\nLevenshtein Distance From Random to Reasoner Data,{}\nLevenshtein Distance From Reasoner to Predicted Data,{}\nLevenshtein Distance From Prediction to Reasoner Data,{}\n".format(
                levTR2, levRT2, levTN2, levNT2))
        log.write(
            "Average Levenshtein Distance From Reasoner to Random Statement,{}\nAverage Levenshtein Distance From Random to Reasoner Statement,{}\nAverage Levenshtein Distance From Reasoner to Predicted Statement,{}\nAverage Levenshtein Distance From Prediction to Reasoner Statement,{}\n".format(
                levTR2 / sizeTrue2, levRT2 / sizeRan2, levTN2 / sizeTrue2, 0 if sizeNew2 == 0 else levNT2 / sizeNew2))

        b = writeAccMeasures(F12, F121, log)

        custTR, custRT, custTN, custNT, countTrue, countNew, countRan, F13, F131 = customDistance(shape, newPreds,
                                                                                                  truePreds,
                                                                                                  conceptSpace,
                                                                                                  roleSpace, syn, mix)

        log.write(
            "\nCustom\nCustom Distance From Reasoner to Random Data,{}\nCustom Distance From Random to Reasoner Data,{}\nCustom Distance From Reasoner to Predicted Data,{}\nCustom Distance From Predicted to Reasoner Data,{}\n".format(
                custTR, custRT, custTN, custNT))
        log.write(
            "Average Custom Distance From Reasoner to Random Statement,{}\nAverage Custom Distance From Random to Reasoner Statement,{}\nAverage Custom Distance From Reasoner to Predicted Statement,{}\nAverage Custom Distance From Prediction to Reasoner Statement,{}\n".format(
                custTR / countTrue, custRT / countRan, custTN / countTrue, 0 if countNew == 0 else custNT / countNew))

        c = writeAccMeasures(F13, F131, log)

        return array([array([levTR, levRT, levTN, levNT, sizeTrue, sizeNew, sizeRan, a]),
                      array([levTR2, levRT2, levTN2, levNT2, sizeTrue2, sizeNew2, sizeRan2, b]),
                      array([custTR, custRT, custTN, custNT, countTrue, countNew, countRan, c])])


def trainingStats(log, mseNew, mse0, mseL):
    log.write(
        "Training Statistics\nPrediction Mean Squared Error,{}\nLearned Reduction MSE,{}\nIncrease MSE on Test,{}\nTraining Percent Change MSE,{}\n".format(
            numpy.float32(mseNew), mse0 - mseL, numpy.float32(mseNew) - mseL, (mseL - mse0) / mse0 * 100))


def shallowSystem(n_epochs0, learning_rate0, trainlog, evallog, conceptSpace, roleSpace, allTheData, syn, mix, n):
    KBs_test, KBs_train, X_train, X_test, y_train, y_test, truePreds, trueStatements, labels, errPreds, errStatements = allTheData

    trainlog.write("Piecewise LSTM Part One\nEpoch,Mean Squared Error,Root Mean Squared Error\n")
    evallog.write("Piecewise LSTM Part One\n")
    print("")

    n_neurons0 = X_train.shape[2]

    X0 = tf.placeholder(tf.float32, shape=[None, KBs_train.shape[1], KBs_train.shape[2]])
    y0 = tf.placeholder(tf.float32, shape=[None, X_train.shape[1], X_train.shape[2]])

    outputs0, states0 = tf.nn.dynamic_rnn(tf.nn.rnn_cell.LSTMCell(num_units=n_neurons0), X0, dtype=tf.float32)

    loss0 = tf.losses.mean_squared_error(y0, outputs0)
    optimizer0 = tf.train.AdamOptimizer(learning_rate=learning_rate0)
    training_op0 = optimizer0.minimize(loss0)

    # saver = tf.train.Saver()

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
        newPreds, newStatements = vecToStatements(y_pred, conceptSpace, roleSpace)

        trainingStats(evallog, mseNew, mse0, mseL)

        writeVectorFile("crossValidationFolds/{}output/learnedSupportsP[{}].txt".format("" if syn else "sn", n),
                        newStatements)

        numpy.savez("crossValidationFolds/{}saves/halfwayData[{}]".format("" if syn else "s", n), y_pred)

        # saver.save(sess,"{}{}saves/firstHalfModel[{}]".format("" if n == 1 else "crossValidationFolds/","" if syn else "s",n))

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

    # saver = tf.train.Saver()

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

        trainingStats(evallog, mseNew, mse0, mseL)

        print("\nEvaluating Result")

        evallog.write("\nReasoner Support Test Data Evaluation\n")

        newPreds, newStatements = vecToStatements(y_pred, conceptSpace, roleSpace)

        writeVectorFile(
            "crossValidationFolds/{}output/predictedOutLeftOverSupportTest[{}].txt".format("" if syn else "sn", n),
            newStatements)

        evals = distanceEvaluations(evallog, y_pred.shape, newPreds, truePreds, newStatements, trueStatements,
                                    conceptSpace, roleSpace, syn, mix, False, False)

        data = numpy.load("crossValidationFolds/{}saves/halfwayData[{}].npz".format("" if syn else "s", n),
                          allow_pickle=True)
        data = data['arr_0']

        evallog.write("\nSaved Test Data From Previous LSTM Evaluation\n")

        y_pred = sess.run(outputs1, feed_dict={X1: data})
        mseNew = loss1.eval(feed_dict={outputs1: y_pred, y1: y_test})

        evallog.write("\nTesting Statistic\nIncrease MSE on Saved,{}\n".format(numpy.float32(mseNew) - mseL))

        newPreds, newStatements = vecToStatementsWithLabels(y_pred, conceptSpace, roleSpace, labels) if (
                    not mix and not syn) else vecToStatements(y_pred, conceptSpace, roleSpace)

        writeVectorFile(
            "crossValidationFolds/{}output/predictionSavedKBPipeline[{}].txt".format("" if syn else "sn", n),
            newStatements)

        if (not mix and not syn):
            newPreds, newStatements = vecToStatements(y_pred, conceptSpace, roleSpace)

        # saver.save(sess,"{}{}saves/secondHalfModel[{}]".format("" if n == 1 else "crossValidationFolds/","" if syn else "s",n))

        return distanceEvaluations(evallog, y_pred.shape, newPreds, truePreds, newStatements, trueStatements,
                                   conceptSpace, roleSpace, syn, mix, errPreds, errStatements)


def deepSystem(n_epochs2, learning_rate2, trainlog, evallog, conceptSpace, roleSpace, allTheData, syn, mix, n):
    KBs_test, KBs_train, X_train, X_test, y_train, y_test, truePreds, trueStatements, labels, errPreds, errStatements = allTheData

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

    # saver = tf.train.Saver()

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

        trainingStats(evallog, mseNew, mse0, mseL)

        evallog.write("\nTest Data Evaluation\n")

        newPreds, newStatements = vecToStatementsWithLabels(y_pred, conceptSpace, roleSpace, labels) if (
                    not mix and not syn) else vecToStatements(y_pred, conceptSpace, roleSpace)

        writeVectorFile(
            "crossValidationFolds/{}output/predictionDeepArchitecture[{}].txt".format("" if syn else "sn", n),
            newStatements)

        if (not mix and not syn):
            newPreds, newStatements = vecToStatements(y_pred, conceptSpace, roleSpace)

            # saver.save(sess,"{}{}saves/deepModel[{}]".format("" if n == 1 else "crossValidationFolds/","" if syn else "s",n))

        return distanceEvaluations(evallog, y_pred.shape, newPreds, truePreds, newStatements, trueStatements,
                                   conceptSpace, roleSpace, syn, mix, errPreds, errStatements)


def flatSystem(n_epochs2, learning_rate2, trainlog, evallog, conceptSpace, roleSpace, allTheData, syn, mix, n):
    KBs_test, KBs_train, X_train, X_test, y_train, y_test, truePreds, trueStatements, labels, errPreds, errStatements = allTheData

    trainlog.write("Flat LSTM\nEpoch,Mean Squared Error,Root Mean Squared Error\n")
    evallog.write("\nFlat LSTM\n\n")
    print("")

    X0 = tf.placeholder(tf.float32, shape=[None, KBs_train.shape[1], KBs_train.shape[2]])
    y1 = tf.placeholder(tf.float32, shape=[None, y_train.shape[1], y_train.shape[2]])

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
        "\nAverage Random Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            rTPs2, rFPs2, rFNs2, rpre2, rrec2, rF2))


def nTimesCrossValidate(n, epochs, learningRate, conceptSpace, roleSpace, syn, mix, pert):
    if not os.path.isdir("crossValidationFolds"): os.mkdir("crossValidationFolds")
    if syn and not os.path.isdir("crossValidationFolds/output"): os.mkdir("crossValidationFolds/output")
    if (not syn or mix) and not os.path.isdir("crossValidationFolds/snoutput"): os.mkdir(
        "crossValidationFolds/snoutput")
    if not os.path.isdir("crossValidationFolds/training"): os.mkdir("crossValidationFolds/training")
    if not os.path.isdir("crossValidationFolds/evals"): os.mkdir("crossValidationFolds/evals")
    if not os.path.isdir("crossValidationFolds/{}saves".format("" if syn else "s")): os.mkdir(
        "crossValidationFolds/{}saves".format("" if syn else "s"))

    if os.path.isfile("{}saves/{}foldData{}{}.npz".format("" if syn else "s", n, "Mixed" if mix else "",
                                                          "Err[{}]".format(str(pert)) if pert >= 0 else "")):
        data = numpy.load("{}saves/{}foldData{}{}.npz".format("" if syn else "s", n, "Mixed" if mix else "",
                                                              "Err[{}]".format(str(pert)) if pert >= 0 else ""),
                          allow_pickle=True)
        allTheData = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'], data[
            'arr_6'], data['arr_7'], data['arr_8'], data['arr_9'], data['arr_10']
    elif syn:
        if pert >= 0 and os.path.isfile("saves/messData{}.npz".format(pert)):
            data = numpy.load("saves/messData{}.npz".format(pert), allow_pickle=True)
            KBs, supports, outputs, mKBs, mouts = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data[
                'arr_4']
        else:
            KBs, supports, outputs = getSynDataFromFile('saves/data.npz')
            mKBs = None;
            mouts = None
        if mix and pert == 0:
            sKBs, ssupports, soutputs, localMaps, stats = getSnoDataFromFile('ssaves/data.npz')
            labels = collapseLabelMap(localMaps, stats[0][2], stats[1][2], stats[4][1])
            allTheData = crossValidationSplitAllData(n, KBs, supports, outputs, sKBs, ssupports, soutputs, labels, mKBs,
                                                     mouts, conceptSpace, roleSpace, syn, mix, pert)
        else:
            allTheData = crossValidationSplitAllData(n, KBs, supports, outputs, None, None, None, None, mKBs, mouts,
                                                     conceptSpace, roleSpace, syn, mix, pert)
    else:
        if pert >= 0 and os.path.isfile("ssaves/messData{}.npz".format(pert)):
            data = numpy.load("ssaves/messData{}.npz".format(pert), allow_pickle=True)
            KBs, supports, outputs, localMaps, stats, mKBs, mouts = data['arr_0'], data['arr_1'], data['arr_2'], data[
                'arr_3'], data['arr_4'], data['arr_5'], data['arr_6']
        else:
            KBs, supports, outputs, localMaps, stats = getSnoDataFromFile('ssaves/data.npz')
            mKBs = None
            mouts = None
        labels = collapseLabelMap(localMaps, stats[0][2], stats[1][2], stats[4][1])
        if mix and pert == 0:
            sKBs, ssupports, soutputs = getSynDataFromFile('saves/data.npz')
            allTheData = crossValidationSplitAllData(n, KBs, supports, outputs, sKBs, ssupports, soutputs, labels, mKBs,
                                                     mouts, conceptSpace, roleSpace, syn, mix, pert)
        else:
            allTheData = crossValidationSplitAllData(n, KBs, supports, outputs, None, None, None, labels, mKBs, mouts,
                                                     conceptSpace, roleSpace, syn, mix, pert)

    KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, truePredss, trueStatementss, labelss, nErrsPreds, nErrStatements = allTheData

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

        writeFinalAverageDataMess(avgResult[0], log)

        log.write("\nDeep System\n")

        writeFinalAverageDataMess(avgResult[1], log)

        log.write("\nFlat System\n")

        writeFinalAverageDataMess(avgResult[2], log)

    log.close()

    print("\nDone")

    return avgResult


def crossValidationSplitAllData(n, KBs, supports, outputs, sKBs, ssupports, soutputs, localMaps, mKBs, mouts,
                                conceptSpace, roleSpace, syn, mix, pert):
    maxout = None if not isinstance(mouts, numpy.ndarray) else len(max(mouts, key=lambda coll: len(coll))[0])

    fileShapes1 = [len(supports[0]), len(max(supports, key=lambda coll: len(coll[0]))[0]),
                   len(max(outputs, key=lambda coll: len(coll[0]))[0])]
    fileShapes2 = [len(ssupports[0]), len(max(ssupports, key=lambda coll: len(coll[0]))[0]),
                   len(max(soutputs, key=lambda coll: len(coll[0]))[0])] if mix else [0, 0, 0]

    print("Repeating KBs")
    newKBs = numpy.empty([KBs.shape[0], fileShapes1[0], KBs.shape[1]], dtype=float)
    for i in range(len(newKBs)):
        for j in range(fileShapes1[0]):
            newKBs[i][j] = KBs[i]

    KBs = newKBs

    print("Shuffling Split Indices")
    indexes = list(range(len(KBs)))
    random.shuffle(indexes)
    k, m = divmod(len(indexes), n)
    indexes = list(indexes[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    crossKBsTest = numpy.zeros((len(indexes), len(indexes[0]), len(KBs[0]), len(KBs[0][0])), dtype=float)
    crossSupportsTest = numpy.zeros(
        (len(indexes), len(indexes[0]), len(supports[0]), fileShapes1[1] if not mix else fileShapes2[1]), dtype=float)
    crossOutputsTest = numpy.zeros(
        (len(indexes), len(indexes[0]), len(outputs[0]), fileShapes1[2] if not mix else fileShapes2[2]), dtype=float)
    crossErrTest = None if not isinstance(mKBs, numpy.ndarray) else numpy.zeros(
        (len(indexes), len(indexes[0]), len(mouts[0]), maxout), dtype=float)

    if isinstance(localMaps, numpy.ndarray):
        crossLabels = numpy.empty((len(indexes), len(indexes[0])), dtype=dict)
    else:
        crossLabels = None

    print("Extracting Test Sets")
    for i in range(len(indexes)):
        KBns = []
        for j in range(len(indexes[i])):
            if not mix:
                if pert >= 0.0 and isinstance(mKBs, numpy.ndarray):
                    crossKBsTest[i][j] = mKBs[indexes[i][j]]
                    placeholder, KBn = vecToStatement(KBs[indexes[i][j]], conceptSpace, roleSpace)
                    KBns.append(KBn)
                    crossErrTest[i][j] = numpy.hstack(
                        [mouts[indexes[i][j]], numpy.zeros([fileShapes1[0], maxout - len(mouts[indexes[i][j]][0])])])
                else:
                    crossKBsTest[i][j] = KBs[indexes[i][j]]
                crossSupportsTest[i][j] = numpy.hstack([supports[indexes[i][j]], numpy.zeros(
                    [fileShapes1[0], fileShapes1[1] - len(supports[indexes[i][j]][0])])])
                crossOutputsTest[i][j] = numpy.hstack([outputs[indexes[i][j]], numpy.zeros(
                    [fileShapes1[0], fileShapes1[2] - len(outputs[indexes[i][j]][0])])])
            else:
                crossKBsTest[i][j] = sKBs[indexes[i][j]]
                crossSupportsTest[i][j] = numpy.hstack([ssupports[indexes[i][j]], numpy.zeros(
                    [fileShapes2[0], fileShapes2[1] - len(ssupports[indexes[i][j]][0])])])
                crossOutputsTest[i][j] = numpy.hstack([soutputs[indexes[i][j]], numpy.zeros(
                    [fileShapes2[0], fileShapes2[2] - len(soutputs[indexes[i][j]][0])])])
            if isinstance(localMaps, numpy.ndarray):
                crossLabels[i][j] = localMaps[indexes[i][j]]
        writeVectorFile("crossValidationFolds/{}output/originalKBsIn[{}].txt".format("sn" if not syn else "", i),
                        array(KBns))

    crossKBsTrain = numpy.zeros((n, len(KBs) - len(crossKBsTest[0]), len(KBs[0]), len(KBs[0][0])), dtype=float)
    crossOutputsTrain = numpy.zeros((n, len(KBs) - len(crossKBsTest[0]), len(outputs[0]), fileShapes1[2]), dtype=float)
    crossSupportsTrain = numpy.zeros((n, len(KBs) - len(crossKBsTest[0]), len(supports[0]), fileShapes1[1]),
                                     dtype=float)

    print("Extracting Train Sets")
    for i in range(len(indexes)):
        for j in range(len(crossKBsTrain[i])):
            train = list(set(range(len(KBs))).difference(set(indexes[i])))
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


def readInputs():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", help="number of epochs for each system", type=int, default=100)
    parser.add_argument("-l", "--learningRate", help="learning rate of each system", type=float, default=0.0001)
    parser.add_argument("-s", "--snomed", help="use SNOMED dataset", action="store_true", default=True)
    parser.add_argument("-m", "--mix", help="use test set from different souce than train", action="store_true",
                        default=False)
    parser.add_argument("-c", "--cross", help="cross validation k", type=int, default=2)
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


def runAllTests():
    if not os.path.isdir("Tests"): os.mkdir("Tests")
    if not os.path.isdir("Tests/Syn"): os.mkdir("Tests/Syn")
    if not os.path.isdir("Tests/Sno"): os.mkdir("Tests/Sno")

    for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print("{} percent corruption\nSynthetic Data".format(i * 10))
        nTimesCrossValidate(n=10, epochs=20000, learningRate=0.0001, conceptSpace=21, roleSpace=8, syn=True, mix=False,
                            pert=i)
        shutil.move("crossValidationFolds", "Tests/Syn", copy_function=shutil.copytree)
        os.rename("Tests/Syn/crossValidationFolds", "Tests/Syn/crossValidationFolds{}".format(i))

    for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print("{} percent corruption\nSNOMED Data".format(i * 10))
        nTimesCrossValidate(n=10, epochs=20000, learningRate=0.0001, conceptSpace=21, roleSpace=8, syn=False, mix=False,
                            pert=i)
        shutil.move("crossValidationFolds", "Tests/Sno", copy_function=shutil.copytree)
        os.rename("Tests/Sno/crossValidationFolds", "Tests/Sno/crossValidationFolds{}".format(i))


if __name__ == "__main__":
    args = readInputs()

    nTimesCrossValidate(n=args.cross, epochs=args.epochs, learningRate=args.learningRate, conceptSpace=21, roleSpace=8,
                        syn=not args.snomed, mix=args.mix, pert=args.perturb)
