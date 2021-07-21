import numpy

def findBestMatchNoNums(statement, reasonerSteps):
    return min(map(partial(levenshteinIgnoreNum, statement), reasonerSteps))


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


def findBestPredMatch(statement, otherKB, conceptSpace, roleSpace):
    return min(map(partial(custom, conceptSpace, roleSpace, statement), otherKB))


def levDistance(shape, newStatements, trueStatements):
    if (syn and os.path.isfile("saves/randoStr.npz")) or os.path.isfile("ssaves/randoStr.npz"):
        rando = numpy.load("saves/randoStr.npz" if syn else "ssaves/randoStr.npz", allow_pickle=True)
        rando = rando['arr_0'].tolist()
    else:
        rando = makeRandomStrCompletions(shape, conceptSpace, roleSpace, syn)

    flatTrue = [[item for sublist in x for item in sublist] for x in trueStatements]
    flatRand = [[item for sublist in x for item in sublist] for x in rando]
    flatNew = [[item for sublist in x for item in sublist] for x in newStatements]

    F1s = numpy.array([0, 0, 0])
    rF1s = numpy.array([0, 0, 0])

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


# ----------------------------------------------------------------------------------------------------------------------
def write_evaluation_measures(F, rF, log):
    TPs, FPs, FNs = F
    pre = precision(TPs, FPs)
    rec = recall(TPs, FNs)
    F = F1(pre, rec)

    log.write(
        "\nPrediction Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs, FPs, FNs, pre, rec, F))

    x = numpy.array([TPs, FPs, FNs, pre, rec, F])

    TPs, FPs, FNs = rF
    pre = precision(TPs, FPs)
    rec = recall(TPs, FNs)
    F = F1(pre, rec)

    log.write(
        "\nRandom Accuracy For this Distance Measure\nTrue Positives,{}\nFalse Positives,{}\nFalse Negatives,{}\nPrecision,{}\nRecall,{}\nF1 Score,{}\n".format(
            TPs, FPs, FNs, pre, rec, F))

    return numpy.array([x, numpy.array([TPs, FPs, FNs, pre, rec, F])])


def precision(TP, FP):
    return 0 if TP == 0 and FP == 0 else TP / (TP + FP)


def recall(TP, FN):
    return 0 if TP == 0 and FN == 0 else TP / (TP + FN)


def F1(precision, recall):
    return 0 if precision == 0 and recall == 0 else 2 * (precision * recall) / (precision + recall)
