import os
from unittest import TestCase
import numpy as np


class DistanceEvalTests(TestCase):

    def test_custom(self):
        from main import custom as cust
        guess = [("C1", "R1", "C1"), ("C1", "R2", "C1"), ("C200", "R19", "C99"), ("C100", "R1", "R16")]
        true = [("C2", "R1", "C1"), ("C1", "C15", "C1"), ("C100", "R1", "C9"), ("R20", "R1", "C84")]
        distExpect = [1, 17, 208, 220]
        for i in range(len(guess)):
            self.assertEqual(distExpect[i], cust(guess[i], true[i]))

    # def test_custom_distance(self):
    #     from main import custom_distance
    #     shape = (1,2,3)
    #     newPred = [
    #         [
    #             [('C1', 'R2', 'C6'), ('C18', 'R8', 'C12')],
    #             [('C2', 'C1', 'C13'), ('C7', 'R12', 'C9')]
    #         ]
    #     ]
    #     trueLabels = [
    #         [
    #             [('C1', 'R1', 'C3'), ('C12', 'R4', 'C1')],
    #             [('C4', 'R1', 'C8')]
    #         ]
    #     ]
    #     # 22
    #     # 8
    #     custom_distance(shape, newPred, trueLabels, 0, 0)

class EvaluationMetricTests(TestCase):

    def test_precision(self):
        from main import precision as p
        tp = [1, 3, 47, 10500, 0, 20000, 0]
        fp = [4, 61, 17, 14500, 20678, 0, 0]
        answers = [.2, 0.046875, 0.734375, 0.42, 0, 1, 0]
        for x in range(len(tp)):
            self.assertEqual(answers[x], p(tp[x], fp[x]))

    def test_recall(self):
        from main import precision as r
        tp = [1, 3, 47, 10500, 0, 20000, 0]
        fn = [4, 61, 17, 14500, 20678, 0, 0]
        answers = [.2, 0.046875, 0.734375, 0.42, 0, 1, 0]
        for x in range(len(tp)):
            self.assertEqual(answers[x], r(tp[x], fn[x]))

    def test_F1(self):
        from main import F1 as f1
        prec = [0, 0, 1, 2, .044, .0022]
        recall = [0, 1, 0, 2, .16, .035]
        answers = [0, 0, 0, 2, 0.06901961, 0.00413978]
        for x in range(len(prec)):
            self.assertEqual(answers[x], round(f1(prec[x], recall[x]), 8))

class LoggingFunctionTests(TestCase):
    def test_training_stats(self):
        from main import training_stats as ts
        ts(open("test_training_stats.txt", "w"), 1, 1, 1)
        # Ouput should be 1.0, 0, 0.0, 0.0

    def test_write_evaluation_measures(self):
        from main import write_evaluation_measures as wem
        x = wem((3, 1, 1), open("test_write_evaluation_measures.txt", "w"))
        self.assertEqual(x.shape, (6,))
        self.assertTrue(x[0] == 3 and x[1] == 1 and x[2] == 1)
        self.assertTrue(x[3] == .75 and x[4] == .75 and x[5] == .75)

    def test_write_final_average_data(self):
        from main import write_final_average_data as wfinalav
        wfinalav(((1, 1, 2, 2, (0, 0, 0, 0, 0, 0)), (11, 11, 22, 22, (0, 0, 0, 0, 0, 0))),
                 open("test_write_final_average_data.txt", "w"))

class DataWranglingTests(TestCase):

    def test_get_labels_from_encoding_handles_true_and_pred_values_the_same(self):
        from main import get_labels_from_encoding, convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, encodingMap = convert_data_to_arrays(get_rdf_data('rdfData/Descriptions_and_Situations.json'))

        allTheData = cross_validation_split_all_data(5, KB, supports, outputs, encodingMap)

        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, labelss = allTheData

        trueArr, predArr = get_labels_from_encoding(y_tests[0], y_tests[0], 28, 14)

        self.assertEqual(trueArr.all(), predArr.all())

    def test_get_labels_from_encoding_each_sample_has_correct_size(self):
        from main import get_labels_from_encoding, convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, encodingMap = convert_data_to_arrays(get_rdf_data('rdfData/Descriptions_and_Situations.json'))

        allTheData = cross_validation_split_all_data(5, KB, supports, outputs, encodingMap)

        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, labelss = allTheData

        trueArr, predArr = get_labels_from_encoding(y_tests[0], y_tests[0], 28, 14)

        outs = y_tests[0].tolist()

        for sample in range(len(trueArr)):
            for ts in range(len(trueArr[sample])):
                li = outs[sample][ts]
                if li.count(0.0) > 0:
                    outs[sample][ts] = li[:li.index(0.0)]
                if len(outs[sample][ts])/3 != len(trueArr[sample][ts]):
                    print("")

    def test_convert_encoded_item_to_label(self):
        from main import convert_encoded_float_to_label
        decodeNonProperty = 14
        decodeProperty = 7
        floats = [-0.142857143, -0.714285714, -1, 0.071428571, 0.428571429, 1, 0.000001, -0.0001, 1.0234, -2.856]
        labels = ['R1', 'R5', 'R7', 'C1', 'C6', 'C14', '0', '0', 'C14', 'R7']

        for i in range(len(floats)):
            self.assertEqual(labels[i], convert_encoded_float_to_label(floats[i], decodeNonProperty, decodeProperty))

    def test_pad_kb_everyone_same_size(self):
        from main import get_rdf_data, pad_kb
        data = get_rdf_data('rdfData/Descriptions_and_Situations.json')
        kb = data['kB']
        kb = pad_kb(kb)

        for sample in kb:
            self.assertEqual(len(kb[0]), len(sample))

    def test_pad_kb_removing_padding_reveals_original(self):
        from main import get_rdf_data, pad_kb
        data = get_rdf_data('rdfData/Descriptions_and_Situations.json')
        kb = data['kB']
        KB = pad_kb(kb)

        for sample in KB:
            for index in range(len(sample)):
                if sample[index] == 0.0:
                    del sample[index:]
                    break
        self.assertEqual(kb, KB)

    def test_pad_list_of_lists_same_size(self):
        from main import get_rdf_data as getData
        from main import pad_list_of_lists as testMethod

        data = getData('rdfData/gfo.json')
        list = testMethod(data['supports'])

        for i in range(len(list)):
            element = list[i]
            for j in range(len(element)-1):
                self.assertEqual(len(element[j]), len(element[j+1]))

    def test_cross_validation_split_all_data_correct_KB_repetition(self):
        from main import convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, encodingMap = convert_data_to_arrays(get_rdf_data("rdfData/gfo.json"))

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, labelss = \
            cross_validation_split_all_data(8, KB, supports, outputs, encodingMap)

        tests = KBs_tests.tolist()
        for cross in tests:
            for sample in cross:
                standard = sample[0]
                for ts in sample:
                    self.assertTrue(standard == ts)

        trains = KBs_trains.tolist()
        for cross in trains:
            for sample in cross:
                standard = sample[0]
                for ts in sample:
                    self.assertTrue(standard == ts)

    def test_cross_validation_split_all_data_no_kb_data_lost(self):
        from main import convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, encodingMap = convert_data_to_arrays(get_rdf_data("rdfData/gfo.json"))

        numConcepts, numRoles = 28, 14

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, labelss = \
            cross_validation_split_all_data(5, KB, supports, outputs, encodingMap)

        expected = KB.tolist()
        trains = KBs_trains[0].tolist()
        trains.extend(KBs_tests[0].tolist())
        actual = trains

        for sample in range(len(actual)):
            actual[sample] = actual[sample][0]

        for sample in range(len(expected)):
            for t_sample in range(len(actual)):
                if expected[sample] == actual[t_sample]:
                    expected[sample] = None
                    actual[t_sample] = None
                    break

        self.assertEqual(actual, expected)

    def test_cross_validation_split_all_data_no_support_data_lost(self):
        from main import convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, encodingMap = convert_data_to_arrays(get_rdf_data("rdfData/gfo.json"))

        numConcepts, numRoles = 28, 14

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, labelss = \
            cross_validation_split_all_data(7, KB, supports, outputs, encodingMap)

        expected = []
        for sample in supports:
            expected.append(sample.tolist())

        padding_count = 0
        for sample in expected:
            if len(sample[0]) > padding_count:
                padding_count = len(sample[0])

        for sampleNum in range(len(expected)):
            for timestepNum in range(len(expected[sampleNum])):
                while len(expected[sampleNum][timestepNum]) < padding_count:
                    expected[sampleNum][timestepNum].append(0.0)

        trains = X_trains[0].tolist()
        trains.extend(X_tests[0].tolist())
        actual = trains

        for sample in range(len(expected)):
            for t_sample in range(len(actual)):
                if expected[sample] == actual[t_sample]:
                    expected[sample] = None
                    actual[t_sample] = None
                    break

        self.assertEqual(actual, expected)

    def test_cross_validation_split_all_data_no_output_data_lost(self):
        from main import convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, encodingMap = convert_data_to_arrays(get_rdf_data("rdfData/gfo.json"))

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, labelss = \
            cross_validation_split_all_data(7, KB, supports, outputs, encodingMap)

        expected = []
        for sample in outputs:
            expected.append(sample.tolist())

        padding_count = 0
        for sample in expected:
            if len(sample[0]) > padding_count:
                padding_count = len(sample[0])

        for sampleNum in range(len(expected)):
            for timestepNum in range(len(expected[sampleNum])):
                while len(expected[sampleNum][timestepNum]) < padding_count:
                    expected[sampleNum][timestepNum].append(0.0)

        trains = y_trains[0].tolist()
        trains.extend(y_tests[0].tolist())
        actual = trains

        for sample in range(len(expected)):
            for t_sample in range(len(actual)):
                if expected[sample] == actual[t_sample]:
                    expected[sample] = None
                    actual[t_sample] = None
                    break

        self.assertEqual(actual, expected)

    def test_cross_validation_split_all_data_correct_mapping_fromKBToSuppToOuts_of_training(self):
        from main import convert_data_to_arrays, get_rdf_data, \
            cross_validation_split_all_data

        KB, supports, outputs, encodingMap = convert_data_to_arrays(get_rdf_data("rdfData/gfo.json"))

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, labelss = \
            cross_validation_split_all_data(3, KB, supports, outputs, encodingMap)

        trueKB = KB.tolist()
        crossKB = KBs_trains[0].tolist()

        # Outputs ----------------------
        trueOuts = []
        for sample in outputs:
            trueOuts.append(sample.tolist())

        padding_count = 0
        for sample in trueOuts:
            if len(sample[0]) > padding_count:
                padding_count = len(sample[0])

        for sampleNum in range(len(trueOuts)):
            for timestepNum in range(len(trueOuts[sampleNum])):
                while len(trueOuts[sampleNum][timestepNum]) < padding_count:
                    trueOuts[sampleNum][timestepNum].append(0.0)

        actualOuts = y_trains[0].tolist()

        # Supports ---------------------
        trueSupp = []
        for sample in supports:
            trueSupp.append(sample.tolist())

        padding_count = 0
        for sample in trueSupp:
            if len(sample[0]) > padding_count:
                padding_count = len(sample[0])

        for sampleNum in range(len(trueSupp)):
            for timestepNum in range(len(trueSupp[sampleNum])):
                while len(trueSupp[sampleNum][timestepNum]) < padding_count:
                    trueSupp[sampleNum][timestepNum].append(0.0)

        actualSupp = X_trains[0].tolist()

        for sample in range(len(crossKB)):
            for t_sample in range(len(trueKB)):
                if crossKB[sample][0] == trueKB[t_sample]:
                    self.assertTrue(actualOuts[sample] == trueOuts[t_sample])
                    self.assertTrue(actualSupp[sample] == trueSupp[t_sample])

    def test_cross_validation_split_all_data_correct_mapping_fromKBToSuppToOuts_of_test(self):
        from main import convert_data_to_arrays, get_rdf_data, \
            cross_validation_split_all_data

        KB, supports, outputs, encodingMap = convert_data_to_arrays(get_rdf_data("rdfData/gfo.json"))

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, labelss = \
            cross_validation_split_all_data(3, KB, supports, outputs, encodingMap)

        trueKB = KB.tolist()
        crossKB = KBs_tests[0].tolist()

        # Outputs ----------------------
        trueOuts = []
        for sample in outputs:
            trueOuts.append(sample.tolist())

        padding_count = 0
        for sample in trueOuts:
            if len(sample[0]) > padding_count:
                padding_count = len(sample[0])

        for sampleNum in range(len(trueOuts)):
            for timestepNum in range(len(trueOuts[sampleNum])):
                while len(trueOuts[sampleNum][timestepNum]) < padding_count:
                    trueOuts[sampleNum][timestepNum].append(0.0)

        actualOuts = y_tests[0].tolist()

        # Supports ---------------------
        trueSupp = []
        for sample in supports:
            trueSupp.append(sample.tolist())

        padding_count = 0
        for sample in trueSupp:
            if len(sample[0]) > padding_count:
                padding_count = len(sample[0])

        for sampleNum in range(len(trueSupp)):
            for timestepNum in range(len(trueSupp[sampleNum])):
                while len(trueSupp[sampleNum][timestepNum]) < padding_count:
                    trueSupp[sampleNum][timestepNum].append(0.0)

        actualSupp = X_tests[0].tolist()

        for sample in range(len(crossKB)):
            for t_sample in range(len(trueKB)):
                if crossKB[sample][0] == trueKB[t_sample]:
                    self.assertTrue(actualOuts[sample] == trueOuts[t_sample])
                    self.assertTrue(actualSupp[sample] == trueSupp[t_sample])

    def test_cross_validation_split_all_data_correct_numOf_folds_returned(self):
        from main import convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, encodingMap = convert_data_to_arrays(get_rdf_data("rdfData/gfo.json"))

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, labelss = \
            cross_validation_split_all_data(7, KB, supports, outputs, encodingMap)

        self.assertEqual(7, KBs_tests.shape[0])
        self.assertEqual(7, KBs_trains.shape[0])
        self.assertEqual(7, X_tests.shape[0])
        self.assertEqual(7, X_trains.shape[0])
        self.assertEqual(7, y_tests.shape[0])
        self.assertEqual(7, y_tests.shape[0])