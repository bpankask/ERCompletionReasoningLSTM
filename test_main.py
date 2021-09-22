import os
from unittest import TestCase
import numpy as np


class DistanceEvalTests(TestCase):

    def test_custom__synthetic_data(self):
        from main import custom as cust
        guess = [("C1", "R1", "C1"), ("C1", "R2", "C1"), ("C200", "R19", "C99"), ("C100", "R1", "R16")]
        true = [("C2", "R1", "C1"), ("C1", "C15", "C1"), ("C100", "R1", "C9"), ("R20", "R1", "C84")]
        distExpect = [1, 17, 208, 220]
        for i in range(len(guess)):
            self.assertEqual(distExpect[i], cust(guess[i], true[i]))

    def test_find_best_prediction_custom__synthetic_data_full_lists(self):
        from main import find_best_prediction_custom
        tuples1 = [('C1', 'R1', 'C1'), ('C2', 'R2', 'C2'), ('C3', 'R3', 'C3')]
        tuples2 = [
            [('C10', 'R10', 'C10'), ('C2', 'R2', 'C2'), ('C30', 'R30', 'C30')],  # dist = 3
            [('C1', 'R1', 'C1'), ('C2', 'R2', 'C2'), ('C3', 'R3', 'C3')],  # dist = 0
            [('C10', 'R15', 'C12'), ('C15', 'R16', 'C20'), ('R3', 'R3', 'C3')]]  # dist = 6
        results = [3, 0, 6]

        for i in range(len(tuples1)):
            self.assertEqual(results[i], find_best_prediction_custom(tuples1[i], tuples2[i]))

    def test_find_best_prediction_custom__synthetic_data_empty_list(self):
        from main import find_best_prediction_custom
        tuples1 = [('C1', 'R1', 'C1'), ('C2', 'R2', 'C2'), ('C3', 'R3', 'C3')]

        results = [3, 6, 9]

        for i in range(len(tuples1)):
            self.assertEqual(results[i], find_best_prediction_custom(tuples1[i], []))

    def test_custom_distance__synthetic_data_predictions_are_empty(self):
        from main import custom_distance
        import numpy as np

        # ('R1', 'R1', 'C6')
        pred = [[list([]), list([]), list([]), list([]), list([]), list([])],
                [list([]), list([]), list([]), list([]), list([]), list([])]]
        pred = np.array(pred)

        labels = [[list([('C1', 'R1', 'C2'), ('C3', 'R2', 'C4'), ('C5', 'R3', 'C6')]),
                        list([('C7', 'R1', 'C8'), ('C9', 'R1', 'C10'), ('C11', 'R4', 'C12')]),
                        list([('C13', 'R5', 'C10'), ('C14', 'R5', 'C14')]),
                        list([('C15', 'R5', 'C10'), ('C16', 'R1', 'C2')]),
                        list([('C17', 'R1', 'C2'), ('C15', 'R1', 'C2')]),
                        list([('C18', 'R5', 'C18'), ('C19', 'R5', 'C19')])],
                  [list([('C1', 'R1', 'C2'), ('C3', 'R2', 'C4'), ('C5', 'R3', 'C6')]),
                       list([('C7', 'R1', 'C8'), ('C9', 'R1', 'C10'), ('C11', 'R4', 'C12')]),
                       list([('C13', 'R5', 'C10'), ('C14', 'R5', 'C14')]),
                       list([('C15', 'R5', 'C10'), ('C16', 'R1', 'C2')]),
                       list([('C17', 'R1', 'C2'), ('C15', 'R1', 'C2')]),
                       list([('C18', 'R5', 'C18'), ('C19', 'R5', 'C19')])]
                  ]
        labels = np.array(labels)

        shape = (2, 6, 3)

        custTN, custNT, countTrue, countNew, evalInfoNew = custom_distance(shape, pred, labels)
        self.assertEqual(644, custTN)
        self.assertEqual(0, custNT)
        self.assertEqual(28, countTrue)
        self.assertEqual(0, countNew)

        self.assertEqual(0, evalInfoNew[0])
        self.assertEqual(0, evalInfoNew[1])
        self.assertEqual(28, evalInfoNew[2])

    def test_custom_distance__synthetic_data_truelabels_are_empty(self):
        from main import custom_distance
        import numpy as np

        # ('R1', 'R1', 'C6')
        labels = [[list([]), list([]), list([]), list([]), list([]), list([])],
                [list([]), list([]), list([]), list([]), list([]), list([])]]
        labels = np.array(labels)

        pred = [[list([('C1', 'R1', 'C2'), ('C3', 'R2', 'C4'), ('C5', 'R3', 'C6')]),
                        list([('C7', 'R1', 'C8'), ('C9', 'R1', 'C10'), ('C11', 'R4', 'C12')]),
                        list([('C13', 'R5', 'C10'), ('C14', 'R5', 'C14')]),
                        list([('C15', 'R5', 'C10'), ('C16', 'R1', 'C2')]),
                        list([('C17', 'R1', 'C2'), ('C15', 'R1', 'C2')]),
                        list([('C18', 'R5', 'C18'), ('C19', 'R5', 'C19')])],
                  [list([('C1', 'R1', 'C2'), ('C3', 'R2', 'C4'), ('C5', 'R3', 'C6')]),
                       list([('C7', 'R1', 'C8'), ('C9', 'R1', 'C10'), ('C11', 'R4', 'C12')]),
                       list([('C13', 'R5', 'C10'), ('C14', 'R5', 'C14')]),
                       list([('C15', 'R5', 'C10'), ('C16', 'R1', 'C2')]),
                       list([('C17', 'R1', 'C2'), ('C15', 'R1', 'C2')]),
                       list([('C18', 'R5', 'C18'), ('C19', 'R5', 'C19')])]
                  ]
        pred = np.array(pred)

        shape = (2, 6, 3)

        custTN, custNT, countTrue, countNew, evalInfoNew = custom_distance(shape, pred, labels)
        self.assertEqual(0, custTN)
        self.assertEqual(644, custNT)
        self.assertEqual(0, countTrue)
        self.assertEqual(28, countNew)

        self.assertEqual(0, evalInfoNew[0])
        self.assertEqual(28, evalInfoNew[1])
        self.assertEqual(0, evalInfoNew[2])

    def test_custom_distance__synthetic_data_pred_and_labels_are_same(self):
        from main import custom_distance
        import numpy as np

        # ('R1', 'R1', 'C6')
        labels = [[list([('C1', 'R1', 'C2'), ('C3', 'R2', 'C4'), ('C5', 'R3', 'C6')]),
                        list([('C7', 'R1', 'C8'), ('C9', 'R1', 'C10'), ('C11', 'R4', 'C12')]),
                        list([('C13', 'R5', 'C10'), ('C14', 'R5', 'C14')]),
                        list([('C15', 'R5', 'C10'), ('C16', 'R1', 'C2')]),
                        list([('C17', 'R1', 'C2'), ('C15', 'R1', 'C2')]),
                        list([('C18', 'R5', 'C18'), ('C19', 'R5', 'C19')])],
                  [list([('C1', 'R1', 'C2'), ('C3', 'R2', 'C4'), ('C5', 'R3', 'C6')]),
                       list([('C7', 'R1', 'C8'), ('C9', 'R1', 'C10'), ('C11', 'R4', 'C12')]),
                       list([('C13', 'R5', 'C10'), ('C14', 'R5', 'C14')]),
                       list([('C15', 'R5', 'C10'), ('C16', 'R1', 'C2')]),
                       list([('C17', 'R1', 'C2'), ('C15', 'R1', 'C2')]),
                       list([('C18', 'R5', 'C18'), ('C19', 'R5', 'C19')])]
                  ]
        labels = np.array(labels)

        pred = [[list([('C1', 'R1', 'C2'), ('C3', 'R2', 'C4'), ('C5', 'R3', 'C6')]),
                        list([('C7', 'R1', 'C8'), ('C9', 'R1', 'C10'), ('C11', 'R4', 'C12')]),
                        list([('C13', 'R5', 'C10'), ('C14', 'R5', 'C14')]),
                        list([('C15', 'R5', 'C10'), ('C16', 'R1', 'C2')]),
                        list([('C17', 'R1', 'C2'), ('C15', 'R1', 'C2')]),
                        list([('C18', 'R5', 'C18'), ('C19', 'R5', 'C19')])],
                  [list([('C1', 'R1', 'C2'), ('C3', 'R2', 'C4'), ('C5', 'R3', 'C6')]),
                       list([('C7', 'R1', 'C8'), ('C9', 'R1', 'C10'), ('C11', 'R4', 'C12')]),
                       list([('C13', 'R5', 'C10'), ('C14', 'R5', 'C14')]),
                       list([('C15', 'R5', 'C10'), ('C16', 'R1', 'C2')]),
                       list([('C17', 'R1', 'C2'), ('C15', 'R1', 'C2')]),
                       list([('C18', 'R5', 'C18'), ('C19', 'R5', 'C19')])]
                  ]
        pred = np.array(pred)

        shape = (2, 6, 3)

        custTN, custNT, countTrue, countNew, evalInfoNew = custom_distance(shape, pred, labels)
        self.assertEqual(0, custTN)
        self.assertEqual(0, custNT)
        self.assertEqual(28, countTrue)
        self.assertEqual(28, countNew)

        self.assertEqual(28, evalInfoNew[0])
        self.assertEqual(0, evalInfoNew[1])
        self.assertEqual(0, evalInfoNew[2])

    def test_custom_distance__synthetic_data_pred_and_labels_are_similar(self):
        from main import custom_distance
        import numpy as np

        # ('R1', 'R1', 'C6')
        labels = [[list([('C100', 'R100', 'C100'), ('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                        list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                        list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                        list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                        list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                        list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')])],
                  [list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                   list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                   list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                   list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                   list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                   list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')])]
                  ]
        labels = np.array(labels)

        pred = [[list([('C99', 'R99', 'C95'), ('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                        list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                        list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1'), ('C50', 'R30', 'C50')]),
                        list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                        list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                        list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')])],
                [list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                 list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                 list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                 list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                 list([('C1', 'R1', 'C1'), ('C1', 'R1', 'C1')]),
                 list([('C1', 'R1', 'C1'), ('C1', 'R1', 'R1')])]
                  ]
        pred = np.array(pred)

        shape = (2, 6, 3)

        custTN, custNT, countTrue, countNew, evalInfoNew = custom_distance(shape, pred, labels)
        self.assertEqual(7, custTN)
        self.assertEqual(136, custNT)
        self.assertEqual(28, countTrue)
        self.assertEqual(29, countNew)

        self.assertEqual(26, evalInfoNew[0])
        self.assertEqual(3, evalInfoNew[1])
        self.assertEqual(2, evalInfoNew[2])

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

    def test_get_labels_from_encoding__handles_true_and_pred_values_the_same(self):
        from main import get_labels_from_encoding, convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, numConcepts, numRoles = convert_data_to_arrays(get_rdf_data('rdfData/gfo-1.0.json'))

        allTheData = cross_validation_split_all_data(5, KB, supports, outputs)

        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests = allTheData

        trueArr, predArr = get_labels_from_encoding(y_tests[0], y_tests[0], 28, 14)

        self.assertEqual(trueArr.all(), predArr.all())

    def test_get_labels_from_encoding__each_sample_has_correct_size(self):
        from main import get_labels_from_encoding, convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, numConcepts, numRoles = convert_data_to_arrays(get_rdf_data('rdfData/gfo-1.0.json'))

        allTheData = cross_validation_split_all_data(5, KB, supports, outputs)

        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests = allTheData

        for i in range(len(y_tests)):
            trueArr, predArr = get_labels_from_encoding(y_tests[i], y_tests[i], 28, 14)

            self.assertEqual(trueArr.all(), predArr.all())

            s = y_tests[i].shape
            s1 = trueArr.shape
            s2 = predArr.shape
            self.assertEqual((s[0],s[1]), s1, s2)

            for sample in range(len(trueArr)):
                for ts in range(len(trueArr[sample])):
                    self.assertEqual(len(trueArr[sample][ts]), s[2]/3)
                    self.assertEqual(len(predArr[sample][ts]), s[2] / 3)

    def test_get_labels_from_encoding__convert_back_to_sample(self):
        from main import get_labels_from_encoding, convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        x = [
            [
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285],
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285],
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285],
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285],
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285],
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285],
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285]
            ],
            [
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285],
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285],
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285],
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285],
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285],
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285],
                [0.5, -0.1428571428, 0.5, 0.1785714285, -0.928571428, 0.1785714285]
            ]
        ]
        x = np.array(x)

        y = [
            [
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')],
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')],
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')],
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')],
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')],
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')],
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')]
            ],
            [
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')],
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')],
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')],
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')],
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')],
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')],
                [('C14', 'R2', 'C14'), ('C5', 'R13', 'C5')]
            ]
        ]
        z = np.zeros(shape=(2,7), dtype=list)
        z[0] = y[0]
        z[1] = y[1]

        trueArr, predArr = get_labels_from_encoding(x, x, 28, 14)

        self.assertEqual(trueArr.all(), z.all())
        self.assertEqual(predArr.all(), z.all())

    def test_convert_encoded_item_to_label(self):
        from main import convert_encoded_float_to_label
        decodeNonProperty = 14
        decodeProperty = 7
        floats = [-0.142857143, -0.714285714, -1, 0.071428571, 0.428571429, 1, 0.000001, -0.0001, 1.0234, -2.856]
        labels = ['R1', 'R5', 'R7', 'C1', 'C6', 'C14', '0', '0', 'C14', 'R7']

        for i in range(len(floats)):
            self.assertEqual(labels[i], convert_encoded_float_to_label(floats[i], decodeNonProperty, decodeProperty))

    def test_pad_kb__everyone_same_size(self):
        from main import get_rdf_data, pad_kb
        data = get_rdf_data('rdfData/gfo-1.0.json')
        kb = data['kB']
        kb = pad_kb(kb)

        for sample in kb:
            self.assertEqual(len(kb[0]), len(sample))

    def test_pad_kb__removing_padding_reveals_original(self):
        from main import get_rdf_data, pad_kb
        data = get_rdf_data('rdfData/gfo-1.0.json')
        kb = data['kB']
        KB = pad_kb(kb)

        for sample in KB:
            for index in range(len(sample)):
                if sample[index] == 0.0:
                    del sample[index:]
                    break
        self.assertEqual(kb, KB)

    def test_convert_data_to_arrays__shape_preserved(self):
        from main import convert_data_to_arrays, get_rdf_data
        data = get_rdf_data('rdfData/gfo-1.0.json')
        kb, supp, outs, numConcepts, numRoles = data['kB'], data['supports'], data['outputs'], data['concepts'], data[
            'roles']
        KB, supports, outputs, numConcepts, numRoles = convert_data_to_arrays(data)

        self.assertEqual(len(kb), len(KB))
        self.assertEqual(len(supp), len(supports))
        self.assertEqual(len(outs), len(outputs))

        for sample in range(len(kb)):
            self.assertEqual(len(supp[sample]), len(supports[sample]))
            self.assertEqual(len(outs[sample]), len(outputs[sample]))
            self.assertEqual(len(outputs[sample]), len(supports[sample]))

    def test_pad_list_of_lists__same_size(self):
        from main import get_rdf_data as getData
        from main import pad_list_of_lists as testMethod

        data = getData('rdfData/gfo-1.0.json')
        list = testMethod(data['supports'])

        for i in range(len(list)):
            element = list[i]
            for j in range(len(element)-1):
                self.assertEqual(len(element[j]), len(element[j+1]))

    def test_cross_validation_split_all_data__correct_KB_repetition(self):
        from main import convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, num1, num2 = convert_data_to_arrays(get_rdf_data('rdfData/gfo-1.0.json'))

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests = cross_validation_split_all_data(8, KB, supports, outputs)

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

    def test_cross_validation_split_all_data__no_kb_data_lost(self):
        from main import convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, numConcepts, numRoles = convert_data_to_arrays(get_rdf_data('rdfData/gfo-1.0.json'))

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests = cross_validation_split_all_data(5, KB, supports, outputs)

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

    def test_cross_validation_split_all_data__no_support_data_lost(self):
        from main import convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, numConcepts, numRoles = convert_data_to_arrays(get_rdf_data('rdfData/gfo-1.0.json'))

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests = \
            cross_validation_split_all_data(7, KB, supports, outputs)

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

    def test_cross_validation_split_all_data__no_output_data_lost(self):
        from main import convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, num1, num2 = convert_data_to_arrays(get_rdf_data('rdfData/gfo-1.0.json'))

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests = cross_validation_split_all_data(7, KB, supports, outputs)

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

    def test_cross_validation_split_all_data__correct_mapping_fromKBToSuppToOuts_of_training(self):
        from main import convert_data_to_arrays, get_rdf_data, \
            cross_validation_split_all_data

        KB, supports, outputs, num1, num2 = convert_data_to_arrays(get_rdf_data('rdfData/gfo-1.0.json'))

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests = cross_validation_split_all_data(3, KB, supports, outputs)

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

    def test_cross_validation_split_all_data__correct_mapping_fromKBToSuppToOuts_of_test(self):
        from main import convert_data_to_arrays, get_rdf_data, \
            cross_validation_split_all_data

        KB, supports, outputs, num1, num2 = convert_data_to_arrays(get_rdf_data('rdfData/gfo-1.0.json'))

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests = cross_validation_split_all_data(3, KB, supports, outputs)

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

    def test_cross_validation_split_all_data__correct_numOf_folds_returned(self):
        from main import convert_data_to_arrays, get_rdf_data, cross_validation_split_all_data

        KB, supports, outputs, num1, num2 = convert_data_to_arrays(get_rdf_data('rdfData/gfo-1.0.json'))

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests = cross_validation_split_all_data(7, KB, supports, outputs)

        self.assertEqual(7, KBs_tests.shape[0])
        self.assertEqual(7, KBs_trains.shape[0])
        self.assertEqual(7, X_tests.shape[0])
        self.assertEqual(7, X_trains.shape[0])
        self.assertEqual(7, y_tests.shape[0])
        self.assertEqual(7, y_tests.shape[0])