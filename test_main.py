from unittest import TestCase


class DistanceEvalTests(TestCase):

    def test_levenshtein_strings_are_equal(self):
        from main import levenshtein as lev
        self.assertEqual(0, lev("levenshtein", "levenshtein"))

    def test_levenshtein_incomprehensibilities_to_inconceivable(self):
        from main import levenshtein as lev
        self.assertEqual(13, lev("incomprehensibilities", "inconceivable"))

    def test_levenshtein_cat_to_dog(self):
        from main import levenshtein as lev
        self.assertEqual(3, lev("cat", "dog"))

    def test_levenshtein_hypothetically_to_(self):
        from main import levenshtein as lev
        self.assertEqual(14, lev("hypothetically", " "))

    def test_custom(self):
        from main import custom as cust
        guess = [("C1", "R1", "C1"), ("C1", "R2", "C1"), ("C200", "R19", "C99"), ("C100", "R1", "R16")]
        true = [("C2", "R1", "C1"), ("C1", "C15", "C1"), ("C100", "R1", "C9"), ("R20", "R1", "C84")]
        distExpect = [1, 17, 208, 220]
        for i in range(len(guess)):
            self.assertEqual(distExpect[i], cust(None, None, guess[i], true[i]))


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


class DataWranglingTests(TestCase):

    def test_pad_list_of_lists_same_size(self):
        from main import get_rdf_data as getData
        from main import pad_list_of_lists as testMethod

        data = getData('rdfData/gfo.json')
        list = testMethod(data['supports'])

        for i in range(len(list)):
            element = list[i]
            for j in range(len(element)-1):
                self.assertEqual(len(element[j]), len(element[j+1]))

