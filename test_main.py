from unittest import TestCase


class Test(TestCase):
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


    def test_pad_list_of_lists_same_size(self):
        from main import get_rdf_data as getData
        from main import pad_list_of_lists as testMethod

        data = getData('rdfData/test1.1.json')
        list = testMethod(data['supports'])

        for i in range(len(list)):
            element = list[i]
            for j in range(len(element)-1):
                self.assertEqual(len(element[j]), len(element[j+1]))
