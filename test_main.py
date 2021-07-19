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