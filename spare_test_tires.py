# def test_levenshtein_strings_are_equal(self):
#     from main import levenshtein as lev
#     self.assertEqual(0, lev("levenshtein", "levenshtein"))
#
#
# def test_levenshtein_incomprehensibilities_to_inconceivable(self):
#     from main import levenshtein as lev
#     self.assertEqual(13, lev("incomprehensibilities", "inconceivable"))
#
#
# def test_levenshtein_cat_to_dog(self):
#     from main import levenshtein as lev
#     self.assertEqual(3, lev("cat", "dog"))
#
#
# def test_levenshtein_hypothetically_to_(self):
#     from main import levenshtein as lev
#     self.assertEqual(14, lev("hypothetically", " "))
#
#
# def test_get_concept_and_role_count(self):
#     from main import get_concept_and_role_count
#     labels1 = {'1': 'label1', '2': 'label2', '3': 'label3', '4': 'label4'}
#     labels2 = {'-1': 'label1', '-2': 'label2', '-3': 'label3', '-4': 'label4'}
#     labels3 = {'-1': 'label1', '2': 'label2', '3': 'label3'}
#
#     con, roles = get_concept_and_role_count(labels1)
#     self.assertEqual(con, 4)
#     self.assertEqual(roles, 0)
#
#     con, roles = get_concept_and_role_count(labels2)
#     self.assertEqual(con, 0)
#     self.assertEqual(roles, 4)
#
#     con, roles = get_concept_and_role_count(labels3)
#     self.assertEqual(con, 2)
#     self.assertEqual(roles, 1)
#
#
# def test_random_label_creator(self):
#     from main import create_random_label_predictions as randLabel
#     shape = (3, 2, 10)
#     numConcepts = 20
#     numRoles = 5
#     rando = randLabel(shape, numConcepts, numRoles)
#     for i in range(rando.shape[0]):
#         for j in range(rando.shape[1]):
#             for tup in range(len(rando[i][j])):
#                 for label in rando[i][j][tup]:
#                     self.assertTrue(label[0] == "C" or label[0] == "R")
#                     if label[0] == "C":
#                         self.assertTrue(numConcepts >= int(label[1:]) > 0)
#                     elif label[0] == "R":
#                         self.assertTrue(numRoles >= int(label[1:]) >= 0)
#
#
# def test_remove_padding(self):
#     from main import remove_padding, get_label_and_iri_from_encoding
#     from main import convert_data_to_arrays, get_rdf_data, get_concept_and_role_count, \
#         cross_validation_split_all_data
#
#     KB, supports, outputs, encodingMap, labels = convert_data_to_arrays(get_rdf_data("rdfData/gfo.json"))
#
#     numConcepts, numRoles = get_concept_and_role_count(labels)
#
#     # Processes data.
#     KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, trueLabels, trueIRIs, labelss = \
#         cross_validation_split_all_data(8, KB, supports, outputs, encodingMap, labels, numConcepts, numRoles)
#
#     data = get_label_and_iri_from_encoding(y_tests[0], labels, numConcepts, numRoles)
#     data = remove_padding(data[0])
#
#     for sampleNum in range(len(data)):
#         for timestepNum in range(len(data[sampleNum])):
#             for tripleNum in range(len(data[sampleNum][timestepNum])):
#                 self.assertTrue(data[sampleNum][timestepNum][tripleNum] != ('0', '0', '0'))
#
#     data = get_label_and_iri_from_encoding(X_tests[0], labels, numConcepts, numRoles)
#     data = remove_padding(data[0])
#
#     for sampleNum in range(len(data)):
#         for timestepNum in range(len(data[sampleNum])):
#             for tripleNum in range(len(data[sampleNum][timestepNum])):
#                 self.assertTrue(data[sampleNum][timestepNum][tripleNum] != ('0', '0', '0'))
