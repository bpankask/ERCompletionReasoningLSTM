from unittest import TestCase


class Test(TestCase):
    def test_cross_validation_split_all_data_no_kb_data_lost(self):
        from main import convert_data_to_arrays, get_rdf_data, get_concept_and_role_count, \
            cross_validation_split_all_data

        KB, supports, outputs, encodingMap, labels = convert_data_to_arrays(get_rdf_data("rdfData/gfo.json"))

        numConcepts, numRoles = get_concept_and_role_count(labels)

        # Processes data.
        KBs_tests, KBs_trains, X_trains, X_tests, y_trains, y_tests, trueLabels, trueIRIs, labelss = \
            cross_validation_split_all_data(5, KB, supports, outputs, encodingMap, labels, numConcepts, numRoles)

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
