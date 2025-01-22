from modules import GLV, NetworkImpact, NeuralNetwork, DistanceCheck, RandomForest, IDOA, ROC
import numpy as np
import sys


class Testing:
    def __int__(self):
        pass

    @staticmethod
    def run_test_semi_supervised_drop_one_out(healthy_cohort, other_cohort):
        IDOA.IDOA.real = True
        GLV.GLV.numOfPopulations = len(healthy_cohort[0])
        all_predictions = {}
        auc_values = {}
        method_labels = Testing.get_methods_names_semi_supervised()
        for method_label in method_labels:
            all_predictions[method_label] = []
            auc_values[method_label] = []
        total_real = []

        cohorts_data = [healthy_cohort, other_cohort]
        for i in range(len(cohorts_data)):
            for j in range(len(cohorts_data[i])):
                cohorts_list = [cohort for cohort in cohorts_data]
                cohorts_list[i] = np.delete(cohorts_list[i], j, axis=0)
                print(
                    f'first cohort: {len(cohorts_list[0])}, second cohort: {len(cohorts_list[1])}, total first: {len(cohorts_data[0])}, total second: {len(cohorts_data[1])}')
                models = [{'cohort': np.array(cohort)} for cohort in cohorts_list]
                data = {
                    'models': models
                }
                samples = [cohorts_data[i][j]]
                total_real.append(i)

                idoa = IDOA.IDOA(data)
                network_impact = NetworkImpact.NetworkImpact(data)
                distance_check = DistanceCheck.DistanceCheck(data)
                distance_check2 = DistanceCheck.DistanceCheck(data, 1)

                # predictions
                network_impact_predictions = network_impact.predict_real(data['models'][0]['cohort'], samples)
                network_impact1 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 0)
                network_impact2 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 1)
                network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)
                network_impact4 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 3)
                network_impact5 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 4)

                methods = [idoa, distance_check, distance_check2, network_impact1, network_impact2, network_impact3,
                           network_impact4, network_impact5]

                some_results = {}
                for method in methods:
                    predictions = method.predict_real(data['models'][0]['cohort'], np.array(samples))
                    all_predictions[method_labels[methods.index(method)]].append(predictions[0])
                    some_results[method_labels[methods.index(method)]] = predictions[0]
                print(f'some results: {some_results}')


        roc = ROC.ROC(True)
        for method_label in method_labels:
            auc = roc.add_graph(total_real, all_predictions[method_label], str(method))
            auc_values[method_label].append(auc)

        return [all_predictions, auc_values]

    @staticmethod
    def run_test_supervised_drop_one_out(cohorts_data):
        all_results = {}
        method_labels = Testing.get_full_methods_names()
        IDOA.IDOA.real = True
        GLV.GLV.numOfPopulations = len(cohorts_data[0][0])
        for method_label in method_labels:
            all_results[method_label] = []

        for i in range(len(cohorts_data)):
            for j in range(len(cohorts_data[i])):
                # for each sample run tests
                cohorts_list = [cohort for cohort in cohorts_data]
                cohorts_list[i] = np.delete(cohorts_list[i], j, axis=0)
                print(
                    f'first cohort: {len(cohorts_list[0])}, second cohort: {len(cohorts_list[1])}, total first: {len(cohorts_data[0])}, total second: {len(cohorts_data[1])}')
                models = [{'cohort': np.array(cohort)} for cohort in cohorts_list]
                data = {
                    'models': models
                }
                samples = [cohorts_data[i][j]]
                real = i

                network = NeuralNetwork.NeuralNetwork(data)
                idoa = IDOA.IDOA(data)
                network_impact = NetworkImpact.NetworkImpact(data)
                distance_check = DistanceCheck.DistanceCheck(data)
                distance_check2 = DistanceCheck.DistanceCheck(data, 1)
                random_forest = RandomForest.RandomForest(data)

                # predictions
                network_impact_predictions = network_impact.predict(samples)

                network_impact1 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 0)
                network_impact2 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 1)
                network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)
                network_impact4 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 3)
                network_impact5 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 4)

                methods = [idoa, network, distance_check, distance_check2, network_impact1, network_impact2,
                           network_impact3,
                           network_impact4, network_impact5, random_forest]

                some_results = {}
                method_labels = Testing.get_full_methods_names()
                for method_label in method_labels:
                    some_results[method_label] = []

                for method in methods:
                    predictions = method.predict(np.array(samples))
                    all_results[str(method)].append(1 if real == predictions[0] else 0)
                    some_results[str(method)].append(1 if real == predictions[0] else 0)
                print(f'finished sample {j} in cohort {i} from {len(cohorts_data)} and {len(cohorts_data[i])}, real={real}')
                print(f'results are {some_results}')
        return all_results

    @staticmethod
    def run_test_semi_supervised_real_data(healthy_cohort, other_cohort, number_of_runs):
        IDOA.IDOA.real = True
        GLV.GLV.numOfPopulations = len(healthy_cohort[0])
        all_results = {}
        method_labels = Testing.get_methods_names_semi_supervised()
        for method_label in method_labels:
            all_results[method_label] = []

        for nnnn in range(number_of_runs):
            print(f'run number {nnnn+1} out of {number_of_runs}')
            samples = []
            real = []
            np.random.shuffle(healthy_cohort)

            # Split the array into 80% and 20%
            split_index = int(len(healthy_cohort) * 0.8)  # Calculate the index for the 80% split
            array_80, array_20 = healthy_cohort[:split_index], healthy_cohort[split_index:]

            cohort = array_80
            samples.extend(array_20)

            np.random.shuffle(other_cohort)

            samples.extend(other_cohort[:len(array_20)])
            real.extend([0] * len(array_20))
            real.extend([1] * len(other_cohort[:len(array_20)]))

            data = {
                'models':
                    [
                        {'cohort': cohort}
                    ]
            }


            idoa = IDOA.IDOA(data)
            network_impact = NetworkImpact.NetworkImpact(data)
            distance_check = DistanceCheck.DistanceCheck(data)
            distance_check2 = DistanceCheck.DistanceCheck(data, 1)

            # predictions
            network_impact_predictions = network_impact.predict_real(data['models'][0]['cohort'], samples)
            network_impact1 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 0)
            network_impact2 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 1)
            network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)
            network_impact4 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 3)
            network_impact5 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 4)

            methods = [idoa, distance_check, distance_check2, network_impact1, network_impact2, network_impact3,
                       network_impact4, network_impact5]

            roc = ROC.ROC(False)

            auc_r = []

            for method in methods:
                predictions = method.predict_real(data['models'][0]['cohort'], np.array(samples))
                auc = roc.add_graph(real, predictions, str(method))
                all_results[method_labels[methods.index(method)]].append(auc)
                auc_r.append(auc)
        return all_results

    @staticmethod
    def run_gradual_change_test(file_path, m, num_of_samples, delta_values, bound=0.025, probability=0.1):
        A = np.array(GLV.GLV.get_random_A())
        B = np.array(GLV.GLV.get_random_A())
        r = GLV.GLV.get_random_r()
        cohorts = 2
        all_results = []

        for delta in delta_values:
            print(f'delta: {delta}')
            models = []
            model = GLV.GLV(r=r, A=A)
            samples = [sample.tolist() for sample in model.get_samples(m)]
            models.append({
                "A": model.get_A(),
                "cohort": samples,
                "index": 0,
                "r": r.tolist()
            })

            model2 = GLV.GLV(r=r, A=(A * (1-delta) + B * delta))
            samples = [sample.tolist() for sample in model2.get_samples(m)]
            models.append({
                "A": A * (1-delta) + B * delta,
                "cohort": samples,
                "index": 1,
                "r": r.tolist()
            })

            data = {'models': models}
            cohorts_list = [data['models'][i]['cohort'] for i in range(cohorts)]
            samples, real = GLV.generate_random_samples(data, num_of_samples)

            network = NeuralNetwork.NeuralNetwork(data)
            idoa = IDOA.IDOA(data)
            network_impact = NetworkImpact.NetworkImpact(data)
            distance_check = DistanceCheck.DistanceCheck(data)
            distance_check2 = DistanceCheck.DistanceCheck(data, 1)
            random_forest = RandomForest.RandomForest(data)

            # predictions
            network_impact_predictions = network_impact.predict(samples)

            network_impact1 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 0)
            network_impact2 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 1)
            network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)
            network_impact4 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 3)
            network_impact5 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 4)

            methods = [idoa, network, distance_check, distance_check2, network_impact1, network_impact2,
                       network_impact3,
                       network_impact4, network_impact5, random_forest]

            states = []

            in_group_distances = []
            between_groups_distances = []
            for i in range(cohorts):
                states.extend(cohorts_list[i])
                distance = DistanceCheck.DistanceCheck.calculate_in_group_distance(cohorts_list[i])
                in_group_distances.append(distance)
            for i in range(cohorts):
                for j in range(i + 1, cohorts):
                    distance = DistanceCheck.DistanceCheck.calculate_between_group_distance(cohorts_list[i],
                                                                                            cohorts_list[i])
                    between_groups_distances.append(distance)

            test_results = {
                'm': m,
                'cohorts': 2,
                'tests': len(real),
                'nn_accuracy': network.get_accuracy(),
                'distance': {
                    'in_group': np.array(in_group_distances).mean(),
                    'between_groups': np.array(between_groups_distances).mean()
                },
                'delta': delta,
                'results': {

                }
            }

            for method in methods:
                num_of_success = 0
                predictions = method.predict(np.array(samples))
                results = []
                for prediction, re in zip(predictions, real):
                    if re == prediction:
                        num_of_success += 1
                    results.append((prediction, re, re == prediction))
                success_rate = (num_of_success / len(real)) * 100
                test_results['results'][str(method)] = success_rate
            print(test_results)
            all_results += [test_results]
        return all_results

    @staticmethod
    def run_test(file_path, cohorts, m, num_of_samples, data=None, bound=0.025, probability=0.1, delta=None):
        # create samples if not given
        if data is None:
            data = GLV.generate_models(m, cohorts, file_path, bound=bound, probability=probability, force=True)
            cohorts_list = [data['models'][i]['cohort'] for i in range(cohorts)]
            samples, real = GLV.generate_random_samples(data, num_of_samples)
        else:
            samples = []
            real = []
            cohorts_list = []
            IDOA.IDOA.real = True
            GLV.GLV.numOfPopulations = len(data['models'][0]['cohort'][0])
            for i in range(cohorts):
                cohort = data['models'][i]['cohort']
                np.random.shuffle(cohort)

                # Split the array into 80% and 20%
                split_index = int(len(cohort) * 0.8)  # Calculate the index for the 80% split
                array_80, array_20 = cohort[:split_index], cohort[split_index:]

                cohorts_list.append(array_80)
                samples.extend(array_20)
                real.extend([i] * len(array_20))
            samples = np.array(samples)
            real = np.array(real)
            models = [{'cohort': np.array(cohorts_list[i])} for i in range(cohorts)]
            data = {
                'models': models
            }

        if delta is not None:
            GLV.GLV.delta = delta

        network = NeuralNetwork.NeuralNetwork(data)
        idoa = IDOA.IDOA(data)
        network_impact = NetworkImpact.NetworkImpact(data)
        distance_check = DistanceCheck.DistanceCheck(data)
        distance_check2 = DistanceCheck.DistanceCheck(data, 1)
        random_forest = RandomForest.RandomForest(data)


        # predictions
        network_impact_predictions = network_impact.predict(samples)

        network_impact1 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 0)
        network_impact2 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 1)
        network_impact3 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 2)
        network_impact4 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 3)
        network_impact5 = NetworkImpact.NetworkImpactHandler(network_impact_predictions, 4)

        methods = [idoa, network, distance_check, distance_check2, network_impact1, network_impact2,
                   network_impact3,
                   network_impact4, network_impact5, random_forest]


        states = []

        in_group_distances = []
        between_groups_distances = []
        for i in range(cohorts):
            states.extend(cohorts_list[i])
            distance = DistanceCheck.DistanceCheck.calculate_in_group_distance(cohorts_list[i])
            in_group_distances.append(distance)
        for i in range(cohorts):
            for j in range(i + 1, cohorts):
                distance = DistanceCheck.DistanceCheck.calculate_between_group_distance(cohorts_list[i],
                                                                                        cohorts_list[i])
                between_groups_distances.append(distance)

        test_results = {
            'm': m,
            'cohorts': cohorts,
            'tests': len(real),
            'nn_accuracy': network.get_accuracy(),
            'distance': {
                'in_group': np.array(in_group_distances).mean(),
                'between_groups': np.array(between_groups_distances).mean()
            },
            'results': {

            }
        }

        if delta is not None:
            test_results['delta'] = delta

        for method in methods:
            num_of_success = 0
            predictions = method.predict(np.array(samples))
            results = []
            for prediction, re in zip(predictions, real):
                if re == prediction:
                    num_of_success += 1
                results.append((prediction, re, re == prediction))
            success_rate = (num_of_success / len(real)) * 100
            test_results['results'][str(method)] = success_rate
        print(test_results)
        return test_results

    @staticmethod
    def get_methods_names():
        return ['IDOA', 'NN', 'DIS - BC', 'DIS - EUC', 'NI - SD', 'NI - WD1', 'NI - WD2', 'NI - T1', 'NI - T2', 'RF']

    @staticmethod
    def get_full_methods_names():
        return ['IDOA', 'Neural Network', 'Bray-Curtis Dissimilarity', 'Euclidean Dissimilarity',
                'Network Impact - structural difference', 'Network Impact - weight difference',
                'Network Impact - origin weight difference', 'Network Impact - theta',
                'Network Impact - origin theta', 'Random Forest']

    @staticmethod
    def get_methods_names_semi_supervised():
        return ['IDOA', 'DIS - BC', 'DIS - EUC', 'NI - SD', 'NI - WD1', 'NI - WD2', 'NI - T1', 'NI - T2']