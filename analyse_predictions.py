import numpy as np

def get_predictions(filenames):
    # Read in all true labels and corresponding predictions from filenames (list of files).
    true_labels = []
    predictions = []

    for filename in filenames:
        with open(filename, "r") as file:
            for line in file:
                tl_start = line.find("[") + 1
                tl_end = line.find("]")
                pred_start = line.rfind("[") + 1
                pred_end = line.rfind("]")

                tl = np.array(list(line[tl_start:tl_end].replace(".","").replace(",","").replace(" ","")))
                pred = np.array(list(line[pred_start:pred_end].replace(".","").replace(",","").replace(" ","")))

                true_labels.append(tl)
                predictions.append(pred)

    return true_labels, predictions


def check_counts(true_labels):
    # Check true labels correspond to all training examples 
    n_activities = len(true_labels[0])
    counts = np.zeros((n_activities, n_activities))

    for tl in true_labels:
        activities = np.argwhere(tl=="1")
        if len(activities) == 1:
            activity = activities[0][0]
            counts[activity, activity] += 1
        elif len(activities) == 2:
            activity1 = activities[0][0]
            activity2 = activities[1][0]
            counts[activity1, activity2] += 1
            counts[activity2, activity1] += 1

    return counts


def calc_accuracy(true_labels, predictions):
    # Calculate overall accuracies per activity
    n_activities = len(true_labels[0])
    accuracies = np.zeros(n_activities)

    for activity in range(n_activities):
        for i in range(len(true_labels)):
            if true_labels[i][activity] ==  predictions[i][activity]:
                accuracies[activity] += 1

    accuracies /= len(true_labels)

    return accuracies.round(3)


def analyse_false_negatives(activity, true_labels, predictions):
    # For the specified activity, output the percentage of false negatives per activity
    n_activities = len(true_labels[0])
    fn_percentages = np.zeros(n_activities) # Percentage of false negatives for specified activity which occur with each activity

    for i in range(len(true_labels)):
        tl = true_labels[i]
        pred = predictions[i]

        if tl[activity] == "1" and pred[activity] == "0": # Identify false negatives
            activities = np.argwhere(tl=="1")
            if len(activities) == 1:
                fn_percentages[activity] += 1
            elif len(activities) == 2:
                activity1 = activities[0][0]
                activity2 = activities[1][0]
                if activity == activity1:
                    fn_percentages[activity2] += 1
                else:
                    fn_percentages[activity1] += 1

    fn_total = np.sum(fn_percentages)
    fn_percentages /= fn_total
    
    return fn_total, fn_percentages.round(3)


def analyse_false_positives(activity, true_labels, predictions):
    # For the specified activity, output the percentage of false positives per activity
    n_activities = len(true_labels[0])
    fp_percentages = np.zeros(n_activities)

    for i in range(len(true_labels)):
        tl = true_labels[i]
        pred = predictions[i]

        if tl[activity] == "0" and pred[activity] == "1": # Identify false positives
            activities = np.argwhere(tl=="1")
            if len(activities) == 1:
                activity1 = activities[0][0]
                fp_percentages[activity1] += 1
            elif len(activities) == 2:
                activity1 = activities[0][0]
                activity2 = activities[1][0]
                fp_percentages[activity1] += 0.5
                fp_percentages[activity2] += 0.5

    fp_total = np.sum(fp_percentages)
    fp_percentages /= fp_total
    
    return fp_total, fp_percentages.round(3)


if __name__ == "__main__":
    filenames = [
        "./LSTM_final/LSTM_predictions_fold0.txt",
        "./LSTM_final/LSTM_predictions_fold1.txt",
        "./LSTM_final/LSTM_predictions_fold2.txt",
        "./LSTM_final/LSTM_predictions_fold3.txt",
        "./LSTM_final/LSTM_predictions_fold4.txt",
    ]

    true_labels, predictions = get_predictions(filenames)

    print("Check activity counts (should be 240 for all entries apart from run-stand ((2,4) and (4,2)), which should be 239):")
    counts = check_counts(true_labels)
    print(counts) 
    print()

    print("Average accuracies:")
    accuracies = calc_accuracy(true_labels, predictions)
    print(f"Per activity = {accuracies}, overall = {np.sum(accuracies)/len(accuracies)}")
    print()

    print("Analysis of false negatives:")
    for activity in range(5):
        fn_total, fn_percentages = analyse_false_negatives(activity, true_labels, predictions)
        print(f"Activity {activity}: {int(fn_total)} false negatives, percentages = {fn_percentages}")
    print()

    print("Analysis of false positives:")
    for activity in range(5):
        fp_total, fp_percentages = analyse_false_positives(activity, true_labels, predictions)
        print(f"Activity {activity}: {int(fp_total)} false positives, percentages = {fp_percentages}")