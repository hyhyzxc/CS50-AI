import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    with open(filename, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            current = []
            current.append(int(row["Administrative"]))
            current.append(float(row["Administrative_Duration"]))
            current.append(int(row["Informational"]))
            current.append(float(row["Informational_Duration"]))
            current.append(int(row["ProductRelated"]))
            current.append(float(row["ProductRelated_Duration"]))
            current.append(float(row["BounceRates"]))
            current.append(float(row["ExitRates"]))
            current.append(float(row["PageValues"]))
            current.append(float(row["SpecialDay"]))
            month = {"Jan": 0, "Feb": 1, "Mar" : 2, "Apr" : 3, "May" : 4, "June" : 5, "Jul" : 6, "Aug" : 7, "Sep" : 8, "Oct" : 9, "Nov" : 10, "Dec" : 11}
            m = month[row["Month"]]
            current.append(m)
            current.append(int(row["OperatingSystems"]))
            current.append(int(row["Browser"]))
            current.append(int(row["Region"]))
            current.append(int(row["TrafficType"]))
            if row["VisitorType"] == "Returning_Visitor":
                current.append(1)
            else:
                current.append(0)
            if row["Weekend"] == "TRUE":
                current.append(1)
            else:
                current.append(0)
            evidence.append(current)
            if row["Revenue"] == "TRUE":
                labels.append(1)
            else:
                labels.append(0)
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0
    specificity = 0
    total_true = 0
    total_false = 0
    for answer, prediction in zip(labels, predictions):
        if answer == 1:
            total_true += 1
            if prediction == 1:
                sensitivity += 1
        else:
            total_false += 1
            if prediction == 0:
                specificity += 1
    sensitivity = sensitivity / total_true
    specificity = specificity / total_false
    return (sensitivity, specificity)



if __name__ == "__main__":
    main()
