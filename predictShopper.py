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
    # load shopping data
    with open("shopping.csv", "r") as input:
        labels = []
        evidence = []
        input_reader = csv.DictReader(input)
        page_analytics = input_reader.fieldnames
        months = { 'Jan': 0,
                   'Feb': 1,
                   'Mar': 2,
                   'Apr': 3,
                   'May': 4,
                   'June': 5,
                   'Jul': 6,
                   'Aug': 7,
                   'Sep': 8,
                   'Oct': 9,
                   'Nov': 10,
                   'Dec': 11
        }

        for row in input_reader:
            newEvidence = []
            newEvidence.append(int(row[page_analytics[0]]))
            newEvidence.append(float(row[page_analytics[1]]))
            newEvidence.append(int(row[page_analytics[2]]))
            newEvidence.append(float(row[page_analytics[3]]))
            newEvidence.append(int(row[page_analytics[4]]))
            newEvidence.append(float(row[page_analytics[5]]))
            newEvidence.append(float(row[page_analytics[6]]))
            newEvidence.append(float(row[page_analytics[7]]))
            newEvidence.append(float(row[page_analytics[8]]))
            newEvidence.append(float(row[page_analytics[9]]))
            newEvidence.append(months[row[page_analytics[10]]])
            newEvidence.append(int(row[page_analytics[11]]))
            newEvidence.append(int(row[page_analytics[12]]))
            newEvidence.append(int(row[page_analytics[13]]))
            newEvidence.append(int(row[page_analytics[14]]))
            newEvidence.append(1 if row[page_analytics[15]] == "Returning_Visitor" else 0)
            newEvidence.append(1 if row[page_analytics[16]] == "TRUE" else 0)
            evidence.append(newEvidence)
            labels.append(1 if row[page_analytics[17]] == "TRUE" else 0)


    return (evidence,labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    knn_model = KNeighborsClassifier(n_neighbors=1)
    knn_model.fit(evidence,labels)

    return knn_model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    neg = 0
    pos = 0

    for label, predict in zip(labels,predictions):
        # sensitivity represents true postive rate 
        # if prediction and value are both true
        if predict == 1 and label == 1:
            pos = pos + 1
        # specificity - true negative rate
        # if prediction and value are both false
        elif predict == 0 and label == 0:
            neg = neg + 1

    # return the postive predictions by the number of postives 
    # return the negatvie predictions by the negatives 
    return(pos / labels.count(1), neg / labels.count(0))


if __name__ == "__main__":
    main()