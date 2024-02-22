import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
import seaborn as sns
import glob
from sklearn.linear_model import LogisticRegression


def motif_fold(folder):
    all_seqs, seqs, num_seqs = read_lib_from_file(folder)
    PCA_y = PCA_reduction(all_seqs)

    while True:
        iterations_input = input("Choose the number of iterations before proceeding: ")
                
        try:
            iterations = int(iterations_input)
            break
        except TypeError:
            print("Invalid type. Please try again.")

    while True:
        motif_input = input("Choose the motif length before proceeding: ")
             
        try:
            motif_length = int(motif_input)
            break
        except TypeError:
            print("Invalid type. Please try again.")

    motifs = motif_generation(motif_length)

    motifs_of_seqs = motifs_from_seqs(num_seqs, motif_length, seqs, motifs)

    CV_predictions, CV_std_predictions = train_test_cv(num_seqs, motifs_of_seqs, motif_length, all_seqs, iterations)

    export_predictions(all_seqs, PCA_y, CV_predictions, CV_std_predictions, folder)


    while True:
        plotting = input("Would you like a box plot of predictions by cross-validation? Say 'yes' or 'no': ")

        if plotting == "yes":
            RMSE_box_plot(CV_predictions, PCA_y, all_seqs, folder)
            break
        elif plotting == "no":
            break
        else:
            print("Unexpected input. Please try again.")

    while True:
        unknown_preds = input("Is there a file with unknown sequences to predict in this directory? Say 'yes' or 'no': ")

        if unknown_preds == "yes":
            predictions = unknown_predictions(folder, motifs_of_seqs, motif_length, motifs, PCA_y, all_seqs)
            export_unknowns(predictions, folder)
            break
        elif unknown_preds == "no":
            break
        else:
            print("Unexpected input. Please try again.")




def read_lib_from_file(input_folder):
    print("Fetching sequence data from folder...")

    for file in glob.iglob(input_folder + "/*_library.xlsx"):
        all_seqs = pd.read_excel(file)

    seqs = pd.DataFrame(all_seqs.iloc[:,7:27])

    seqs.replace(to_replace="G", value=0, inplace=True)
    seqs.replace(to_replace="B", value=1, inplace=True)
    seqs = seqs.to_numpy()

    num_seqs = len(seqs[:,0])

    return all_seqs, seqs, num_seqs


def read_unknown_from_file(folder):
    print("Fetching data for unknown sequences...")

    for file in glob.iglob(folder + "/*_unknown.xlsx"):
        all_seqs = pd.read_excel(file)

    seqs = pd.DataFrame(all_seqs.iloc[:,3:23])

    seqs.replace(to_replace="G", value=0, inplace=True)
    seqs.replace(to_replace="B", value=1, inplace=True)
    seqs = seqs.to_numpy()

    num_seqs = len(seqs[:,0])

    return all_seqs, seqs, num_seqs


def PCA_reduction(all_seqs):
    print("Performing PCA on RGB colors in library...")

    pca = PCA(n_components = 1)
    
    pca.fit(scale(all_seqs[["Red","Green","Blue"]]))
    print("Variance is ", pca.explained_variance_ratio_)
    y_fit = pca.transform(scale(all_seqs[["Red","Green","Blue"]]))

    return np.ravel(y_fit)



def motif_generation(motif_length):
    print("Generating motifs for selected motif length...")

    motifs = np.zeros((motif_length, 2**motif_length))

    for i in range(motif_length):
        for j in range(2**motif_length):
            motifs[i,j] = ((j+1) // (2**i)) % 2

    motifs = motifs.transpose() # each motif is a row in this matrix, the index of each motif is the row index

    return motifs


def motifs_from_seqs(num_seqs, motif_length, seqs, motifs):
    print("Converting sequences from library into motif features...")

    motifs_of_seqs = pd.DataFrame(np.zeros((num_seqs, 20 - motif_length + 1)))

    column_names = []
    column_names_string = ""

    for i in range(20 - motif_length + 1):
        column_names.append("motif_" + str(i+1))
        column_names_string += "motif_" + str(i+1) + " + " 

    motifs_of_seqs.columns = column_names

    for i in range(num_seqs):
        for j in range(20 - motif_length + 1):
            test_motif = seqs[i,j:(j+motif_length)]

            for k in range(len(motifs[:,0])):
                if (test_motif == motifs[k,:]).all():
                    motifs_of_seqs.iloc[i,j] = str(k) #alphabet[k]
                    break

    return motifs_of_seqs



def train_test_cv(num_seqs, motifs_of_seqs, motif_length, all_seqs, iterations):
    print("Beginning cross validation...")

    motifs_with_RGB = motifs_of_seqs.copy()
    motifs_with_RGB["target"] = all_seqs["Colorz"]
    motifs_with_RGB["Hydrophobicity"] = all_seqs["Hydrophobicity"]

    skf = StratifiedKFold(n_splits = 5, shuffle = True)

    logloss = []
    predictions = np.zeros((num_seqs, iterations))

    for fit in range(iterations):
        if (fit % 10) == 0:
            print("Iteration number ", fit)
        
        ll_predictions = np.zeros((num_seqs, 8))

        for train_index, test_index in skf.split(motifs_with_RGB, all_seqs["Colorz"]):
            train, test = motifs_with_RGB.iloc[train_index,:], motifs_with_RGB.iloc[test_index,:]
            columns = list(range(0,20 - motif_length + 1))

            X = train.iloc[:,columns]
            y = train["target"]
            encoder = ce.JamesSteinEncoder().fit(X, y)
            train_enc = encoder.transform(X)
            train_enc["Hydrophobicity"] = train["Hydrophobicity"]
            # train_enc = train["Hydrophobicity"].to_numpy().reshape(-1,1)

            clf = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=10000, class_weight='balanced')
            # clf = GradientBoostingClassifier(learning_rate=0.1, max_depth=7, n_estimators = 100, subsample = 0.7)
            clf.fit(train_enc, y)

            # print(clf.predict(train_enc))
            # print(y)

            test_enc = encoder.transform(test.iloc[:,columns])
            test_enc["Hydrophobicity"] = test["Hydrophobicity"]

            # test_enc = test["Hydrophobicity"].to_numpy().reshape(-1,1)

            pred_prob = clf.predict_proba(test_enc)
            ll_predictions[test_index, :] = pred_prob
            # print(pred_prob)

            predictions[test_index, fit] = clf.predict(test_enc)
            # print(clf.predict(test_enc))
        
        logloss.append(log_loss(all_seqs["Colorz"], ll_predictions))
    
    avg_predictions = np.mean(predictions, axis=1)
    SD_predictions = np.std(predictions, axis=1)

    print("averaged log loss is ", sum(logloss) / len(logloss))

    return avg_predictions, SD_predictions



def export_predictions(all_seqs, PCA_y, CV_predictions, CV_SD_predictions, folder):
    print("Exporting out-of-sample predictions to excel folder...")
    CV_df = pd.DataFrame()
    CV_df["Color"] = all_seqs["Color"]
    CV_df["PCA of color group"] = PCA_y
    CV_df["Predictions"] = CV_predictions
    CV_df["Standard deviation"] = CV_SD_predictions
    CV_df.to_excel(folder + "/prediction_statistics.xlsx")



def RMSE_box_plot(predictions, PCA_y, all_seqs, folder):
    print("Plotting data as box plot...")

    pred_df = pd.DataFrame()
    pred_df["predictions"] = predictions
    pred_df["ground truth"] = PCA_y
    pred_df["color group"] = all_seqs["Colorz"]

    sns.boxplot(data=pred_df, x="color group", y="predictions",boxprops={'alpha': 0.3},linewidth=0.5,fliersize=4)
    sns.stripplot(data=pred_df, x="color group", y="predictions", size=7)
    plt.tight_layout()
    plt.savefig(folder + "/scatter_boxplot.pdf",dpi=300)




def unknown_predictions(folder, motifs_of_seqs, motif_length, motifs, PCA_y, all_seqs):
    print("Fitting unknown sequences to model...")

    full_encoder = ce.JamesSteinEncoder().fit(motifs_of_seqs, PCA_y)
    train_encoded = full_encoder.transform(motifs_of_seqs, PCA_y)
    train_encoded["Hydrophobicity"] = all_seqs["Hydrophobicity"]

    clf = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, n_estimators=100, subsample= 0.7)
    clf.fit(train_encoded, PCA_y)

    # read in your sequences
    unknown_seq_df, unknown_seqs, num_unknown = read_unknown_from_file(folder)

    # motifs of unknowns
    motifs_unknowns = motifs_from_seqs(num_unknown, motif_length, unknown_seqs, motifs)
    encoded_unknowns = full_encoder.transform(motifs_unknowns)
    encoded_unknowns["Hydrophobicity"] = unknown_seq_df["Hydrophobicity"]

    # these numbers will be the PCA values. Use the table exported in line 76 to back out which PCA number corresponds to which color group
    predictions = clf.predict(encoded_unknowns) 

    return predictions




def export_unknowns(predictions, folder):
    print("Exporting unknowns to folder...")

    preds_df = pd.DataFrame(predictions)
    preds_df.to_excel(folder + "/PCA_of_unknowns.xlsx")

motif_fold('/Users/suprajachittari/Documents/GitHub/MotifFold/motif_fold/sample_data')