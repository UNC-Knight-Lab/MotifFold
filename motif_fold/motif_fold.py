from hmac import trans_36
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import scale, OneHotEncoder, StandardScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, log_loss
import seaborn as sns
import glob
import shap


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
    
    while True:
        embedding_input = input("This platform supports three types of embedding of motifs: (1) frequency, (2) one-hot, and (3) James-Stein. Choose one by typing '1', '2', or '3'.")

        try:
            embedding_choice = int(embedding_input)
            break
        except TypeError:
            print("Invalid selection. Please try again.")

    while True:
        shap_input = input("Do you want an associated SHAP plot generated over cross-validation? Type 'yes' or 'no'.")

        if shap_input == "yes":
            shap_input = True
            break
        elif shap_input == "no":
            shap_input = False
            break
        else:
            print("Invalid selection. Please try again.")
    
    if embedding_choice == 1: # frequency embedding
        motif_freq = motif_freq_embedding(num_seqs, motifs, motifs_of_seqs)

        if shap_input == True:
            CV_predictions, CV_std_predictions = freq_embedding_CV(num_seqs, motif_freq, motif_length, all_seqs, PCA_y, 5, True)

        CV_predictions, CV_std_predictions = freq_embedding_CV(num_seqs, motif_freq, motif_length, all_seqs, PCA_y, iterations, False)

    elif embedding_choice == 2: # James-Stein embedding
        if shap_input == True:
            CV_predictions, CV_std_predictions = JS_train_test_cv(num_seqs, motifs_of_seqs, motif_length, all_seqs, PCA_y, 5, True)

        CV_predictions, CV_std_predictions = JS_train_test_cv(num_seqs, motifs_of_seqs, motif_length, all_seqs, PCA_y, iterations, False)

    else: # One-hot embedding
        if shap_input == True:
            CV_predictions, CV_std_predictions = onehot_train_test_cv(num_seqs, motifs_of_seqs, motif_length, all_seqs, PCA_y, 5, True)

        CV_predictions, CV_std_predictions = onehot_train_test_cv(num_seqs, motifs_of_seqs, motif_length, all_seqs, PCA_y, iterations, False)

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
            if embedding_choice == 1: # frequency
                predictions = freq_unknown_predictions(folder, motifs_of_seqs, motif_length, motifs, PCA_y, all_seqs)

            elif embedding_choice == 2: # James-Stein
                predictions = JS_unknown_predictions(folder, motifs_of_seqs, motif_length, motifs, PCA_y, all_seqs)

            else: # One-hot encoding
                predictions = onehot_unknown_predictions(folder, motifs_of_seqs, motif_length, motifs, PCA_y, all_seqs)

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

    seqs = pd.DataFrame(all_seqs.iloc[:,12:32])

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

def motif_freq_embedding(num_seqs, motifs, motifs_of_seqs):
    motif_freq = np.zeros((num_seqs, len(motifs[:,1])))
    ints_motifs_seqs = motifs_of_seqs.iloc[:,:-1].astype(float)

    for i in range(num_seqs):
        unique, counts = np.unique(ints_motifs_seqs.iloc[i,:], return_counts=True)
        unique = unique.astype(int)

        motif_freq[i,unique] = counts

    return motif_freq


def JS_train_test_cv(num_seqs, motifs_of_seqs, motif_length, all_seqs, y_fit, iterations, shap_plot):
    print("Beginning cross validation...")

    existing_descriptors = all_seqs.iloc[:,[6,10]]
    y_data = y_fit

    skf = StratifiedKFold(n_splits = 5, shuffle = True)

    predictions = np.zeros((num_seqs, iterations))

    all_categories = []
    cat_each = np.arange(0, 2**motif_length, 1)

    for i in range(20 - motif_length + 1):
        all_categories.append(cat_each)

    for fit in range(iterations):
        if (fit % 10) == 0:
            print("Iteration number ", fit)

        for train_index, test_index in skf.split(all_seqs, all_seqs["Colorz"]):
            existing_train, existing_test = existing_descriptors.iloc[train_index,:], existing_descriptors.iloc[test_index,:]
            motifs_train, motifs_test = motifs_of_seqs.iloc[train_index,:], motifs_of_seqs.iloc[test_index,:]
            y_train, y_test = y_data[train_index], y_data[test_index]
            
            encoder = ce.JamesSteinEncoder().fit(motifs_train, y_train)
            motifs_enc = encoder.transform(motifs_train)

            scaler = StandardScaler().fit(existing_train)
            existing_train_enc = scaler.transform(existing_train)

            X_train = np.hstack((motifs_enc, existing_train_enc))

            clf = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, n_estimators = 100, subsample = 0.7)
            clf.fit(X_train, y_train)


            test_enc = encoder.transform(motifs_test)
            existing_test_enc = scaler.transform(existing_test)
            X_test = np.hstack((test_enc, existing_test_enc))

            preds = clf.predict(X_test)

            predictions[test_index, fit] = preds

            if shap_plot == True:
                explainer = shap.Explainer(clf)
                explanation = explainer(X_test)
                shap_values = explanation.values

                shap.plots.beeswarm(explanation)
    
    avg_predictions = np.mean(predictions, axis=1)
    SD_predictions = np.std(predictions, axis=1)

    return avg_predictions, SD_predictions

def onehot_train_test_cv(num_seqs, motifs_of_seqs, motif_length, all_seqs, y_fit, iterations, shap_plot):
    print("Beginning cross validation...")

    existing_descriptors = all_seqs.iloc[:,[6,10]]
    y_data = y_fit

    skf = StratifiedKFold(n_splits = 5, shuffle = True)

    predictions = np.zeros((num_seqs, iterations))

    all_categories = []
    cat_each = np.arange(0, 2**motif_length, 1)

    for i in range(20 - motif_length + 1):
        all_categories.append(cat_each)

    for fit in range(iterations):
        if (fit % 10) == 0:
            print("Iteration number ", fit)

        for train_index, test_index in skf.split(all_seqs, all_seqs["Colorz"]):
            existing_train, existing_test = existing_descriptors.iloc[train_index,:], existing_descriptors.iloc[test_index,:]
            motifs_train, motifs_test = motifs_of_seqs.iloc[train_index,:], motifs_of_seqs.iloc[test_index,:]
            y_train, y_test = y_data[train_index], y_data[test_index]

            encoder = OneHotEncoder(categories=all_categories).fit(motifs_train)
            motifs_enc = encoder.transform(motifs_train).toarray()

            scaler = StandardScaler().fit(existing_train)
            existing_train_enc = scaler.transform(existing_train)

            X_train = np.hstack((motifs_enc, existing_train_enc))

            clf = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, n_estimators = 100, subsample = 0.7)
            clf.fit(X_train, y_train)

            test_enc = encoder.transform(motifs_test).toarray()
            existing_test_enc = scaler.transform(existing_test)
            X_test = np.hstack((test_enc, existing_test_enc))

            preds = clf.predict(X_test)

            predictions[test_index, fit] = preds

            if shap_plot == True:
                explainer = shap.Explainer(clf)
                explanation = explainer(X_test)
                shap_values = explanation.values

                shap.plots.beeswarm(explanation)
    
    avg_predictions = np.mean(predictions, axis=1)
    SD_predictions = np.std(predictions, axis=1)

    return avg_predictions, SD_predictions


def freq_embedding_CV(num_seqs, motif_freq, motif_length, all_seqs, PCA_y, iterations, shap_plot):
    print("Beginning cross validation...")

    existing_descriptors = all_seqs.iloc[:,[6,8,10]]
    y_data = PCA_y

    skf = StratifiedKFold(n_splits = 5, shuffle = True)

    predictions = np.zeros((num_seqs, iterations))

    idx = 0

    for fit in range(iterations):
        if (fit % 10) == 0:
            print("Iteration number ", fit)

        for train_index, test_index in skf.split(all_seqs, all_seqs["Colorz"]):
            existing_train, existing_test = existing_descriptors.iloc[train_index], existing_descriptors.iloc[test_index]
            motifs_train, motifs_test = motif_freq[train_index,:], motif_freq[test_index,:]
            y_train, y_test = y_data[train_index], y_data[test_index]
                
            scaler = StandardScaler().fit(existing_train)
            existing_train_enc = scaler.transform(existing_train)

            X_train = np.hstack((motifs_train, existing_train_enc))

            clf = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, n_estimators = 100, subsample = 0.7)
            clf.fit(X_train, y_train)

            existing_test_enc = scaler.transform(existing_test)
            X_test = np.hstack((motifs_test, existing_test_enc))

            preds = clf.predict(X_test)

            predictions[test_index, fit] = preds
            
            if shap_plot == True:
                explainer = shap.Explainer(clf)
                explanation = explainer(X_test)
                shap_values = explanation.values
                shap.plots.beeswarm(explanation)
    
    avg_predictions = np.mean(predictions, axis=1)
    SD_predictions = np.std(predictions, axis=1)

    return avg_predictions, SD_predictions


def export_predictions(all_seqs, PCA_y, CV_predictions, CV_SD_predictions, folder):
    print("Exporting out-of-sample predictions to excel folder...")
    CV_df = pd.DataFrame()
    CV_df["Color"] = all_seqs["Color"]
    CV_df["PCA of color group"] = PCA_y
    CV_df["Predictions"] = CV_predictions
    CV_df["Standard deviation"] = CV_SD_predictions

    print("MSE is ", mean_squared_error(PCA_y, CV_predictions, squared=False))
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


def freq_unknown_predictions(folder, motif_freq, motif_length, motifs, PCA_y, all_seqs):
    print("Fitting unknown sequences to model...")
    existing_descriptors = all_seqs.iloc[:,[6,8,10]]

    scaler = StandardScaler().fit(existing_descriptors)
    existing_train_enc = scaler.transform(existing_descriptors)

    X_train = np.hstack((motif_freq, existing_train_enc))

    clf = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, n_estimators = 100, subsample = 0.7)
    clf.fit(X_train, PCA_y)

    # read in your sequences
    unknown_seq_df, unknown_seqs, num_unknown = read_unknown_from_file(folder)
    unknown_descriptors = unknown_seq_df.iloc[:,[6,8,10]]

    # motifs of unknowns
    motifs_unknowns = motifs_from_seqs(num_unknown, motif_length, unknown_seqs, motifs)
    unknown_freq = motif_freq_embedding(unknown_seqs, motifs, motifs_unknowns)

    scaled_descriptors = scaler.transform(unknown_descriptors)

    X_test = np.hstack((unknown_freq, scaled_descriptors))

    # these numbers will be the PCA values. Use the table exported in line 76 to back out which PCA number corresponds to which color group
    preds = clf.predict(X_test)

    return preds


def onehot_unknown_predictions(folder, motifs_of_seqs, motif_length, motifs, PCA_y, all_seqs):
    print("Fitting unknown sequences to model...")
    existing_descriptors = all_seqs.iloc[:,[6,8,10]]
        
    encoder = OneHotEncoder(categories=all_categories).fit(motifs_of_seqs)
    motifs_enc = encoder.transform(motifs_of_seqs).toarray()

    scaler = StandardScaler().fit(existing_descriptors)
    existing_train_enc = scaler.transform(existing_descriptors)

    X_train = np.hstack((motifs_enc, existing_train_enc))

    clf = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, n_estimators = 100, subsample = 0.7)
    clf.fit(X_train, PCA_y)

    # read in your sequences
    unknown_seq_df, unknown_seqs, num_unknown = read_unknown_from_file(folder)
    unknown_descriptors = unknown_seq_df.iloc[:,[6,8,10]]

    # motifs of unknowns
    motifs_unknowns = motifs_from_seqs(num_unknown, motif_length, unknown_seqs, motifs)
    encoded_unknowns = encoder.transform(motifs_unknowns).toarray()

    scaled_descriptors = scaler.transform(unknown_descriptors)

    X_test = np.hstack((encoded_unknowns, scaled_descriptors))

    # these numbers will be the PCA values. Use the table exported in line 76 to back out which PCA number corresponds to which color group
    preds = clf.predict(X_test)

    return preds


def JS_unknown_predictions(folder, motifs_of_seqs, motif_length, motifs, PCA_y, all_seqs):
    print("Fitting unknown sequences to model...")
    existing_descriptors = all_seqs.iloc[:,[6,8,10]]

    full_encoder = ce.JamesSteinEncoder().fit(motifs_of_seqs, PCA_y)
    train_encoded = full_encoder.transform(motifs_of_seqs, PCA_y)

    scaler = StandardScaler().fit(existing_descriptors)
    existing_train_enc = scaler.transform(existing_descriptors)

    X_train = np.hstack((train_encoded, existing_train_enc))

    clf = GradientBoostingRegressor(learning_rate=0.1, max_depth=7, n_estimators=100, subsample= 0.7)
    clf.fit(X_train, PCA_y)

    # read in your sequences
    unknown_seq_df, unknown_seqs, num_unknown = read_unknown_from_file(folder)
    unknown_descriptors = unknown_seq_df.iloc[:,[6,8,10]]

    # motifs of unknowns
    motifs_unknowns = motifs_from_seqs(num_unknown, motif_length, unknown_seqs, motifs)
    encoded_unknowns = full_encoder.transform(motifs_unknowns)

    scaled_descriptors = scaler.transform(unknown_descriptors)

    X_test = np.hstack((encoded_unknowns, scaled_descriptors))

    # these numbers will be the PCA values. Use the table exported in line 76 to back out which PCA number corresponds to which color group
    predictions = clf.predict(X_test) 

    return predictions




def export_unknowns(predictions, folder):
    print("Exporting unknowns to folder...")

    preds_df = pd.DataFrame(predictions)
    preds_df.to_excel(folder + "/PCA_of_unknowns.xlsx")