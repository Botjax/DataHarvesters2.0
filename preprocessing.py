import pandas as pd
import os
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import OneHotEncoder
import numpy

# saving to this directory
PROCESSED_DATA_DIR = 'Resources\processed_data'

def load_dataset(filename):
    return pd.read_csv(filename)

# saving the dataset
def save_dataset(df, filename):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Dataset saved at {path}")

def missing_vals_check(df):
    missing_data = df.isnull().sum()  # Number of null values for each column stored in the "missing_data" pandas series

    if missing_data.sum() == 0: #total amount of missing data across all columns
        print("--No missing data in the dataset.")
    else:
        print("--Missing number of data: ", missing_data.sum())
    return df

def detect_outliers(df):
    outliers = {}
    total_outliers = 0

    # age outliers (should be between 0 and 110)
    age_outliers = df[(df['Age'] < 0) | (df['Age'] > 110)]
    if not age_outliers.empty:
        outliers['Age'] = age_outliers
        total_outliers += len(age_outliers)

    # BMI outliers (should be between 10 and 100)
    bmi_outliers = df[(df['BMI'] < 10) | (df['BMI'] > 100)]
    if not bmi_outliers.empty:
        outliers['BMI'] = bmi_outliers
        total_outliers += len(bmi_outliers)

    # Systolic BP outliers (should be between 90 and 200)
    sbp_outliers = df[(df['SystolicBP'] < 90) | (df['SystolicBP'] > 200)]
    if not sbp_outliers.empty:
        outliers['SystolicBP'] = sbp_outliers
        total_outliers += len(sbp_outliers)

    # Diastolic BP outliers (should be between 50 and 130)
    dbp_outliers = df[(df['DiastolicBP'] < 50) | (df['DiastolicBP'] > 130)]
    if not dbp_outliers.empty:
        outliers['DiastolicBP'] = dbp_outliers
        total_outliers += len(dbp_outliers)

    ##ADD MORE OUTLIER CHECKS FOR OTHER COLUMNS

    if total_outliers == 0:
        print("--No outliers in the dataset.")
    else:
        print("--Total number of outliers is: ", total_outliers)

    return df


# Function to check and handle missing data
def clean_ckd_dataset(df):

    # dropping an irrelevant column
    df = df.drop(columns=['DoctorInCharge'])
    print("--Removed irrelevant columns.")

    # checks if there are NaNs or null values and
    cleaned_df = missing_vals_check(df)

    # check for outliers
    cleaned_df = detect_outliers(cleaned_df)

    #cleaned_df =

    return cleaned_df

def balancing_ckd_dataset(df):
    print("Balancing dataset using SMOTE...")

    # Separating features, target, and IDs before sampling
    #original_ids = df['PatientID']
    features = df.drop(columns=['Diagnosis', 'PatientID'])
    target = df['Diagnosis']

    categorical_features = ['Gender', 'Ethnicity', 'SocioeconomicStatus',
                            'EducationLevel', 'Smoking', 'FamilyHistoryKidneyDisease',
                            'FamilyHistoryHypertension', 'FamilyHistoryDiabetes', 'PreviousAcuteKidneyInjury',
                            'UrinaryTractInfections', 'ACEInhibitors', 'Diuretics', 'Statins', 'AntidiabeticMedications',
                            'Edema', 'HeavyMetalsExposure', 'OccupationalExposureChemicals', 'WaterQuality']  # hardcoding the names of categorical features
    categorical_indices = [features.columns.get_loc(col) for col in categorical_features] #finding indices of those columns

    # applying SMOTENC
    smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42) #setting a seed value for reproducibility
    features_resampled, target_resampled = smote_nc.fit_resample(features, target)

    df_balanced = pd.DataFrame(features_resampled, columns=features.columns)
    df_balanced['Diagnosis'] = target_resampled
    df_balanced.insert(0, 'PatientID', range(1, len(df_balanced) + 1)) #adding back the PatientID column and numbering them

    print("Before SMOTENC:")
    print(target.value_counts())
    print("\nAfter SMOTENC:")
    print(pd.Series(target_resampled).value_counts())

    return categorical_features, df_balanced

def onehot_encode_ckd_dataset(df, categorical_columns):
    # separating features and target
    features = df.drop(columns=['Diagnosis'])
    target = df['Diagnosis']

    #don't get one-hot encoded
    binary_ordinal_columns = ['Gender', 'Smoking', 'FamilyHistoryKidneyDisease',
                              'FamilyHistoryHypertension', 'FamilyHistoryDiabetes',
                              'PreviousAcuteKidneyInjury', 'UrinaryTractInfections',
                              'ACEInhibitors', 'Diuretics', 'Statins', 'AntidiabeticMedications',
                              'Edema', 'HeavyMetalsExposure', 'OccupationalExposureChemicals', 'WaterQuality']

    categorical_columns_for_onehot = [col for col in categorical_columns if col not in binary_ordinal_columns]

    encoder = OneHotEncoder(sparse_output=False)  # sparse=false to make sure it shows 0s. drop='first' to avoid multicollinearity.
    encoded_array = encoder.fit_transform(features[categorical_columns_for_onehot]) #applying one-hot encoding to the categorical_columns not in binary_ordinal_columns
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_columns_for_onehot))

    features_remaining = features.drop(columns=categorical_columns_for_onehot).reset_index(drop=True)
    features_encoded = pd.concat([features_remaining, encoded_df], axis=1)

    df_encoded = features_encoded.copy()
    df_encoded['Diagnosis'] = target.reset_index(drop=True)

    print("One-hot encoding complete.")
    return df_encoded


def main():
    print("Loading Chronic_Kidney_Dsease_data.csv...")
    df1 = load_dataset("Resources/datasets/Chronic_Kidney_Dsease_data.csv")

    print("||Cleaning dataset||")
    df1_clean = clean_ckd_dataset(df1)
    save_dataset(df1_clean, "processed_ckd_new.csv")

    print("||Balancing dataset||")
    categorical_features, balanced_df = balancing_ckd_dataset(df1_clean)
    print("Dataset is balanced: ", balanced_df)
    save_dataset(balanced_df, "processed_ckd_balanced.csv")

    print("||One-hot encoding dataset||")
    onehot_df = onehot_encode_ckd_dataset(balanced_df, categorical_features)
    save_dataset(onehot_df, "processed_ckd_onehot.csv")

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()

