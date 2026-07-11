import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek

def load_and_process_mimic(file_path, num_clients, dirichlet_alpha=0.5):
    """
    Loads the MIMIC-IV dataset (EHR features), performs preprocessing, 
    and partitions the data among federated clients.

    Args:
        file_path (str): Path to the CSV file containing patient features.
        num_clients (int): Number of federated clients (hospitals) to split the data into.

    Returns:
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        X_test (np.array): Testing features.
        y_test (np.array): Testing labels.
        user_groups (dict): A dictionary where keys are client IDs (0 to num_clients-1) 
                            and values are lists of data indices belonging to that client.
    """
    print(f"📂 Loading data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Fallback for demonstration purposes if CSV is missing
        print("⚠️  CSV not found. Generating synthetic data for demonstration...")
        df = generate_synthetic_mimic_data()

    # 1. Define Target Variable
    # Assuming 'hospital_expire_flag' is the target (Mortality Prediction)
    target_col = 'hospital_expire_flag'
    
    # Check if target column exists, otherwise use the last column
    if target_col not in df.columns:
        target_col = df.columns[-1]

    # Drop non-clinical identifier columns (e.g. the admission ID): these carry no
    # predictive signal and are not part of the paper's feature set (Sec. 3.1.1),
    # so leaving them in would let the MLP fit noise from an arbitrary row ID.
    id_cols = [c for c in df.columns if c.lower() in ('hadm_id', 'subject_id', 'row_id')]
    df = df.drop(columns=id_cols)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Data Preprocessing
    # Handle Missing Values (Imputation)
    # Numerical: Mean | Categorical: Mode
    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='mean')
        X[num_cols] = imputer_num.fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        # One-Hot Encoding for categorical variables
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Convert to NumPy for PyTorch compatibility
    X = X.values
    y = y.values

    # Normalize/Scale Numerical Features (Critical for Neural Networks)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"✅ Data Processed. Shape: {X.shape}")

    # 3. Train/Test Split (80/20)
    # The Test set is held out globally to evaluate the Global Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Federated Partitioning (User Groups)
    # Distribute training data among clients using a Dirichlet(alpha) distribution over
    # the label distribution, producing realistic non-IID class-prevalence heterogeneity
    # across federated clients (Sec. 3.1.2 of the paper).
    user_groups = partition_data_dirichlet(y_train, num_clients, alpha=dirichlet_alpha)

    return X_train, y_train, X_test, y_test, user_groups

def partition_data_dirichlet(y_train, num_clients, alpha=0.5, seed=42):
    """
    Splits training data indices among clients using a Dirichlet(alpha) distribution,
    the standard approach for simulating non-IID class-prevalence skew in Federated
    Learning benchmarks. Lower alpha => more heterogeneous (skewed) clients.

    Args:
        y_train (np.array): Labels of training data.
        num_clients (int): Number of clients.
        alpha (float): Dirichlet concentration parameter (paper uses alpha=0.5).
        seed (int): RNG seed for reproducibility.

    Returns:
        dict: {client_id: set(index_1, index_2, ...)}
    """
    rng = np.random.default_rng(seed)
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)

    client_idxs = [[] for _ in range(num_clients)]
    for c in classes:
        c_idxs = np.where(y_train == c)[0]
        rng.shuffle(c_idxs)

        # Sample a class-prevalence proportion per client from a Dirichlet distribution
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions) * len(c_idxs)).astype(int)[:-1]
        for client_id, split in enumerate(np.split(c_idxs, split_points)):
            client_idxs[client_id].extend(split.tolist())

    return {i: set(client_idxs[i]) for i in range(num_clients)}

def build_client_datasets(X_train, y_train, user_groups):
    """
    Builds one locally-balanced dataset per federated client.

    Per Sec. 3.1.4 of the paper, the train/test split happens first (globally), and
    SMOTETomek is then applied exclusively to each client's local training fold, never
    to the held-out test set, which remains in its original imbalanced distribution.

    Args:
        X_train (np.array): Global training features (post train/test split).
        y_train (np.array): Global training labels.
        user_groups (dict): {client_id: set(indices)} from partition_data_dirichlet.

    Returns:
        dict: {client_id: (X_balanced, y_balanced)}
    """
    client_data = {}
    for client_id, idxs in user_groups.items():
        idxs = sorted(idxs)
        X_c, y_c = X_train[idxs], y_train[idxs]

        # SMOTETomek needs a handful of minority-class samples to draw neighbors from;
        # a Dirichlet split can occasionally starve a client of one class entirely, in
        # which case we skip resampling for that client rather than error out.
        class_counts = np.bincount(y_c.astype(int))
        minority_count = class_counts.min() if len(class_counts) > 1 else 0

        if len(class_counts) > 1 and minority_count > 5:
            smt = SMOTETomek(random_state=42)
            X_bal, y_bal = smt.fit_resample(X_c, y_c)
        else:
            X_bal, y_bal = X_c, y_c

        client_data[client_id] = (X_bal, y_bal)

    return client_data

def generate_synthetic_mimic_data(samples=1000):
    """
    Generates a synthetic dataframe mimicking MIMIC-IV structure 
    if the real CSV is not present (facilitates reproducibility check by reviewers).
    """
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 90, samples),
        'heart_rate': np.random.normal(80, 15, samples),
        'sbp': np.random.normal(120, 20, samples), # Systolic Blood Pressure
        'wbc': np.random.normal(9, 3, samples),    # White Blood Cells
        'gender': np.random.choice([0, 1], samples),
        'icu_los': np.random.exponential(3, samples), # Length of Stay
        'hospital_expire_flag': np.random.choice([0, 1], samples, p=[0.85, 0.15]) # 15% Mortality rate
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Test the loader independently
    X_tr, y_tr, X_te, y_te, groups = load_and_process_mimic('dummy.csv', 3)
    print(f"Test Run: {len(X_tr)} training samples distributed among 3 clients.")
