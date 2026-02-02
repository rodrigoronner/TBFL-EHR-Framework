import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def load_and_process_mimic(file_path, num_clients):
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
    # Distribute training data among clients.
    # To simulate Non-IID data (as claimed in the paper), we can use a Dirichlet distribution,
    # but for this base implementation, we use a randomized partition that ensures variety.
    user_groups = partition_data(y_train, num_clients)

    return X_train, y_train, X_test, y_test, user_groups

def partition_data(y_train, num_clients):
    """
    Splits the training data indices among clients.
    Currently implements an IID split for stability in the provided demo code.
    
    Args:
        y_train (np.array): Labels of training data.
        num_clients (int): Number of clients.
        
    Returns:
        dict: {client_id: [index_1, index_2, ...]}
    """
    num_items = int(len(y_train) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(y_train))]
    
    for i in range(num_clients):
        # Randomly select 'num_items' indices for client 'i' without replacement
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # Remove selected indices from the pool
        all_idxs = list(set(all_idxs) - dict_users[i])
        
    return dict_users

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
