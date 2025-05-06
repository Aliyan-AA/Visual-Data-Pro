import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score, roc_curve, auc
)

def preprocess_data(data, handle_missing='Drop rows with missing values', 
                   handle_outliers=False, normalize=False):
    """
    Preprocess the dataset by handling missing values, outliers, and normalization.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataset
    handle_missing : str
        How to handle missing values: drop, fill with mean, median, or mode
    handle_outliers : bool
        Whether to remove outliers using IQR method
    normalize : bool
        Whether to normalize numeric features
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataset
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Handle missing values
    if handle_missing == 'Drop rows with missing values':
        df = df.dropna()
    elif handle_missing == 'Fill numeric with mean':
        for col in df.select_dtypes(include=['number']).columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
        # For non-numeric columns, fill with mode
        for col in df.select_dtypes(exclude=['number']).columns:
            if df[col].isna().any():
                mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_value)
    elif handle_missing == 'Fill with median':
        for col in df.select_dtypes(include=['number']).columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        # For non-numeric columns, fill with mode
        for col in df.select_dtypes(exclude=['number']).columns:
            if df[col].isna().any():
                mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_value)
    elif handle_missing == 'Fill with mode':
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                else:
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(mode_value)
    
    # Handle outliers using IQR method for numeric columns
    if handle_outliers:
        for col in df.select_dtypes(include=['number']).columns:
            if df[col].nunique() > 1:  # Only process if there's more than one unique value
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Only process if IQR is not zero
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Normalize numeric features
    if normalize:
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

def feature_engineering(data, target_column, feature_columns=None, create_indicators=None):
    """
    Perform feature engineering on the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The preprocessed dataset
    target_column : str
        The target variable column name
    feature_columns : list
        List of columns to use as features (if None, use all except target)
    create_indicators : list
        List of technical indicators to create for Yahoo Finance data
        
    Returns:
    --------
    tuple
        (X, y) - feature dataframe and target series
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # If no feature columns specified, use all except target
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    # For stock data, create technical indicators
    if create_indicators:
        if "Moving Averages" in create_indicators and "Close" in df.columns:
            # Create simple moving averages
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Create exponential moving averages
            df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # Add moving average crossover signals
            df['SMA_5_20_Cross'] = np.where(df['SMA_5'] > df['SMA_20'], 1, 0)
        
        if "RSI" in create_indicators and "Close" in df.columns:
            # Calculate RSI (Relative Strength Index)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        if "MACD" in create_indicators and "Close" in df.columns:
            # Calculate MACD (Moving Average Convergence Divergence)
            ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_12 - ema_26
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        if "Bollinger Bands" in create_indicators and "Close" in df.columns:
            # Calculate Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
            df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        if "Volume Features" in create_indicators and "Volume" in df.columns:
            # Create volume indicators
            df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            
            # On-Balance Volume (OBV)
            df['OBV'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], 
                       np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0)).cumsum()
    
    # Handle date columns - extract features if any datetime columns exist
    date_cols = [col for col in feature_columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    for col in date_cols:
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_dayofweek'] = df[col].dt.dayofweek
        
        # Remove original date column from features
        feature_columns.remove(col)
        # Add new date-based features
        feature_columns.extend([f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek'])
    
    # Handle categorical features
    cat_cols = [col for col in feature_columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']
    for col in cat_cols:
        # For binary categorical variables
        if df[col].nunique() == 2:
            # Convert to 0-1
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            # For multi-category variables, create dummies and drop the original
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            
            # Update feature columns list
            feature_columns.remove(col)
            feature_columns.extend(dummies.columns)
    
    # Drop rows with NaN values that may have been introduced during feature engineering
    df = df.dropna()
    
    # Extract features and target
    updated_features = [col for col in feature_columns if col in df.columns]
    X = df[updated_features]
    y = df[target_column]
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, model_type, params=None):
    """
    Train a machine learning model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    model_type : str
        Type of model to train ('Linear Regression', 'Logistic Regression', 'K-Means Clustering')
    params : dict
        Model parameters
        
    Returns:
    --------
    object
        Trained model
    """
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, (pd.Series, np.ndarray)):
        raise ValueError("X_train must be a DataFrame and y_train must be a Series or numpy array")
    
    if X_train.empty or len(y_train) == 0:
        raise ValueError("Training data cannot be empty")
    
    if params is None:
        params = {}
    
    try:
        if model_type == "Linear Regression":
            model = LinearRegression(**params)
            model.fit(X_train, y_train)
        
        elif model_type == "Logistic Regression":
            # Convert y to binary if it's not already (threshold at median)
            if len(np.unique(y_train)) > 2:
                y_binary = (y_train > np.median(y_train)).astype(int)
                model = LogisticRegression(**params)
                model.fit(X_train, y_binary)
            else:
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
        
        elif model_type == "K-Means Clustering":
            if 'n_clusters' not in params:
                params['n_clusters'] = 3  # Default number of clusters
            model = KMeans(**params)
            model.fit(X_train)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    except Exception as e:
        raise RuntimeError(f"Error training model: {str(e)}")

def evaluate_model(model, X_test, y_test, model_type):
    """
    Evaluate the trained model on the test data.
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    model_type : str
        Type of model to evaluate
        
    Returns:
    --------
    tuple
        (evaluation_metrics, predictions)
    """
    results = {}
    
    if model_type == "Linear Regression":
        predictions = model.predict(X_test)
        results['r2'] = r2_score(y_test, predictions)
        results['mse'] = mean_squared_error(y_test, predictions)
        results['mae'] = mean_absolute_error(y_test, predictions)
    
    elif model_type == "Logistic Regression":
        # Convert y_test to binary if needed (threshold at median)
        if len(np.unique(y_test)) > 2:
            y_test_binary = (y_test > y_test.median()).astype(int)
            y_test_for_eval = y_test_binary
        else:
            y_test_for_eval = y_test
            
        predictions = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results['accuracy'] = accuracy_score(y_test_for_eval, predictions)
        results['precision'] = precision_score(y_test_for_eval, predictions, zero_division=0)
        results['recall'] = recall_score(y_test_for_eval, predictions, zero_division=0)
        results['f1'] = f1_score(y_test_for_eval, predictions, zero_division=0)
        results['confusion_matrix'] = pd.DataFrame(
            confusion_matrix(y_test_for_eval, predictions),
            columns=['Predicted Negative', 'Predicted Positive'],
            index=['Actual Negative', 'Actual Positive']
        )
        
        # Add ROC data if we have probability predictions
        if pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test_for_eval, pred_proba)
            results['roc_auc'] = auc(fpr, tpr)
            results['fpr'] = fpr
            results['tpr'] = tpr
    
    elif model_type == "K-Means Clustering":
        predictions = model.predict(X_test)
        
        # Silhouette score
        if len(np.unique(predictions)) > 1:  # Ensure we have at least 2 clusters
            results['silhouette'] = silhouette_score(X_test, predictions)
        else:
            results['silhouette'] = 0  # Default if only one cluster
            
        results['inertia'] = model.inertia_  # Sum of squared distances to closest centroid
    
    return results, predictions
