import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix

def plot_missing_values(data):
    """
    Create a bar chart of missing values in each column.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to analyze
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart of missing values
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Empty dataset provided",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    missing = data.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    
    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values in the dataset",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    try:
        # Create missing values bar chart
        fig = px.bar(
            x=missing.index, 
            y=missing.values,
            labels={'x': 'Columns', 'y': 'Missing Value Count'},
            title='Missing Values by Column',
            color=missing.values,
            color_continuous_scale='Viridis'
        )
        
        # Add percentage labels
        for i, col in enumerate(missing.index):
            fig.add_annotation(
                x=col,
                y=missing[col],
                text=f"{(missing[col]/len(data)*100):.1f}%",
                showarrow=False,
                yshift=10
            )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig

def plot_split(train_size, test_size):
    """
    Create a pie chart showing the train/test split.
    
    Parameters:
    -----------
    train_size : int
        Number of training samples
    test_size : int
        Number of testing samples
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Pie chart of the data split
    """
    if not isinstance(train_size, (int, np.integer)) or not isinstance(test_size, (int, np.integer)):
        raise ValueError("train_size and test_size must be integers")
    
    if train_size < 0 or test_size < 0:
        raise ValueError("train_size and test_size must be non-negative")
    
    if train_size == 0 and test_size == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for split visualization",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    try:
        labels = ['Training Data', 'Testing Data']
        values = [train_size, test_size]
        
        fig = px.pie(
            values=values,
            names=labels,
            title='Train/Test Split',
            color_discrete_sequence=['#3366CC', '#FF9900'],
            hole=0.4
        )
        
        # Add percentages
        total = train_size + test_size
        train_pct = train_size / total * 100
        test_pct = test_size / total * 100
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Train: {train_pct:.1f}%<br>Test: {test_pct:.1f}%",
            showarrow=False,
            font=dict(size=14)
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig

def plot_feature_importance(model, feature_names):
    """
    Create a horizontal bar chart of feature importance.
    
    Parameters:
    -----------
    model : object
        Trained model with coefficients
    feature_names : list
        Names of the features
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Bar chart of feature importance
    """
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("feature_names must be a non-empty list")
    
    if not hasattr(model, 'coef_'):
        fig = go.Figure()
        fig.add_annotation(
            text="Model does not have coefficients attribute",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    try:
        # Get coefficients based on model type
        coefficients = model.coef_
        if coefficients.ndim > 1:
            coefficients = coefficients[0]  # Take first row for multi-class
        
        if len(coefficients) != len(feature_names):
            raise ValueError("Number of coefficients does not match number of feature names")
        
        # Create a DataFrame of feature importance
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(coefficients)
        }).sort_values('Importance', ascending=False)
        
        # Plot horizontal bar chart
        fig = px.bar(
            importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig

def plot_regression_results(y_true, y_pred):
    """
    Create a scatter plot of actual vs predicted values for regression.
    
    Parameters:
    -----------
    y_true : pandas.Series or numpy.ndarray
        Actual target values
    y_pred : pandas.Series or numpy.ndarray
        Predicted target values
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Scatter plot of actual vs predicted values
    """
    if not isinstance(y_true, (pd.Series, np.ndarray)) or not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise ValueError("y_true and y_pred must be pandas Series or numpy arrays")
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(y_true) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for regression visualization",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    try:
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred
        })
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='Actual',
            y='Predicted',
            title='Actual vs Predicted Values',
            labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
            opacity=0.7
        )
        
        # Add perfect prediction line
        min_val = min(df['Actual'].min(), df['Predicted'].min())
        max_val = max(df['Actual'].max(), df['Predicted'].max())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating visualization: {str(e)}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig

def plot_classification_results(evaluation_results, y_true, y_pred):
    """
    Create visualizations for classification model results.
    
    Parameters:
    -----------
    evaluation_results : dict
        Evaluation metrics from the model
    y_true : pandas.Series
        Actual target values
    y_pred : pandas.Series
        Predicted target values
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure with ROC curve and confusion matrix
    """
    # Create a figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('ROC Curve', 'Confusion Matrix'),
        specs=[[{'type': 'xy'}, {'type': 'heatmap'}]]
    )
    
    # Add ROC curve if available
    if 'fpr' in evaluation_results and 'tpr' in evaluation_results:
        fig.add_trace(
            go.Scatter(
                x=evaluation_results['fpr'],
                y=evaluation_results['tpr'],
                mode='lines',
                name=f"ROC (AUC = {evaluation_results['roc_auc']:.3f})",
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add diagonal line for random classifier
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Update ROC curve layout
        fig.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", range=[0, 1], row=1, col=1)
    
    # Add confusion matrix
    if 'confusion_matrix' in evaluation_results:
        cm = evaluation_results['confusion_matrix'].values
        
        # Calculate percentages for annotation
        cm_sum = np.sum(cm)
        cm_percentages = cm / cm_sum * 100
        annotations = []
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations.append({
                    'x': j,
                    'y': i,
                    'text': f"{cm[i, j]}<br>({cm_percentages[i, j]:.1f}%)",
                    'showarrow': False,
                    'font': {'color': 'white' if cm[i, j] > cm_sum/4 else 'black'}
                })
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                colorscale='Blues',
                showscale=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(annotations=annotations)
    
    # Update overall layout
    fig.update_layout(
        title='Classification Model Evaluation',
        height=500,
        width=1000
    )
    
    return fig

def plot_clusters(X, clusters, x_feature=None, y_feature=None):
    """
    Create a scatter plot of K-Means clustering results.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature data
    clusters : numpy.ndarray
        Cluster assignments from K-Means
    x_feature : str, optional
        Name of the feature to plot on x-axis
    y_feature : str, optional
        Name of the feature to plot on y-axis
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Scatter plot of clusters
    """
    # If no features specified, use the first two columns
    if x_feature is None:
        x_feature = X.columns[0]
    if y_feature is None:
        y_feature = X.columns[1] if X.shape[1] > 1 else X.columns[0]
    
    # Create a DataFrame for plotting
    df_plot = pd.DataFrame({
        'x': X[x_feature],
        'y': X[y_feature],
        'Cluster': clusters.astype(str)
    })
    
    # Create scatter plot
    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        color='Cluster',
        title='K-Means Clustering Results',
        labels={'x': x_feature, 'y': y_feature},
        opacity=0.7
    )
    
    return fig
