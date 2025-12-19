import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from langchain_core.tools import tool
import plotly.express as px

@tool
def clean_data(file_path: str, drop_null_thresh: float = 0.5, impute_num: str = "median", impute_cat: str = "mode", convert_dates: list[str] = None, drop_columns: list[str] = None, remove_outliers: bool = True) -> str:
    """
    Cleans the dataset by handling null values, outlier detection, and data validation.
    - drop_null_thresh: drop columns with more than this % of nulls.
    - impute_num: 'median' or 'mean'.
    - impute_cat: 'mode' or 'constant'.
    - convert_dates: list of columns to convert to datetime.
    - drop_columns: list of specific columns to drop.
    - remove_outliers: whether to apply IQR-based removal.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return f"Error: The file at {file_path} is empty."

        # 0. Manual Column Dropping
        if drop_columns:
            existing_drops = [c for c in drop_columns if c in df.columns]
            if existing_drops:
                df = df.drop(columns=existing_drops)

        # 1. Null Value Handling
        limit = len(df) * drop_null_thresh
        df = df.dropna(axis=1, thresh=limit)
        
        for col in df.columns:
            if df[col].isnull().any():
                if np.issubdtype(df[col].dtype, np.number):
                    fill_val = df[col].median() if impute_num == "median" else df[col].mean()
                    df[col] = df[col].fillna(fill_val)
                else:
                    fill_val = df[col].mode()[0] if impute_cat == "mode" else "Missing"
                    df[col] = df[col].fillna(fill_val)
        
        # 2. Data Validation
        if convert_dates:
            for col in convert_dates:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

        # 3. Outlier Detection (Safe IQR)
        if remove_outliers and not df.empty:
            original_len = len(df)
            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                temp_df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                # Safety check: Don't remove if it drops > 90% of current data or results in < 5 rows
                if len(temp_df) > 5 and len(temp_df) > (len(df) * 0.1):
                    df = temp_df

        if df.empty:
            return "Error: Cleaning process resulted in an empty dataset. Check your null threshold or outlier settings."

        # 4. Save
        # Robust absolute path handling
        base, ext = os.path.splitext(file_path)
        if base.endswith("_cleaned"):
            output_path = file_path
        else:
            output_path = f"{base}_cleaned{ext}"
            
        df.to_csv(output_path, index=False)
        msg = f"Data cleaning complete. Saved to: {output_path}."
        if drop_columns:
            msg += f" Dropped: {existing_drops}."
        return msg + f" Handled nulls, validated types, and sanitized {len(df)} rows."
    except Exception as e:
        return f"Error during cleaning: {str(e)}"

@tool
def perform_eda(file_path: str) -> str:
    """
    Performs Exploratory Data Analysis. 
    Identifies high correlations and suggests specific columns to drop.
    Generates signals for the UI to render distribution and correlation plots.
    """
    try:
        df = pd.read_csv(file_path)
        num_df = df.select_dtypes(include=[np.number])
        
        if num_df.empty:
            return "No numerical data available for EDA."
            
        # Correlation
        corr = num_df.corr()
        high_corr_pairs = []
        drop_suggestions = set()
        
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > 0.85:
                    col_i = corr.columns[i]
                    col_j = corr.columns[j]
                    high_corr_pairs.append(f"{col_i} & {col_j} ({corr.iloc[i, j]:.2f})")
                    # Suggest dropping the second column in the pair
                    drop_suggestions.add(col_j)
        
        report = "EDA Report:\n"
        if high_corr_pairs:
            report += f"- High Correlations Found: {', '.join(high_corr_pairs)}.\n"
            report += f"- SUGGESTED DROPS to avoid multicollinearity: {list(drop_suggestions)}\n"
        else:
            report += "- No extreme multicollinearity detected (>0.85).\n"
        
        report += f"- Data Shape: {df.shape}\n"
        report += "- Signals generated for UI: Correlation Heatmap and Feature Distributions."
        
        return report
    except Exception as e:
        return f"Error during EDA: {str(e)}"

@tool
def perform_clustering(file_path: str, columns: list[str], k: int) -> str:
    """
    Performs K-Means clustering on specified columns of a CSV file.
    Identifies numerical vs categorical columns, preprocesses them, 
    applies dimensionality reduction (PCA 2D), and saves results.
    Returns the path to the new CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        data = df[columns].copy()

        # 1. Identify types
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

        # 2. Preprocessing
        processed_parts = []
        if num_cols:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data[num_cols])
            processed_parts.append(pd.DataFrame(scaled, columns=num_cols))
        
        if cat_cols:
            encoder = OneHotEncoder(sparse_output=False)
            encoded = encoder.fit_transform(data[cat_cols])
            encoded_cols = encoder.get_feature_names_out(cat_cols)
            processed_parts.append(pd.DataFrame(encoded, columns=encoded_cols))

        processed_df = pd.concat(processed_parts, axis=1)

        # 3. K-Means
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(processed_df)
        df['Cluster'] = clusters

        # 4. PCA for Visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(processed_df)
        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]

        # 5. Save Results
        output_path = file_path.replace(".csv", "_clustered.csv")
        df.to_csv(output_path, index=False)
        
        return f"Clustering complete. Results saved to: {output_path}. PCA components (PCA1, PCA2) and 'Cluster' labels added."
    except Exception as e:
        return f"Error during clustering: {str(e)}"

@tool
def generate_visualization(file_path: str) -> str:
    """
    Generates a 2D scatter plot from PCA components in a clustered CSV file.
    Expects 'PCA1', 'PCA2', and 'Cluster' columns to exist.
    """
    try:
        df = pd.read_csv(file_path)
        if 'PCA1' not in df.columns or 'PCA2' not in df.columns or 'Cluster' not in df.columns:
            return "Error: File does not contain 'PCA1', 'PCA2' or 'Cluster' columns. Perform clustering first."
        
        fig = px.scatter(
            df, 
            x='PCA1', y='PCA2', 
            color='Cluster', 
            title=f"Cluster Visualization for {os.path.basename(file_path)}",
            labels={'Cluster': 'Cluster ID'},
            template="plotly_dark"
        )
        
        # We'll handle the actual display in the Streamlit app.
        # Here we save the figure to a session state or global var is not possible, 
        # so we return a signal that visualization is ready and passing JSON or similar is complex.
        # Actually, let's just return success and the app will know to render it if it sees this.
        return "Visualization generated. Use the provided dataframe to render the scatter plot."
    except Exception as e:
        return f"Error during visualization: {str(e)}"
