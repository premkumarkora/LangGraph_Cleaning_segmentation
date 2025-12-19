# Requirements Document: Autonomous AI Data Analytics Application

## 1. Overview
The **Autonomous AI Data Analytics Application** is a multi-agent system designed to automate the end-to-end process of data analysis. By uploading a CSV file, users can trigger an autonomous pipeline that cleans the data, performs exploratory data analysis (EDA), identifies patterns through clustering, and generates interactive visualizations—all without manual intervention.

## 2. Core Objectives
- **Autonomous Operation:** Eliminate the need for step-by-step user prompting.
- **Robust Data Handling:** Automatically manage real-world data issues like missing values, outliers, and varying formats.
- **Explainability:** Provide real-time visibility into the AI's "thought process" and decision-making logic.
- **Interactive Visualization:** Render high-quality, interactive charts directly in the user interface.

## 3. Functional Requirements

### 3.1 Data Ingestion
- **File Upload:** Support for CSV file uploads via the Streamlit sidebar.
- **Local Storage:** Securely save uploaded files using absolute paths to prevent access errors.
- **Data Preview:** Immediately display the raw dataframe's head and shape upon upload.

### 3.2 Autonomous Workflow Agents
The system must employ a **Supervisor-Worker** architecture with the following specialized roles:

#### A. Supervisor Agent
- **Role:** Orchestrator.
- **Responsibility:** Review state, route tasks to specialists, and ensure the entire pipeline (Clean -> EDA -> Cluster -> Viz) is completed before finishing.
- **Logic:** Must enforce a strict "Definition of Done" to prevent premature termination.

#### B. Cleaning & EDA Agent
- **Data Cleaning:**
  - Detect and impute null values (Median/Mode).
  - Identify and remove outliers using the IQR method (with safety guardrails to preserve data).
  - Handle valid timestamp conversions.
  - **Feature:** Must support specific column dropping based on EDA feedback.
- **Exploratory Data Analysis (EDA):**
  - Calculate correlation matrices.
  - Detect multicollinearity (>0.85).
  - Suggest specific columns to drop to improve model quality.
  - Generate signals for UI-rendered heatmaps and histograms.

#### C. Clustering Agent
- **Preprocessing:** Auto-detect numerical/categorical columns, scale data (`StandardScaler`), and encode categories (`OneHotEncoder`).
- **Algorithm:** Perform K-Means clustering (default k=3).
- **Dimensionality Reduction:** Apply PCA to reduce features to 2 components (`PCA1`, `PCA2`) for 2D visualization.
- **Output:** Save a new CSV with `Cluster` labels and PCA coordinates.

#### D. Visualization Agent
- **Charting:** specific logic to check for clustering results and generate a 2D Scatter Plot.
- **Rendering:** Pass signals to the UI to render Plotly charts.

### 3.3 User Interface (Streamlit)
- **Real-Time Trace:** A dedicated expandable section showing the live execution log of the Graph (Agent decisions, Tool calls, Arguments).
- **In-Chat Reports:** display concise Markdown summaries of each agent's findings in the main chat stream.
- **Dynamic Tabs:** Automatically render EDA plots (Correlation Heatmap, Histograms) when available.
- **File Download:** Provide a button to download the final processed dataset (`_clustered.csv`) upon completion.

## 4. Non-Functional Requirements
- **Reliability:**
  - **Recursion Handling:** System must support high recursion limits (100+) for complex multi-step workflows.
  - **Path Safety:** All file operations must use absolute paths.
  - **Data Safety:** Cleaning tools must never delete more than 90% of the dataset.
- **Performance:** UI updates must be streamed to provide a "live" feel.
- **Maintainability:** Modular codebase separating App (`app.py`), Logic (`agents.py`), Tools (`tools.py`), and State (`state.py`).

## 5. Future Scope
- Support for other file formats (Excel, JSON).
- User-selectable clustering parameters (k-value).
- Advanced outlier detection algorithms (Isolation Forest).
- Persistent database storage for user sessions.
