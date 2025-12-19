# LangGraph Data Cleaning & Segmentation Pipeline

This repository implements an intelligent, agentic workflow using **LangGraph** to automate the process of data cleaning and customer/data segmentation. By leveraging a state-graph architecture, the system can handle complex, non-linear tasks such as outlier detection, missing value imputation, and automated clustering with LLM-driven decision-making.

## Overview

Traditional data pipelines are often rigid. This project uses **LangGraph** to create a cyclic, stateful workflow where an AI agent can:

1. **Analyze** the raw dataset schema and quality.
2. **Clean** the data dynamically (handling nulls, encoding, and scaling).
3. **Segment** the data using machine learning (e.g., K-Means or LLM-based categorization).
4. **Reflect** on the results and iterate if the quality doesn't meet specific thresholds.

## Architecture

The workflow is designed as a directed graph where each node represents a specific functional step:

* **Data Loader Node:** Imports CSV/Excel files and initializes the graph state.
* **Cleaning Agent:** Detects anomalies, handles missing values, and prepares features.
* **Segmentation Node:** Applies clustering algorithms or classification logic to group data points.
* **Evaluation Node:** Uses an LLM to interpret the clusters/segments and verify if the cleaning was successful.
* **Router:** Directs the flow based on data quality (e.g., "Back to cleaning" or "Proceed to Final Report").

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/premkumarkora/LangGraph_Cleaning_segmentation.git
cd LangGraph_Cleaning_segmentation

```


2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



## Configuration

Create a `.env` file in the root directory and add your API keys (if using LLMs like OpenAI or Anthropic for analysis):

```env
OPENAI_API_KEY=your_api_key_here
# Optional: LANGCHAIN_TRACING_V2=true for debugging

```

## 💻 Usage

Run the main pipeline script to process your data:

```python
from langgraph_pipeline import run_pipeline

# Provide path to your dataset
results = run_pipeline("data/raw_data.csv")

# Access cleaned data and segmentation labels
print(results["cleaned_data"])
print(results["segments"])

```

## Project Structure

```text
├── data/                   # Sample datasets
├── agents/                 # Logic for cleaning and segmentation agents
│   ├── cleaner.py          # Data preprocessing functions
│   └── segmenter.py        # ML clustering/segmentation logic
├── graph.py                # LangGraph StateGraph definition
├── main.py                 # Entry point for the application
├── requirements.txt        # Dependencies (pandas, scikit-learn, langgraph, etc.)
└── README.md

```

## Key Features

* **State Persistence:** The graph maintains the state of the data across multiple cleaning iterations.
* **Conditional Logic:** Dynamically decides whether to scale features or encode categorical variables based on data profiling.
* **Human-in-the-Loop:** (Optional) Ability to pause the graph for human approval before final segmentation.
* **Extensible:** Easily add new "nodes" for specific domain-based cleaning rules.

## Contributing

Contributions are welcome! If you have suggestions for better segmentation techniques or cleaning heuristics:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Author:** [Prem Kumar Kora](https://www.google.com/search?q=https://github.com/premkumarkora)
