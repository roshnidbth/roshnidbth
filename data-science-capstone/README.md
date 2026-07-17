# IBM Data Science Capstone - SpaceX Falcon 9

This repository contains the completed notebook and Python files for the IBM Applied Data Science Capstone project. The project predicts whether a SpaceX Falcon 9 first stage will land successfully using public launch data.

## Project workflow

1. Collect launch records from the SpaceX REST API and public launch tables.
2. Clean and transform the launch-level dataset.
3. Perform exploratory analysis with Pandas, visualizations, and SQL.
4. Analyze launch-site geography with Folium.
5. Build an interactive Plotly Dash dashboard.
6. Train and evaluate classification models.
7. Summarize conclusions, limitations, and operational recommendations.

## Included files

- `notebooks/SpaceX_Capstone_Complete.ipynb` - completed end-to-end notebook
- `spacex_data_collection.py` - API data collection and flattening
- `spacex_dashboard.py` - Plotly Dash application
- `spacex_modeling.py` - preprocessing, model tuning, and evaluation
- `requirements.txt` - Python dependencies

## Key reported results

- 90 historical Falcon 9 launches analyzed
- 55 launches from CCAFS SLC-40
- 27 launches to GTO
- 41 successful drone-ship outcomes (`True ASDS`)
- 80 encoded model features
- 72 training records and 18 test records
- Best validation SVM kernel: sigmoid
- Held-out accuracy: 83.33%

The dataset is historical and the model is intended as an educational baseline rather than a production launch-risk system.
