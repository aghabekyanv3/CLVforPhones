# SQL ETL Operations README

This repository contains a Python script for performing ETL (Extract, Transform, Load) operations with SQLite databases. The script allows you to create and manage SQLite tables, insert, delete, and load data, and perform various database operations. Below, you will find instructions on how to use this script and explanations of its functionalities.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dependencies](#dependencies)
3. [Usage](#usage)
    - [Creating a Table](#creating-a-table)
    - [Inserting Data](#inserting-data)
    - [Loading Data](#loading-data)
    - [Truncating a Table](#truncating-a-table)
    - [Deleting a Table](#deleting-a-table)
    - [Extracting Data from SQLite to a DataFrame](#extracting-data-from-sqlite-to-a-dataframe)
    - [Inserting Data from CSV](#inserting-data-from-csv)
4. [Contributing](#contributing)
5. [License](#license)

## Getting Started

To use this script for ETL operations with SQLite databases, follow these steps:

1. Clone this repository to your local machine or download the script directly.
2. Make sure you have the required dependencies installed (see [Dependencies](#dependencies)).
3. Open the Python script in your preferred code editor.

## Dependencies

Before using the script, ensure you have the following dependencies installed:

- `sqlite3`: This module provides an interface to the SQLite database engine.
- `pandas`: Pandas is used for data manipulation and analysis.
- `numpy`: NumPy is used for numerical operations.
- `csv`: This module allows reading and writing CSV files.

You can install these dependencies using pip:

```bash
pip install pandas numpy
```

## Usage

The script `sqlfuncs.py` provides several functions for performing common ETL operations. Below are explanations and examples of how to use these functions:

### Creating a Table

To create an SQLite table, you can use the `create_sql_table` function. Provide the table name and a list of column names and their data types as parameters. If the table already exists, it will not be recreated.

Example:

```python
etl_instance = etl()
table_name = 'my_table'
columns = ['id INTEGER PRIMARY KEY', 'name TEXT', 'age INTEGER']
etl_instance.create_sql_table(table_name, columns)
```

### Inserting Data

You can insert data into an SQLite table using the `row_insert` function. Provide the data as a string, formatted for insertion. This function inserts a single row of data.

Example:

```python
etl_instance = etl()
row_data = "'John Doe', 30"
etl_instance.row_insert(row_data)
```

### Loading Data

To load data from an SQLite table based on an ID number, use the `load` function. Provide the ID number and the table name as parameters.

Example:

```python
etl_instance = etl()
id_number = 1
table_name = 'my_table'
etl_instance.load(id_number, table_name)
```

### Truncating a Table

To delete all rows from an SQLite table while preserving its structure, you can use the `truncate_table` function. Provide the table name as a parameter.

Example:

```python
etl_instance = etl()
table_name = 'my_table'
etl_instance.truncate_table(table_name)
```

### Deleting a Table

If you want to completely delete an SQLite table, use the `delete_table` function. Provide the table name as a parameter.

Example:

```python
etl_instance = etl()
table_name = 'my_table'
etl_instance.delete_table(table_name)
```

### Extracting Data from SQLite to a DataFrame

You can extract data from an SQLite table to a Pandas DataFrame using the `from_sql_to_local` function. Provide the chunk size, ID value for sorting, database name, and table name as parameters. This function retrieves data in chunks and returns a DataFrame.

Example:

```python
etl_instance = etl()
chunk_size = 1000
id_value = 'MSISDN, DATE_ID'
df = etl_instance.from_sql_to_local(chunk_size, id_value, dbname='clv.db', tablename='transactions')
```

### Inserting Data from CSV

To insert data from a CSV file into an SQLite table, use the `insert_data_from_csv` function. Provide the table name and the path to the CSV file as parameters.

Example:

```python
etl_instance = etl()
table_name = 'my_table'
csv_file_path = 'data.csv'
etl_instance.insert_data_from_csv(table_name, csv_file_path)
```

# Customer Lifetime Value (CLV) Modeling Readme

## Overview

This repository contains code for performing Customer Lifetime Value (CLV) modeling using Python and various data analysis libraries. CLV is a crucial metric for businesses, as it helps estimate the total revenue a company can expect from a customer during their entire relationship. The code in this repository includes data preprocessing, summary statistics calculation, data visualization, and CLV modeling using the Pareto/NBD model.

## Code Structure

The code is organized into several classes and functions to perform different tasks in the CLV modeling process. Here's an overview of the main components:

### 1. Data Transformation (`DataTransform` class)

- This class is responsible for cleaning and transforming the raw data.
- It handles operations like OS transformation, removing punctuation from columns, calculating RFM (Recency, Frequency, Monetary) scores, and more.

### 2. Summary Statistics Calculator (`SummaryStatisticsCalculator` class)

- This class calculates summary statistics for numerical columns in the dataset.
- It provides insights into the distribution and characteristics of the data.

### 3. Data Visualizer (`DataVisualizer` class)

- The Data Visualizer class helps in visualizing the data.
- It provides various plots and graphs such as scatter plots, histograms, box plots, and correlation heatmaps for both numerical and categorical columns.

### 4. Modeling (`Modeling` class)

- The Modeling class performs CLV modeling using the Pareto/NBD model.
- It includes methods for segmenting customers based on purchasing behavior, training the Pareto/NBD model, predicting customer frequencies, and calculating CLV.

## Running the Code

To run the code in this repository, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required Python packages. You can use a virtual environment to manage dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary data files or database connections set up as required by the code.

4. Open and run the Jupyter Notebook or Python script containing the code. Make sure to update any file paths or data sources as needed.

5. The code will perform data preprocessing, calculate summary statistics, create data visualizations, and execute CLV modeling tasks.

## Data Sources

The code assumes that you have access to a dataset or SQLite database containing customer transaction data. Ensure that the data is structured with the required columns, such as customer IDs, transaction dates, and transaction values.

## Output

The code generates various output, including summary statistics, data visualizations, and CLV calculations. You can customize the output format and save the results as needed.

## Acknowledgments

The CLV modeling code in this repository is a general framework and may require customization to fit specific business needs and data structures. Consider consulting with data analysts and domain experts to adapt the code for your use case.
