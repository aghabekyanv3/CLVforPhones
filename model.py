import pandas as pd
import numpy as np
import logging
from logger.logger import CustomFormatter
import re
import os
import string
from datetime import datetime
from lifetimes import ParetoNBDFitter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

dtype_sorted = {
    #'Unnamed: 0': 'int',
    'IMEI': 'int',
    'Total_Duration': 'int',
    'MaxDuration': 'int',
    'SameDeviceQty': 'int',
    'ID': 'float',
    'Purchasing_Behaviour': 'float',
    'CAMERA_MATRIX': 'float',
    'FRONTCAM_MATRIX': 'float',
    'PIXEL_DENSITY': 'float',
    'OS': 'str',
    'PhoneType': 'str',
    'BRAND': 'str',
    'MODEL': 'str',
    # 'RELEASE_DATE': sqlite3.Date,  # Use TEXT for dates
    'SUPPORTS_VOLTE': 'bool',
    'SUPPORTS_VOWIFI': 'bool',
    'SUPPORTS_NFC': 'bool',
    'SUPPORTS_HTML5': 'str',
    'BATTERY_CAPACITY': 'str',
    'DATA_ONLY': 'bool',
    'SCREEN_RESOLUTION': 'str',
    'SUPPORTS_ESIM': 'str',
    # |'Min_END_DATE': sqlite3.Date,  # Use TEXT for dates
    # 'Max_END_DATE': sqlite3.Date,  # Use TEXT for dates
    # 'START_DATE': sqlite3.Date    # Use TEXT for dates
}

countries_list = ["Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia",
    "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin",
    "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi",
    "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia",
    "Comoros", "Congo, Democratic Republic of the", "Congo, Republic of the", "Costa Rica", "Cote d'Ivoire", "Croatia",
    "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "Ecuador", "Egypt",
    "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon",
    "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti",
    "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan",
    "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Korea, North", "Korea, South", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos",
    "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi",
    "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia",
    "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal",
    "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau",
    "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania",
    "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino",
    "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia",
    "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname",
    "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
    "United Kingdom", "United States of America", "USA" "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela",
    "Vietnam", "Yemen", "Zambia", "Zimbabwe", "Dual", "Duos", "mini", "Global", "Mini", "DUOS", "dual"
]
cat_cols=['PhoneType','MODEL','BRAND','OS']

# Set up the logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the desired logging level

# Create a console handler and set the level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the formatter to the handler
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)

df = pd.read_csv('df1.csv')

class DataTransform:
    def __init__(self, df, country_names):
        self.df = df
        self.country_names = country_names
        self.current_date = datetime.now()

    def OS_transformation(self, df):
        logger.info("Performing OS transformation...")
        """
        Perform OS transformation on the given DataFrame.

        Args:
            df (DataFrame): The input DataFrame containing 'OS', 'PhoneType', and 'BRAND' columns.

        Returns:
            Series: A Series containing the value counts of the transformed OS values.
        """
        replacement_dict = {
            'Apple': 'iOS',
            'Apple iOS': 'iOS',
            'Android5.1': 'Android'
        }

        df['OS'] = df['OS'].replace(replacement_dict)
        condition_huawei_samsung = (df['OS'].isin(['NONE', 'Unknown', np.nan])) & \
                                (df['BRAND'].isin(['Huawei', 'Samsung'])) & \
                                (df['PhoneType'] == 'Smart')
        df.loc[condition_huawei_samsung, 'OS'] = 'Android'
        
        condition_apple = df['BRAND'] == 'Apple'
        df.loc[condition_apple, 'OS'] = 'iOS'

        condition_other = (~df['OS'].isin(['iOS', 'Android']))

        df.loc[condition_other, 'OS'] = 'other'

        os_counts = df['OS'].value_counts()

        logger.info(f"OS transformation completed. {os_counts}")
        df.to_csv('df1.csv', index=False)

    def remove_punctuation_inplace(self, df, column_name:str):
        """
        Remove punctuation from the values in a specified column of a DataFrame (inplace).

        Args:
            df (pandas.DataFrame): The DataFrame containing the column.
            column_name (str): The name of the column to remove punctuation from.
        """
        logger.info("Removing punctuation")
        df[column_name] = df[column_name].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))
        logger.info("Punctuation removed")
        df.to_csv('df1.csv', index=False)

    def calculate_rfm_scores(self,df):
        """Calculate RFM scores for each customer in the DataFrame"""
        logging.info("Calculating RFM scores for each customer ID")
        df['START_DATE'] = pd.to_datetime(df['START_DATE'])
        df['Max_END_DATE'] = pd.to_datetime(df['Max_END_DATE'])  # Convert Max_END_DATE to datetime format
        
        # Calculate the recency as the minimum of all recencies for each ID
        df['recency'] = (df.groupby('ID')['Max_END_DATE'].transform('max') - df['START_DATE']).dt.days

        # Convert RELEASE_DATE and START_DATE to datetime
        df['RELEASE_DATE'] = pd.to_datetime(df['RELEASE_DATE'])
        
        # Calculate the Purchasing_Behaviour as the number of days between release_date and start_date
        df['Purchasing_Behaviour'] = (df['START_DATE'] - df['RELEASE_DATE']).dt.days
        
        monetary_bins = [0, 180, 360, 540, 720, float('inf')]  # Adjust the days ranges accordingly
        monetary_labels = [5000, 4000, 3000, 2000, 1000]  # Reverse the order

        df['frequency'] = df.groupby('ID')['ID'].transform('count')
        df['monetary_score'] = pd.cut(df['Purchasing_Behaviour'], bins=monetary_bins, labels=monetary_labels)
        
        # Calculate total duration for each customer ID
        total_duration = (df.groupby('ID')['Max_END_DATE'].transform('max') - df.groupby('ID')['START_DATE'].transform('min')).dt.days
        df['Total_Duration'] = total_duration
        
        # Calculate monetary score aggregated by ID
        df['monetary_score_aggregated'] = df.groupby('ID')['monetary_score'].transform('max')
        # Calculate the average transaction value using monetary score aggregated by ID
        self.df['average_transaction_value'] = self.df.groupby('ID')['monetary_score_aggregated'].first()
        self.df['average_transaction_value'] = self.df['average_transaction_value'].astype(float)
        self.df['average_transaction_value'] = self.df['average_transaction_value'].fillna(0)  # Replace missing values with 0 or another appropriate value.
        logger.info(f"RFM scores calculated: {df[['ID', 'recency', 'frequency','monetary_score']].head(5)}")
        
        return 

    def display_rfm_scores(self):
        """Display the calculated RFM scores."""
        logger.info("Displaying RFM scores...")
        print(self.df[['ID', 'recency', 'frequency', 'monetary_score']])
        logger.info("RFM scores displayed.")

    def check_country_name(self, df, country_names: list):
        """
        Check if any country name is present in the 'MODEL' column of the DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to check.
            country_names (list): List of country names to check.

        Returns:
            None
        """
        # Combine country names with proper escaping for use in regex pattern
        escaped_country_names = [re.escape(country) for country in country_names]
        countries_found = df['MODEL'].str.contains('|'.join(escaped_country_names), case=False, na=False)

        if any(countries_found):
            logger.info("Country names found in the 'MODEL' column.")
            # Print the rows where country names were found
            for index, row in df[countries_found].iterrows():
                print(f"Row {index}: {row['MODEL']}")
        else:
            logger.info("No country names found in the 'MODEL' column.")

    def remove_country_names(self, df, countries: list):
        """
        Remove country names from the 'MODEL' column of a DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to modify.
            country_names (list): List of country names to be removed.

        Returns:
            None. Modifies the 'MODEL' column in-place.
        """
        logger.info("Removing country names from 'MODEL' column...")

        pattern = '|'.join(countries)

        # Replace everything including and after country name with an empty string
        df['MODEL'] = df['MODEL'].str.replace(fr'({pattern}).*', '', regex=True)

        self.check_country_name(df,countries)
        logger.info("Country names removed from 'MODEL' column." )

        df.to_csv('df1.csv', index=False)

    def label_encode_columns(self, df, cat_columns):
        logger.info("Label encoding for categorical variables in progress.")
        """
        Label encodes specified columns in a DataFrame.
        
        Parameters:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            columns (list): List of column names to be label encoded.
        
        Returns:
            pd.DataFrame: DataFrame with added encoded columns.
            dict: Dictionary of label mappings for each encoded column.
        """
        label_encoder = LabelEncoder()
        label_mapping = {}        
        for col in cat_columns:
            encoded_col_name = f"{col}_Encoded"
            df[encoded_col_name] = label_encoder.fit_transform(df[col])
            label_mapping[col] = dict(zip(df[col], df[encoded_col_name]))
        
        logger.info("Label encoding for categorical variables completed.")
        return label_mapping
    
class SummaryStatisticsCalculator:
    def __init__(self, df):
        self.df = df

    def calculate_summary_statistics(self, df, cat_cols):
        """
        Calculate summary statistics for numerical columns in the DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing numerical columns.

        Returns:
            dict: Dictionary containing summary statistics for each numerical column.
        """
        summary_stats = {}
        logger.info("Printing summary statistics..")
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in cat_cols:
                stats = {
                    'Mean': df[col].mean(),
                    'Median': df[col].median(),
                    'StdDev': df[col].std(),
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Count': df[col].count(),
                    '25th Percentile': df[col].quantile(0.25),
                    '75th Percentile': df[col].quantile(0.75),
                    'Skewness': df[col].skew(),
                    'Kurtosis': df[col].kurtosis(),
                    'Coefficient of Variation': (df[col].std() / df[col].mean()) * 100,
                    'Range': df[col].max() - df[col].min(),
                    'Interquartile Range': df[col].quantile(0.75) - df[col].quantile(0.25),
                    'Variance': df[col].var()
                }
                summary_stats[col] = stats
        logger.info("Summary statistics printed. Returning summary stats dataframe.")
        summary_df = pd.DataFrame(summary_stats)
        return summary_df
        
class DataVisualizer:
    """Visualize data using various plotting methods."""

    def __init__(self, df: pd.DataFrame):
        """Initialize the DataVisualizer class.

        Args:
            df (pd.DataFrame): The DataFrame containing data to visualize.
        """
        self.df = df
    
    def df_head(self, n_rows: int):
        print(self.df.head(n_rows)) 

    def visualize_pairwise_scatter(self):
        """Visualize pairwise scatter plots for numerical columns."""
        logger.info("Visualising pairwise scatter plot.")
        sns.pairplot(self.df.select_dtypes(include=[np.number]))
        plt.title("Pairwise Scatter Plot Matrix")
        plt.show()
    
    def visualize_histograms(self):
        """Visualize histograms for numerical columns."""
        logger.info("Visualising histograms for numerical columns.")
        self.df.select_dtypes(include=[np.number]).hist(bins=20, figsize=(12, 10))
        plt.suptitle("Histograms of Numerical Columns")
        plt.show()
    
    def visualize_box_plots(self):
        """Visualize box plots for numerical columns."""
        logger.info("Visualising boxplots for numerica columns.")
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.df.select_dtypes(include=[np.number]))
        plt.title("Box Plots of Numerical Columns")
        plt.xticks(rotation=45)
        plt.show()
    
    def visualize_correlation_heatmap(self):
        """Visualize a correlation heatmap for numerical columns."""
        logger.info("Visualising correlation heatmap.")
        correlation_matrix = self.df.select_dtypes(include=[np.number]).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap of Numerical Columns")
        plt.show()
    
    def visualize_categorical_bar_plot(self, cat_columns):
        """Visualize bar plots for categorical columns.

        Args:
            cat_columns (list): List of categorical column names to visualize.
        """
        logger.info(f"Visualising bar plots for categorical columns.{cat_cols}")
        num_cols = len(cat_columns)
        fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(num_cols * 6, 6))

        for idx, col in enumerate(cat_columns):
            sns.countplot(data=self.df, x=col, ax=axes[idx])
            axes[idx].set_title(f"Count of {col}")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Count")
        
        plt.tight_layout()
        plt.show()


class Modeling:
    """Perform modeling tasks such as segmentation and Pareto/NBD modeling."""

    def __init__(self, df: pd.DataFrame):
        """Initialize the Modeling class.

        Args:
            df (pd.DataFrame): The DataFrame containing data for modeling.
        """
        self.df = df

    def min_max_normalize(self, data):
        """
        Performing min-max normalisation of data. 

        Args: 
            data: The data to normalise.
        """
        logger.info("Carrying out min-max normalisation.")
        if data is None:
            logger.error("data not provided")
        min_val = data.min()
        max_val = data.max()
        normalized_data = (data - min_val) / (max_val - min_val)
        logger.info(f"min-max normalisation carried out.{normalized_data}")
        return normalized_data

    def extract_sample_data(self, sample_size: int) -> pd.DataFrame:
        """
        Extract a sample of data from the DataFrame.

        Args:
            sample_size (int): The size of the sample to be extracted.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted sample data.
        """
        logger.info(f"Extracting sample data of size {sample_size}")
        if sample_size > len(self.df):
           
            sample_data = self.df.copy()  # Use the entire DataFrame
        else:
            sample_data = self.df.sample(n=sample_size, random_state=42)
        logger.info("sample data extracted")
        return sample_data

    
    def classify_purchasing_behavior(self, purchasing_behavior):
        purchasing_behavior = float(purchasing_behavior) 

        if purchasing_behavior <= 180:
            return 'Early Adopters'
        elif 180 <= purchasing_behavior <= 360:
            return 'Regular Users'
        elif 360 <= purchasing_behavior <= 540:
            return 'Late Adopters'
        else:
            return 'Very Late Adopters'

    def segment_based_on_purchasing_behavior(self):
        logger.info("Segmenting customers based on purchasing behaviour.")
        self.df['Purchasing_Segment'] = self.df['Purchasing_Behaviour'].apply(self.classify_purchasing_behavior)
        
    def train_pareto_nbd_model(self):
        """Train a Pareto/NBD model for customer lifetime value prediction."""
       
        # Prepare the data
        summary_data = self.df.groupby('ID').agg(
            frequency=('frequency', 'max'),
            recency=('recency', 'max'),
            Total_Duration=('Total_Duration', 'max')
        )
        
        # Train the Pareto/NBD model
        logger.info("training Pareto/NBD model")
        pareto_nbd_model = ParetoNBDFitter()
        pareto_nbd_model.fit(summary_data['frequency'], summary_data['recency'], summary_data['Total_Duration'])
       
        logger.info("Pareto/NBD model trained")
        self.pareto_nbd_model = pareto_nbd_model

        return pareto_nbd_model


    def calculate_conditional_expected_transactions(self, time_frame):
        """Calculate the conditional expected number of transactions for a given time frame using the Pareto/NBD model.

        Args:
            time_frame (int): The time frame for which to calculate the expected transactions.

        Returns:
            pd.Series: A Series containing the calculated expected transactions for each customer.
        """
       
        logger.info(f"calculating conditionall expected transactionf for time frame {time_frame}")
        # Prepare the data
        summary_data = self.df.groupby('ID').agg(
            frequency=('frequency', 'max'),
            recency=('recency', 'max'),
            Total_Duration=('Total_Duration', 'max')
        )

        # Calculate the conditional expected transactions
        expected_transactions = self.pareto_nbd_model.conditional_expected_number_of_purchases_up_to_time(
            time_frame, summary_data['frequency'], summary_data['recency'], summary_data['Total_Duration']
        )

        
        expected_transactions = expected_transactions.sort_values(ascending=False)
        minmax_expected_transactions = self.min_max_normalize(expected_transactions)
        logger.info("visualising expected transactions")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=expected_transactions.value_counts().index, y=expected_transactions.value_counts().values)
        plt.title("Conditional Expected Transactions Count Distribution (Without Min-Max Normalization)")
        plt.xlabel("Expected Transactions")
        plt.ylabel("Count of Customer IDs")
        plt.xticks(rotation=45)
        plt.show()
        logger.info("visualising min-max adjusted expected transactions")
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=minmax_expected_transactions.value_counts().index, y=minmax_expected_transactions.value_counts().values)
        plt.title("Conditional Expected Transactions Count Distribution (With Min-Max Normalization)")
        plt.xlabel("Expected Transactions (Min-Max Normalized)")
        plt.ylabel("Count of Customer IDs")
        plt.xticks(rotation=45)
        plt.show()
        return expected_transactions, minmax_expected_transactions


    def predict_frequency(self, time_frame_months):
        """
        Predict the frequency scores for customers over a specified time frame.

        Parameters:
        - time_frame_months (int): The time frame in months.

        Returns:
        - pd.Series: Series containing customer IDs and their predicted frequency scores.
        """
        if self.pareto_nbd_model is None:
            logger.error("Pareto/NBD model has not been trained yet.")

        logger.info("predicting frequency scores, normalising and sorting in descending order")
        predicted_freq = self.pareto_nbd_model.predict(
            time_frame_months, self.df['frequency'], self.df['recency'], self.df['Total_Duration']
        )

        # Sort the normalized frequency scores in descending order
        predicted_freq = predicted_freq.sort_values(ascending=False)
        minmax_predicted_freq = self.min_max_normalize(predicted_freq)
        logger.info("visualising predicted frequency and min-max adjusted predicted frequency score.")
        # Plotting
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=predicted_freq.value_counts().index, y=predicted_freq.value_counts().values)
        plt.title("Predicted Frequency Scores Count Distribution (Without Min-Max Normalization)")
        plt.xlabel("Predicted Frequency Scores")
        plt.ylabel("Count of Customer IDs")
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=minmax_predicted_freq.value_counts().index, y=minmax_predicted_freq.value_counts().values)
        plt.title("Predicted Frequency Scores Count Distribution (With Min-Max Normalization)")
        plt.xlabel("Predicted Frequency Scores (Min-Max Normalized)")
        plt.ylabel("Count of Customer IDs")
        plt.xticks(rotation=45)
        plt.show()
        
        return predicted_freq, minmax_predicted_freq
            
    def interpret_model_parameters(self):
        """Interpret the parameters of the Pareto/NBD model."""
        
        logging.info("Interpreting Pareto/NBD Model Parameters:")
        print("Lambda:", self.pareto_nbd_model.params_['alpha'] * self.pareto_nbd_model.params_['r'])
        print("Mu:", self.pareto_nbd_model.params_['beta'] * self.pareto_nbd_model.params_['s'])
       
        sorted_parameters = self.pareto_nbd_model.params_.sort_values(ascending=False)

        print("Pareto/NBD Model Parameters (sorted):")
        for param, value in sorted_parameters.items():
            print(f"{param.capitalize()}: {value}")

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_parameters.index, y=sorted_parameters.values)
        plt.title("Pareto/NBD Model Parameters")
        plt.xlabel("Parameters")
        plt.ylabel("Values")
        plt.xticks(rotation=45)
        plt.show()

    def calculate_customer_lifetime_value(self, time_horizon, discount_rate=0.1):
        """
        Calculate the Customer Lifetime Value (CLV) using the Pareto/NBD model.

        Parameters:
        - time_horizon (int): The time horizon for which to calculate CLV.
        - discount_rate (float): The discount rate used to discount future values (default is 0.1).

        Returns:
        - pd.Series: Series containing customer IDs and their calculated CLVs.
        """
        if self.pareto_nbd_model is None:
            logger.error("Pareto/NBD model has not been trained yet.")

        # Calculate the conditional expected transactions
        expected_transactions = self.pareto_nbd_model.conditional_expected_number_of_purchases_up_to_time(
            time_horizon, self.df['frequency'], self.df['recency'], self.df['Total_Duration']
        )

        # Calculate the CLV using the Pareto/NBD formula
        clv = (expected_transactions * self.df['average_transaction_value']) / (1 + discount_rate)

        # Normalize the CLV using Min-Max normalization
        normalized_clv = self.min_max_normalize(clv)

        # Sort the normalized CLV in descending order
        sorted_normalized_clv = normalized_clv.sort_values(ascending=False)
        logger.info(f"CLV calculated: {sorted_normalized_clv}")
         

    def run_all_models(self):
        """Run all modeling tasks."""
        
        self.segment_based_on_purchasing_behavior()
        self.train_pareto_nbd_model()
        self.calculate_conditional_expected_transactions(3)
        self.predict_frequency(3)
        self.interpret_model_parameters()
        self.calculate_customer_lifetime_value(3)
if __name__ == "__main__":
    # Load data from SQLite database
    cnxn = sqlite3.connect('clv.db')
    query = "SELECT * FROM transactions LIMIT 100"
    df = pd.read_sql(query, cnxn)
    cnxn.close()

    # Initialize classes and perform tasks
    transformer = DataTransform(df, country_names=countries_list)
    transformer.OS_transformation(df)
    transformer.remove_punctuation_inplace(df, column_name='MODEL')
    transformer.calculate_rfm_scores(df)
    transformer.display_rfm_scores()
    transformer.check_country_name(df, country_names=countries_list)

    calculator = SummaryStatisticsCalculator(df)
    summary_stats = calculator.calculate_summary_statistics(df, cat_cols=cat_cols)

    visualizer = DataVisualizer(df)
    visualizer.visualize_pairwise_scatter()
    visualizer.visualize_histograms()
    visualizer.visualize_box_plots()
    visualizer.visualize_correlation_heatmap()
    visualizer.visualize_categorical_bar_plot(cat_columns=cat_cols)

    modeler = Modeling(df)
    modeler.run_all_models()
