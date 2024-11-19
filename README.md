![](https://github.com/Ambigapathi-V/NLP/blob/main/img/download%20(3).jpeg)


#  Text Prediction Hateful and Abusive



![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/pragyy/datascience-readme-template?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/pragyy/datascience-readme-template)
![GitHub pull requests](https://img.shields.io/github/issues-pr/pragyy/datascience-readme-template)
![GitHub](https://img.shields.io/github/license/pragyy/datascience-readme-template)
![contributors](https://img.shields.io/github/contributors/pragyy/datascience-readme-template) 
![codesize](https://img.shields.io/github/languages/code-size/pragyy/datascience-readme-template) 



## Project Overview

This project is an end-to-end Natural Language Processing (NLP) pipeline for predicting hateful and Abusive news. It aims to streamline data ingestion, preprocessing, and model training to extract meaningful insights from textual data. The pipeline supports various NLP tasks, including sentiment analysis, topic modeling, and text classification, making it a versatile tool for researchers and developers alike.

The project incorporates modern techniques in NLP, leveraging libraries such as NLTK, SpaCy, and TensorFlow to deliver robust performance. By providing a comprehensive framework, this project enables users to easily adapt and extend the pipeline for their specific needs, facilitating efficient experimentation and deployment in real-world applications.

## Features

- **Data Ingestion**: Seamlessly fetch and load text data from various sources, including CSV files, APIs, and web scraping.

- **Text Preprocessing**: Includes essential preprocessing steps such as tokenization, stemming, lemmatization, and removal of stop words to prepare data for analysis.

- **Sentiment Analysis**: Implement sentiment analysis to gauge the emotional tone of the text, providing insights into public opinion and user feedback.

- **Topic Modeling**: Utilize algorithms like LDA (Latent Dirichlet Allocation) to discover underlying topics within large text corpora.

- **Text Classification**: Train and evaluate models for categorizing text into predefined classes using machine learning techniques.

- **Visualization Tools**: Generate visualizations for data exploration and model evaluation, including word clouds, bar charts, and confusion matrices.

- **Model Evaluation**: Comprehensive metrics and evaluation techniques to assess model performance, including accuracy, precision, recall, and F1 score.

- **User-Friendly Interface**: An interactive Jupyter Notebook interface for easy experimentation and visualization of results.

- **Cross-Platform Compatibility**: The pipeline can be run on various platforms, ensuring accessibility for different users.

- **Documentation and Tutorials**: Detailed documentation and tutorials to guide users through the setup and usage of the pipeline, making it beginner-friendly.


## Demo

You can access the deployed application here: [Hateful and Abusive Content Prediction App](https://hateful-abusive-prediction.streamlit.app/).  


## Screenshots

![App Screenshot](https://github.com/Ambigapathi-V/NLP/blob/main/img/image.png)



# Installation and Setup

**1.Clone the Repository:**

```bash
  git clone https://github.com/Ambigapathi-V/NLP
  cd Credit-Risk-Model
```
**2.Set Up a Virtual Environment:**

   ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
**3. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**4. Run the main.py:**

```bash
python main.py
```

**4. Run the Streamlit**

```bash
streamlit run app.py
```

I like to structure it as below - 

## Codes and Resources Used

### Editor Used
- **Jupyter Notebook**: The primary environment for developing and running the NLP pipeline.

### Python Version
- **Python 3.8+**: Ensure you are using Python version 3.8 or higher for compatibility with the libraries used.

### Python Packages Used
- **General Purpose**: 
  - `requests`
  - `os`
  
- **Data Manipulation**: 
  - `pandas`
  - `numpy`
  
- **Data Visualization**: 
  - `matplotlib`
  - `seaborn`
  
- **Natural Language Processing**: 
  - `nltk`
  - `spacy`
  
- **Machine Learning**: 
  - `scikit-learn`
  - `tensorflow`
  
These packages are included in the `requirements.txt` file for easy installation.

## Data

The very crucial part of any data science project is dataset. Therefore list all the data sources used in the project, including links to the original data, descriptions of the data, and any pre-processing steps that were taken.

I structure this as follows - 

## Source Data

- **Kaggle Datasets**: This project utilizes datasets available on Kaggle. To access the datasets, you need a Kaggle account and the Kaggle API installed.

## Data Ingestion

### Overview
The data ingestion process involves loading datasets from various sources, including DagsHub S3 buckets, into the NLP pipeline. This section outlines how to upload data to DagsHub and subsequently ingest it into your project.

### Uploading Data to DagsHub S3 Bucket

1. **Create a DagsHub Account**: If you don't have one, sign up at [DagsHub](https://dagshub.com/).

2. **Create a New Repository**: Once logged in, create a new repository where you will store your data.

3. **Upload Data**:
   - Navigate to the "Data" tab of your repository.
   - Click on "Upload files" and select the datasets you want to upload.
   - Alternatively, you can use the DagsHub CLI to upload files directly to your repository:
     ```bash
     dagshub upload path/to/your/local/file.csv
     ```

### Ingesting Data from DagsHub S3 Bucket

Once your data is uploaded to the DagsHub S3 bucket, you can ingest it into your pipeline using the following steps:

#### Example Code for Loading Data from DagsHub
```python
import pandas as pd
import os

# Define the DagsHub S3 URL for the dataset
dags_hub_url = 'https://dagshub.com/username/repo-name/raw/main/path/to/your/dataset.csv'  # Update with your details

# Load the dataset into a DataFrame
data = pd.read_csv(dags_hub_url)

# Display the first few rows of the DataFrame
print(data.head())
```


## Data Preprocessing

Data preprocessing is a crucial step in preparing raw data for analysis and modeling. In this repository, I have completed the data preprocessing tasks to ensure the datasets are clean, consistent, and ready for further analysis.

### Steps Taken in Data Preprocessing

1. **Loading Data**:
   - The datasets were loaded from the DagsHub S3 bucket. Initial inspections were conducted to understand the structure and content of the data.

2. **Data Cleaning**:
   - **Handling Missing Values**: Missing values were identified and addressed either by removing affected rows or imputing values based on the context of the data.
   - **Removing Duplicates**: Duplicate entries were detected and eliminated to maintain data integrity.
   - **Correcting Data Types**: Data types were verified and corrected as necessary to ensure proper analysis, such as converting date strings into datetime objects.

3. **Text Preprocessing**:
   - **Lowercasing**: All text data was converted to lowercase to ensure uniformity.
   - **Removing Punctuation and Special Characters**: Non-alphanumeric characters were removed to clean the text.
   - **Tokenization**: Text data was split into individual words or tokens to facilitate analysis.
   - **Stop Word Removal**: Common stop words (e.g., "and", "the", "is") were removed to focus on more meaningful words.
   - **Stemming and Lemmatization**: Words were reduced to their base or root forms to standardize variations and improve consistency.

4. **Feature Engineering**:
   - New features were created based on existing data to enhance model performance. This could involve extracting sentiment scores, creating binary flags, or aggregating data for better insights.

5. **Data Validation**:
   - After preprocessing, the cleaned data was validated to ensure that all transformations were applied correctly. This included checking for any remaining missing values, verifying data types, and examining the distribution of key features to confirm they aligned with expectations.

### Summary
The preprocessing steps outlined above ensure that the data is clean, standardized, and ready for analysis. This process is vital for achieving accurate and meaningful results in subsequent modeling phases.

For further details or specific implementations, please refer to the code in the repository.
## Code Structure


This repository is organized to facilitate easy navigation and understanding of the project. Below is an overview of the directory structure and the purpose of each component.



```cmd
├── data/
│   ├── raw/                # Original, unprocessed data files
│   ├── processed/          # Cleaned and preprocessed data files
│   └── external/           # External datasets used in the project
│
├── notebooks/              # Jupyter notebooks for exploratory data analysis and visualization
│   └── EDA.ipynb           # Notebook containing exploratory data analysis
│
├── src/                    # Source code for the project
│   ├── components          # Reusable components for the project
│   ├── constants           # Common constants used in the project
│   ├── pipeline #          # Pipeline for executing tasks related to the project
│   ├── Configuration      # Script for model configuration
│   ├── entitys            # Data entities used in the project
│   ├── logging            # Script for logging information and errors
│   ├── expection            # Script for handling data expectations and assertions
│   ├── ml flow             # Script for tracking model performance and metrics
│   └── utils           # Script for model training and evaluation
│
├── requirements.txt        # List of dependencies required to run the project
├── README.md               # Project documentation and overview
└── main.py                 # Main entry point for running the project
```

## Results and Evaluation

This section summarizes the performance of the Long Short-Term Memory (LSTM) model used in this project. Detailed results and visualizations can be found on Dagshub.

### Model Performance

The LSTM model was evaluated using the following metrics:

- **Accuracy**: 87%
- **Precision**: 80%
- **Recall**: 85%
- **F1 Score**: 82%
- **ROC-AUC**: 90%

### Key Findings

- The model achieved an accuracy of X% on the test dataset.
- The confusion matrix indicated strong performance in [specific class] but challenges with [another class].
- Loss curves showed effective learning, with validation loss stabilizing after a certain epoch.

### Accessing Results

For detailed metrics and visualizations, visit the Dagshub repository: [Dagshub Repository](https://dagshub.com/Ambigapathi-V/NLP).

## Future Work

Outline potential future work that can be done to extend the project or improve its functionality. This will help others understand the scope of your project and identify areas where they can contribute.

## Deployment

To deploy this project run

```bash
  npm run deploy
```



## Acknowledgments

We would like to extend our sincere appreciation to the following individuals and organizations:

- **Mentors and Advisors**: Our heartfelt thanks to our mentors and advisors for their invaluable guidance and support throughout the duration of this project.
  
- **Open Source Community**: We acknowledge the contributions of the open-source community, particularly the developers of libraries such as TensorFlow, Keras, and NumPy, which were instrumental in our implementation.

- **Data Providers**: We express our gratitude to [source name or organization] for providing the datasets used in this analysis, which were essential for our research.

- **Colleagues and Peers**: We appreciate the constructive feedback and collaboration from our colleagues, which significantly enhanced the quality of our work.

- **Family and Friends**: Lastly, we would like to thank our family and friends for their unwavering support and encouragement during this endeavor.

Your contributions and support have been pivotal in the successful completion of this project.

## License

Specify the license under which your code is released. Moreover, provide the licenses associated with the dataset you are using. This is important for others to know if they want to use or contribute to your project. 

For this github repository, the License used is [MIT License](https://opensource.org/license/mit/).
