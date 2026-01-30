# Diabetes Dataset Analysis

This notebook performs an exploratory data analysis (EDA) on a Diabetes dataset, focusing on identifying and categorizing various data quality issues. The goal is to prepare the dataset for further analysis and model building by understanding its characteristics and potential problems.

## Table of Contents
1.  [Dataset Information](#Dataset-Information)
2.  [Descriptive Statistics](#Descriptive-Statistics)
3.  [Identify and Categorize Data Quality Issues](#Task:-Identify-and-Categorize-Data-Quality-Issues)
    *   [Missing Data](#1.-Missing-Data)
    *   [Duplicate Records](#2.-Duplicate-Records)
    *   [Inconsistent Formats](#3.-Inconsistent-Formats)
    *   [Invalid Values](#4.-Invalid-Values)
    *   [Outliers](#5.-Outliers)
    *   [Data Type Issues](#6.-Data-Type-Issues)

## Dataset Information
This section loads the dataset from a URL and provides a brief overview of its structure, including column names, non-null counts, and data types using `df.info()`. It also displays the first few rows of the dataset using `df.head()`.

## Descriptive Statistics
This section presents summary statistics for the numerical columns in the dataset using `df.describe()`, offering insights into the central tendency, dispersion, and shape of the data's distribution.

## Identify and Categorize Data Quality Issues
This is the core section of the notebook, where a systematic approach is taken to identify common data quality issues:

### 1. Missing Data
Identifies and visualizes missing values using `df.isnull().sum()` and a heatmap for a quick visual assessment.

### 2. Duplicate Records
Checks for and quantifies any duplicate rows in the dataset using `df.duplicated().sum()`.

### 3. Inconsistent Formats
Examines unique values in columns to detect potential inconsistencies, particularly relevant for categorical or string-based data. (In this dataset, columns are primarily numeric, so the focus is on confirming numeric consistency).

### 4. Invalid Values
Looks for values that are logically incorrect, such as negative values where they are not expected (e.g., Blood Pressure, Glucose) or zero values that represent missing data rather than actual measurements.

### 5. Outliers
Detects extreme values (outliers) in numerical columns using box plots and the Interquartile Range (IQR) method. This helps in understanding the spread and potential anomalies in the data.

### 6. Data Type Issues
Verifies the data types of each column using `df.dtypes` and notes any potential issues where the inferred data type might not be appropriate for the data it holds (e.g., float due to NaNs when it should be int).
