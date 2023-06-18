import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler

class Preprocessing:
    @staticmethod
    def correlationMatrix(data:pd.DataFrame) -> pd.DataFrame:
        correlation_matrix = data.corr()
        # Heat map for correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig('src/preprocessing/images/Correlation_matrix.png', dpi=400)
        plt.close()
        return correlation_matrix

    @staticmethod
    def covariationMatrix(data:pd.DataFrame) -> pd.DataFrame:
        covariation_matrix = data.cov()
        # Heat map for covariation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(covariation_matrix, annot=True, cmap='coolwarm')
        plt.title('Covariation Matrix')
        plt.savefig('src/preprocessing/images/Covariation_matrix.png', dpi=400)
        plt.close()
        return covariation_matrix
    
    @staticmethod
    def verifyDatasetBalance(data:pd.DataFrame) -> None:
        # Countign values
        value_counts = data.value_counts()
        # Pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(value_counts, labels=value_counts.index, autopct='%.2f')
        plt.title('Classes Percentage')
        plt.savefig('src/preprocessing/images/PieChart_class.png', dpi=400)
        plt.close()

    @staticmethod
    def resampleDataset(X:np.ndarray, y:np.ndarray):
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        # Countign values
        value_counts = pd.Series(y_resampled).value_counts()
        # Pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(value_counts, labels=value_counts.index, autopct='%.2f')
        plt.title('Classes Percentage')
        plt.savefig('src/preprocessing/images/PieChart_class_new.png', dpi=400)
        plt.close()
        return X_resampled, y_resampled

        


