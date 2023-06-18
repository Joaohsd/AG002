import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Preprocessing:
    @staticmethod
    def correlationMatrix(data:pd.DataFrame) -> pd.DataFrame:
        correlation_matrix = data.corr()
        # Heat map for correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig('src/preprocessing/images/Correlation_matrix.png', dpi=400)
        return correlation_matrix

    @staticmethod
    def covariationMatrix(data:pd.DataFrame) -> pd.DataFrame:
        covariation_matrix = data.cov()
        # Heat map for covariation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(covariation_matrix, annot=True, cmap='coolwarm')
        plt.title('Covariation Matrix')
        plt.savefig('src/preprocessing/images/Covariation_matrix.png', dpi=400)
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

        


