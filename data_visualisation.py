import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    df = pd.read_csv("Human Action Recognition/Training_set.csv")
    
    label_counts = df.label.value_counts()
    label_counts_df = pd.DataFrame(label_counts.reset_index())
    label_counts_df.columns = ['Activity', 'Number of Images']
    
    # Plotting stacked bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(label_counts.index, label_counts.values, color='skyblue', label='Number of Images')
    plt.xlabel('Activity')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images Across Activity Classes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend()
    plt.savefig('bar_chart.png')
    
    # Plotting pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Proportional Distribution of Images Across Activity Classes')
    plt.savefig('pie_chart.png')
    
    print('Charts Generated!')