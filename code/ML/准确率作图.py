import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns

# Load the validation results
validation_results = pd.read_csv('E:/分类任务/模型保存/random_forest_BEST_2.csv')

# Map labels to their respective categories
label_mapping = {4: 'low', 5: 'medium', 6: 'high'}
validation_results['True Label'] = validation_results['True Label'].map(label_mapping)
validation_results['Predicted Label'] = validation_results['Predicted Label'].map(label_mapping)

# Calculate accuracy, precision, recall, and F1 score for each label
accuracy = accuracy_score(validation_results['True Label'], validation_results['Predicted Label'])
precision = precision_score(validation_results['True Label'], validation_results['Predicted Label'], average=None, labels=['low', 'medium', 'high'])
recall = recall_score(validation_results['True Label'], validation_results['Predicted Label'], average=None, labels=['low', 'medium', 'high'])
f1 = f1_score(validation_results['True Label'], validation_results['Predicted Label'], average=None, labels=['low', 'medium', 'high'])

# Generate classification report
report = classification_report(validation_results['True Label'], validation_results['Predicted Label'], target_names=['low', 'medium', 'high'])

# Print classification report
print("Classification Report:\n", report)

# Confusion Matrix with actual counts and percentages
conf_matrix = confusion_matrix(validation_results['True Label'], validation_results['Predicted Label'], labels=['low', 'medium', 'high'])

# Normalize confusion matrix for percentages
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Create annotations with counts and percentages
annotations = np.empty_like(conf_matrix).astype(str)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        annotations[i, j] = f'{conf_matrix[i, j]}\n({conf_matrix_normalized[i, j]:.2%})'

# Plot confusion matrix with counts and percentages
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_normalized, annot=annotations, fmt='', cmap='Blues', xticklabels=['low', 'medium', 'high'], yticklabels=['low', 'medium', 'high'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix with Counts and Percentages')
plt.savefig('E:/分类任务/模型保存/random_forest_BEST_2X.svg', format='svg')
plt.close()
# Save the calculation results of F1 and recall (recovery rate) as a CSV file
metrics_df = pd.DataFrame({
    'Label': ['low', 'medium', 'high'],
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})

metrics_df.to_csv('E:/分类任务/模型保存/metrics_results.csv', index=False)

metrics_df.head(), 'E:/分类任务/模型保存/random_forest_BEST_2X.svg', 'E:/分类任务/模型保存/random_forest_BEST_2X.csv'
