import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#confusion matrix values
TN = 104353
FN = 8526
FP = 15876
TP = 13925
conf_matrix = np.array([[TN, FP],
                        [FN, TP]])

#onvert to array
labels = np.array([[TN, FP],
                   [FN, TP]])

#plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()