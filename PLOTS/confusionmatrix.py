import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#confusion matrix values
TN = 109989
FN = 11966
FP = 10932
TP = 10613
conf_matrix = np.array([[TN, FP],
                        [FN, TP]])

#convert to array
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