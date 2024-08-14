import matplotlib.pyplot as plt
import pandas as pd
import re




df_model1 = pd.read_csv("/data/malaria_proj/building_data/unification/models/Faster-RCNN/model1_results.csv")
df_model2 = pd.read_csv("/data/malaria_proj/building_data/unification/models/Faster-RCNN/model2_results.csv")

plt.figure(figsize=(14, 6))
plt.plot(df_model1['Epoch'], df_model1['Train Loss'], label='Model 1 Training Loss', color='blue')
plt.plot(df_model2['Epoch'], df_model2['Train Loss'], label='Model 2 Training Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.savefig('training_loss_comparison.png')
plt.show()

# Plot Validation Precision Comparison and Save
plt.figure(figsize=(14, 6))
plt.plot(df_model1['Epoch'], df_model1['Val Precision'], label='Model 1 Precision', color='blue')
plt.plot(df_model2['Epoch'], df_model2['Val Precision'], label='Model 2 Precision', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Validation Precision Comparison')
plt.legend()
plt.savefig('val_precision_comparison.png')
plt.show()

# Plot Validation Recall Comparison and Save
plt.figure(figsize=(14, 6))
plt.plot(df_model1['Epoch'], df_model1['Val Recall'], label='Model 1 Recall', color='blue')
plt.plot(df_model2['Epoch'], df_model2['Val Recall'], label='Model 2 Recall', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.title('Validation Recall Comparison')
plt.legend()
plt.savefig('val_recall_comparison.png')
plt.show()

# Plot Validation F1-Score Comparison and Save
plt.figure(figsize=(14, 6))
plt.plot(df_model1['Epoch'], df_model1['Val F1-score'], label='Model 1 F1-Score', color='blue')
plt.plot(df_model2['Epoch'], df_model2['Val F1-score'], label='Model 2 F1-Score', color='orange')
plt.xlabel('Epochs')
plt.ylabel('F1-Score')
plt.title('Validation F1-Score Comparison')
plt.legend()
plt.savefig('val_f1_score_comparison.png')
plt.show()

