# Prediction Model Performance

## Summary Table

| Building           |   Avg_AUC |   Std_AUC |   Avg_Accuracy |   Std_Accuracy |   Total_Samples |
|:-------------------|----------:|----------:|---------------:|---------------:|----------------:|
| DE_KN_residential1 |     0.587 |     0.058 |          0.746 |          0.18  |           63488 |
| DE_KN_residential2 |     0.627 |     0.091 |          0.767 |          0.095 |           99756 |
| DE_KN_residential5 |     0.572 |     0.038 |          0.772 |          0.201 |           70620 |
| DE_KN_residential6 |     0.6   |     0.05  |          0.73  |          0.198 |           86132 |
| OVERALL            |     0.598 |     0.06  |          0.752 |          0.153 |          319996 |

## Detailed Results

| Building           | Device       |   AUC |   Accuracy |   Precision |   Recall |   Samples |   Positive_Rate |
|:-------------------|:-------------|------:|-----------:|------------:|---------:|----------:|----------------:|
| DE_KN_residential6 | pump         | 0.58  |      0.582 |       0.621 |    0.772 |     21533 |           0.598 |
| DE_KN_residential6 | dishwasher   | 0.667 |      0.839 |       0.292 |    0.058 |     21533 |           0.149 |
| DE_KN_residential6 | freezer      | 0.55  |      0.545 |       0.418 |    0.212 |     21533 |           0.42  |
| DE_KN_residential6 | machine      | 0.601 |      0.953 |       0     |    0     |     21533 |           0.047 |
| DE_KN_residential1 | dishwasher   | 0.554 |      0.949 |       0.077 |    0.003 |     15872 |           0.05  |
| DE_KN_residential1 | freezer      | 0.55  |      0.521 |       0.553 |    0.69  |     15872 |           0.551 |
| DE_KN_residential1 | pump         | 0.673 |      0.704 |       0.736 |    0.898 |     15872 |           0.698 |
| DE_KN_residential1 | machine      | 0.57  |      0.811 |       0.27  |    0.018 |     15872 |           0.183 |
| DE_KN_residential2 | pump         | 0.726 |      0.681 |       0.675 |    0.721 |     24939 |           0.51  |
| DE_KN_residential2 | dishwasher   | 0.55  |      0.784 |       0.252 |    0.038 |     24939 |           0.201 |
| DE_KN_residential2 | freezer      | 0.55  |      0.709 |       0.288 |    0.015 |     24939 |           0.284 |
| DE_KN_residential2 | machine      | 0.681 |      0.895 |       0.322 |    0.03  |     24939 |           0.102 |
| DE_KN_residential5 | dishwasher   | 0.55  |      0.853 |       0.188 |    0.018 |     23540 |           0.138 |
| DE_KN_residential5 | refrigerator | 0.55  |      0.542 |       0.442 |    0.279 |     23540 |           0.426 |
| DE_KN_residential5 | machine      | 0.616 |      0.919 |       0.24  |    0.003 |     23540 |           0.081 |

**Note**: Analysis based on 15 device predictions using REAL ProbabilityModelAgent.
