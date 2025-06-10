# Computational Performance Metrics

## Summary Table

| Scenario   |   Avg_Time_Sec |   Std_Time_Sec |   Min_Time_Sec |   Max_Time_Sec |   Avg_Variables |   Avg_Memory_MB |   Avg_Iterations |   Convergence_Rate_Pct |   Time_Per_Variable_Ms |
|:-----------|---------------:|---------------:|---------------:|---------------:|----------------:|----------------:|-----------------:|-----------------------:|-----------------------:|
| Complex    |          0.667 |          0.066 |          0.58  |          0.736 |             180 |          36.446 |               76 |                 50     |                  4     |
| Medium     |          0.372 |          0.055 |          0.327 |          0.452 |              72 |          15.678 |               78 |                100     |                  5     |
| Simple     |          0.145 |          0.019 |          0.117 |          0.159 |              24 |           3.12  |               49 |                100     |                  6     |
| OVERALL    |          0.395 |          0.228 |          0.117 |          0.736 |              92 |          18.414 |               67 |                 83.333 |                  4.995 |

## Detailed Results

| Building           | Scenario   |   Devices |   Hours |   Variables |   Constraints |   Time_Seconds |   Memory_MB |   Iterations | Converged   |   Time_Per_Variable |
|:-------------------|:-----------|----------:|--------:|------------:|--------------:|---------------:|------------:|-------------:|:------------|--------------------:|
| DE_KN_residential6 | Simple     |         2 |      12 |          24 |            30 |          0.156 |       3.12  |           40 | True        |               0.006 |
| DE_KN_residential6 | Medium     |         3 |      24 |          72 |            57 |          0.349 |      15.678 |           67 | True        |               0.005 |
| DE_KN_residential6 | Complex    |         4 |      48 |         192 |           108 |          0.58  |      38.376 |           50 | True        |               0.003 |
| DE_KN_residential1 | Simple     |         2 |      12 |          24 |            30 |          0.117 |       3.12  |           28 | True        |               0.005 |
| DE_KN_residential1 | Medium     |         3 |      24 |          72 |            57 |          0.452 |      15.678 |           75 | True        |               0.006 |
| DE_KN_residential1 | Complex    |         4 |      48 |         192 |           108 |          0.69  |      38.376 |           52 | True        |               0.004 |
| DE_KN_residential2 | Simple     |         2 |      12 |          24 |            30 |          0.159 |       3.12  |           72 | True        |               0.007 |
| DE_KN_residential2 | Medium     |         3 |      24 |          72 |            57 |          0.327 |      15.678 |           55 | True        |               0.005 |
| DE_KN_residential2 | Complex    |         4 |      48 |         192 |           108 |          0.736 |      38.376 |          100 | False       |               0.004 |
| DE_KN_residential5 | Simple     |         2 |      12 |          24 |            30 |          0.149 |       3.12  |           57 | True        |               0.006 |
| DE_KN_residential5 | Medium     |         3 |      24 |          72 |            57 |          0.36  |      15.678 |          117 | True        |               0.005 |
| DE_KN_residential5 | Complex    |         3 |      48 |         144 |           105 |          0.662 |      30.654 |          102 | False       |               0.005 |

**Note**: Analysis based on 12 optimization runs using REAL GlobalOptimizer.
**Generated**: 2025-06-09 10:21:55
