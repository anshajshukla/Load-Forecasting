# Explainable Model Comparison Results

This document provides an explanation of the model comparison results to help you understand the significance of each metric.

## Model Comparison Results

| Model Type   | MSE    | MAPE (%) | R²    | Best Model Based on MAPE |
|--------------|--------|----------|-------|--------------------------|
| GRU          | 0.0641 | 5.80     | 0.95  |                          |
| ENHANCED_GRU | 0.0772 | 5.50     | 0.94  |                          |
| LSTM         | 0.0351 | 4.02     | 0.96  | Yes                      |
| BILSTM       | 0.0728 | 5.53     | 0.95  |                          |

## Explanation of Metrics

- **MSE (Mean Squared Error)**: This metric measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. A lower MSE indicates a better fit to the data.

- **MAPE (Mean Absolute Percentage Error)**: This metric expresses accuracy as a percentage. It measures the size of the error in percentage terms. A lower MAPE indicates better model accuracy.

- **R² (R-squared Score)**: This metric provides an indication of goodness of fit and therefore a measure of how well unseen samples are likely to be predicted by the model, through the proportion of explained variance. An R² score closer to 1 indicates a better fit.

## Conclusion

Based on the MAPE metric, the LSTM model is identified as the best performing model. It has the lowest MAPE, indicating that it has the highest accuracy in predicting the data.

For further questions or clarifications, please contact the project maintainer.
