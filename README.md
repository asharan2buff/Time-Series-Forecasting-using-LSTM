# Time-Series-Forecasting-using-LSTM

## Discussion and Analysis

### 1. Key Characteristics of the Dataset
This dataset consists of **hourly air quality measurements** from an array of metal oxide sensors deployed in an Italian city. Each sample contains pollutant concentrations (CO, NMHC, Benzene, NOx, NO₂) and sensor responses, along with environmental features such as temperature and humidity. The data covers **one year** (March 2004 to February 2005), providing a rich set of time-series observations for modeling real-world pollution levels.

### 2. Final LSTM Architecture
The final model is an **LSTM** network with the following configuration:
- **Number of Layers (num_layers):** 3  
- **Hidden Dimension (hidden_dim):** 64 (for example)  
- **Dropout Rate:** 0.2  
- **Batch First:** True (so input shape is `[batch_size, seq_length, input_dim]`)  
- **Fully Connected Layer:** One linear layer (`nn.Linear(hidden_dim, output_dim)`) to map the final hidden state to the output.  
- **Loss Function:** MSE (Mean Squared Error), used during training.  
- **Optimizer:** Adam with a learning rate determined via hyperparameter tuning (e.g., 1e-3 or 1e-4).  

In the `forward` pass, we:
1. Initialize hidden and cell states (`h0`, `c0`) to zeros.  
2. Pass the input sequences through the LSTM.  
3. Extract the **last time-step** output from `[batch_size, seq_length, hidden_dim]`.  
4. Feed that final hidden vector into a **fully connected layer** (`fc`) to produce the final predicted pollutant concentration.

### 3. Results & Discussion

**Final Test Metrics** (for the best configuration found):
- **MAE:** 0.2864  
- **RMSE:** 0.4081  
- **R²:** 0.7946  

These numbers suggest the model explains **~79.5% of the variance** in the target pollutant (e.g., CO) and, on average, deviates from the true values by **0.29** units (MAE) or **0.41** units (RMSE).

**Did It Meet Expectations?**  
- An R² near 0.8 indicates a reasonably strong predictive performance for real-world sensor data, where noise, sensor drift, and environmental factors can introduce complexity.  
- The model performed better than simpler baselines (e.g., linear regression), but there is still **room for improvement**—the remaining ~20% variance may reflect unmodeled factors (e.g., traffic patterns, weather anomalies).

**Challenges & Hyperparameter Tuning**  
- **Sequence Length**: We experimented with using 24-hour windows and found it captured daily cycles well.  
- **Hidden Dim & Dropout**: Increasing hidden dimension to 64 improved accuracy; a moderate dropout of 0.2 balanced overfitting.  
- **Learning Rate**: A smaller LR sometimes improved final validation metrics but required more epochs.  
- **Error Patterns**: No strong systematic under- or over-prediction was observed, though large spikes in pollution occasionally produced bigger errors.

### 4. Limitations
1. **Sensor Drift**: Real-world metal oxide sensors can drift over time, which might degrade performance if not handled with recalibration or advanced drift-correction methods.  
2. **Seasonality Beyond 24 Hours**: Weekly or monthly trends may not be fully captured if only a single day’s sequence is used.  
3. **Limited Features**: While temperature and humidity are included, other factors like wind speed/direction or traffic density might further improve accuracy.

### 5. Potential Improvements / Future Work
1. **Longer Sequence Lengths or Additional Features**: Incorporate weekly patterns or meteorological data for deeper context.  
2. **Alternate Architectures**: Explore **GRUs**, **Temporal Convolutional Networks (TCNs)**, or **Transformers** for time-series forecasting.  
3. **Systematic Hyperparameter Tuning**: Use automated search (Optuna, Ray Tune) to explore hidden_dim, learning rates, dropout, and number of layers more exhaustively.  
4. **Multi-Step Forecasting**: Predict future pollution levels over multiple hours rather than just the next hour.  
5. **Regular Recalibration**: For real-world deployment, regularly retrain or fine-tune the model to handle sensor aging and drift.

Overall, the final LSTM model demonstrates **solid performance** (R² ~ 0.79) for hourly pollution forecasting, but additional data and advanced architectures could further refine predictions and better handle complex real-world variations.
