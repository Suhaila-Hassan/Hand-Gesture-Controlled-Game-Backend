# Hand Gesture Controlled Game - Backend/API

### Monitoring using Grafana and Prometheus
Following metrics were chosen to monitor backend/API performance:

A. Model Related Metric:
- prediction_latency_seconds

  Prediction Latency: Time taken by the model to make a prediction for measuring model performance and efficiency

- predicted_gestures_total

  Count of Predicted Gestures: Total number of predictions per gesture (up, down, left, right).

B. Data Related Metric:
- invalid_input_requests_total

  Total Inputs with Invalid Data Shape: Number of requests with wrong shape inputs, API has to recieve data in the shape of (21, 2).

C. Server Related Metric:
- http_requests_total

  Total API Requests: Total number of POST /predict requests for calculating traffic.

### Grafana Dashboard
![Grafana Dashboard](https://github.com/user-attachments/assets/846e4c30-e680-4191-b6db-ff082fb56154)
