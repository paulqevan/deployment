![CI](https://github.com/paulqevan/deployment/actions/workflows/ci.yml/badge.svg)


## Live API
Base URL: https://deployment-buhj.onrender.com

- Docs: https://deployment-buhj.onrender.com/docs
- Health: GET /health
- Predict: POST /predict
- Metadata: GET /metadata


### Example
```bash
curl -X POST "https://deployment-buhj.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":2,"Glucose":120,"BloodPressure":70,"SkinThickness":20,"Insulin":79,"BMI":28.0,"DiabetesPedigreeFunction":0.5,"Age":33}'
