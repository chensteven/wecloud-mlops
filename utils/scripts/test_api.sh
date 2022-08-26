curl -X 'POST' \
  'http://127.0.0.1/api/v1/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "sentence": "I love machine learning, deep learning and mathematics!"
}'