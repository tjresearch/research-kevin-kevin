set -uexo pipefail

rm -r -f models
mkdir models
gsutil -m cp gs://autopgn-assets/models/piece_detection_model.h5 ./models

