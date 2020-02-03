set -uexo pipefail

cd board_detection
rm -r -f models
mkdir models
gsutil -m cp gs://autopgn-assets/models/lattice_points_model.h5 ./models
gsutil -m cp gs://autopgn-assets/models/lattice_points_model.json ./models
