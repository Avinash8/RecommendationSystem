# Recommendation Service

1. Open the Windows PowerShell.

2. Change the Directory to the directory of the recommendation system.

3. Run the following Docker commands.

docker build -t rec-sys .

docker images

docker run -d -p 5000:5000 --name rec-server rec-sys

### Build the Model####

python -m recommendation.modelbuilding.restrictedboltzmannmachine.ModelPredict.py

python -m recommendation.modelbuilding.stackingrfxgb.ModelPredict.py

python Webservice.py