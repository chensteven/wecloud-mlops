## Hello


Once a model is trained (and hopefully tested), you want to make its predictive
capability accessible to users. In Chapter 7, we talked at length on how a model can
serve its predictions: online or batch prediction. We also discussed how the simplest
way to deploy a model is to push your model and its dependencies to a location
accessible in production then expose your model as an endpoint to your users. If you
do online prediction, this endpoint will provoke your model to generate a prediction.
If you do batch prediction, this endpoint will fetch a precomputed prediction.

A deployment service can help with both pushing your models and their dependencies
to production and exposing your models as endpoints. Since deploying is
the name of the game, deployment is the most mature among all ML platform
components, and many tools exist for this. All major cloud providers offer tools
for deployment: AWS with SageMaker, GCP with Vertex AI, Azure with Azure ML,
Alibaba with Machine Learning Studio, and so on. There are also a myriad of startups
that offer model deployment tools such as MLflow Models, Seldon, Cortex, Ray
Serve, and so on.

Featurize and predict functions
Given a prediction request, how do you extract features and input these features
into the model to get back a prediction? The featurize and predict functions provide
the instruction to do so. These functions are usually wrapped in endpoints.