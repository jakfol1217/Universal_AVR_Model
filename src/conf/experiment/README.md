# Files including experiment configuration definitions
Files containing:
- \_esnb\_ -- refer to a model which includes the ESBN module.
- \_combined\_ -- refer to real-life/abstract image datasets combination.
- \_scoring\_ -- refer to training models, including image embedding and scoring modules (*scoring_model_feature_transformer_v1.py*, *scoring_model_v1.py*, *scoring_model_wren_v1.py*, *scoring_model_esnb.py*)
- \_images\_ -- refer to training using individual images, not combined into full tasks.
- \_trans\_ or \_transformer\_ -- refer to training with transformer scoring modules.
  
Furthermore, file names contain dataset names on which the training is performed (often it's more than one dataset).
