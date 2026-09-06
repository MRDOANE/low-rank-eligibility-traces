# Data governance and leakage controls

The archive contains no Yearbook images and no Criteo records.

For Criteo, preprocessing creates two physically separate Parquet files. `safe_features.parquet` contains an event identifier and the frozen pre-outcome allowlist. `labels_and_schedule.parquet` contains labels, feedback times, delays, and split codes. Before any model is constructed, the runner rejects an intersection between the safe schema and the forbidden-field set.

The model-input allowlist is:

`timestamp`, `uid`, `campaign`, `cost`, `time_since_last_click`, `cat1` through `cat9`.

The blocked set is:

`conversion`, `conversion_timestamp`, `conversion_id`, `attribution`, `click`, `click_pos`, `click_nb`, `cpo`.

`click` is used once to define the automatically selected click stream and is removed afterward. `conversion` supplies the target only. `conversion_timestamp` supplies the feedback schedule only. Negative outcomes mature after 30 days because the dataset card defines conversion over the 30 days following an impression. No field derived from the eventual conversion enters hashing or the adapter.

Yearbook labels are parsed by the pinned upstream loader from the published class paths. They are available to the evaluator at prediction time but are passed to the learner only when the configured delayed batch is released.

