Add a dataset
-------------
- Implement dataset under [`dataset`](/dataset)
    - Dataset, Metadata
    - Config yaml
    - if has wrapper: _wrap_dataset
- Add config file under [`config/datamodule`](/config/datamodule)
- Create trainer profile under [`config`](/config) with name `train-{dataset_name}.yaml`