from prometheus_client import Summary, Counter, Gauge, Enum


# TODO: PROMEHEUS:
# metrics
# - numeric
# - category
model_creation_metrics = {
    'train_loss':{'value':random.random(), 'description': 'training loss (MSE)', 'type':'numeric'},
    'test_loss':{'value':random.random(), 'description': 'test loss (MSE)', 'type':'numeric'},
    'optimizer':{'value':random.choice(['SGM', 'RMSProp', 'Adagrad']),
        'description':'ml model optimizer function',
        'type': 'category',
        'categories':['SGD', 'RMSProp', 'Adagrad', 'Adam', 'lbfqs']}
}
def pre_recorded_metrics(metrics: dict) -> None:
    """
    Pass pre-recorded metrics from dict to Prometheus.
    This allows recording two types of metrics
        - numeric (int, float, etc.)
        - categorical
    Lists and matrices must be split so that each cell is their own metric.

    Format:
    metrics = {
        'metric_name':{
            'value': int, float or str if 'type' = category,
            'description': str,
            'type': str -> 'numeric' or 'category',
            'categories': [str], e.g. ['A', 'B', 'C']. only required if 'type' = category
        }
    }

    Example: 

    metrics = {
        'train_loss':{'value':0.95, 'description': 'training loss (MSE)', 'type':'numeric'},
        'test_loss':{'value':0.96, 'description': 'test loss (MSE)', 'type':'numeric'},
        'optimizer':{'value':random.choice(['SGM', 'RMSProp', 'Adagrad']),
            'description':'ml model optimizer function',
            'type': 'category',
            'categories':['SGD', 'RMSProp', 'Adagrad', 'Adam']}
    """

    for metric_name in metrics.keys():
        metric = model_creation_metrics[metric_name]
        if metric['type'] == 'numeric':
            g = Gauge(metric_name, metric['description'])
            g.set(metric['value'])
        elif metric['type'] == 'category':
            s = Enum(
                metric_name,
                metric['description'],
                states = metric['categories']
            )
            s.state(metric['value'])



# TODO: PROMEHEUS:
# input:
#   - raw values (if not text or some other weird datatype)
#   - hist/sumstat (a bit more private)
# processing:
#   - time (total / hist )
#   - general resource usage
#   - request counter
# output:
#   - raw (if not text of some other weird datatype)
#   - if category
#   - hist/sumstat (a bit more private)