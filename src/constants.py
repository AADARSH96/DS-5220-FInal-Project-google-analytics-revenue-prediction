# Constants for the project
NUMERICAL_FEATURES = [
    'totals.hits',
    'totals.pageviews',
    'totals.bounces',
    'totals.newVisits',
    'visitNumber'
]

CATEGORICAL_FEATURES = [
    'channelGrouping',
    'device.browser',
    'device.operatingSystem',
    'geoNetwork.continent',
    'geoNetwork.subContinent',
    'geoNetwork.country',
    'trafficSource.campaign',
    'trafficSource.source',
    'trafficSource.medium',
    'trafficSource.keyword',
    'trafficSource.isTrueDirect'
]

TARGET = 'log_transactionRevenue'
