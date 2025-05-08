# Avalanche Data Engineering Pipeline using Snowpark & Pandas on Snowflake

# Setup
import modin.pandas as spd
import snowflake.snowpark.modin.plugin
import snowflake.snowpark.functions as F
from snowflake.snowpark.context import get_active_session

session = get_active_session()
session.query_tag = {
    "origin": "sf_devrel",
    "name": "de_100_vhol",
    "version": {"major": 1, "minor": 0},
    "attributes": {"is_quickstart": 1, "source": "notebook", "vignette": "snowpark_pandas"}
}

# Load local CSV files into Modin (pandas-on-Snowflake)
shipping_logs_mdf = spd.read_csv('shipping-logs.csv', parse_dates=['shipping_date'])
order_history_mdf = spd.read_csv('order-history.csv', parse_dates=['Date'])

# Rename and clean order history data
order_history_mdf = order_history_mdf.rename(columns={
    'Order ID': 'order_id',
    'Customer ID': 'customer_id',
    'Product ID': 'product_id',
    'Product Name': 'product_name',
    'Quantity Ordered': 'quantity_ordered',
    'Price': 'price',
    'Total Price': 'total_price',
    'Date': 'date'
})

def clean_price(price_str):
    return float(price_str.replace('$', '').strip())

order_history_mdf['price'] = order_history_mdf['price'].apply(clean_price)
order_history_mdf['total_price'] = order_history_mdf['total_price'].apply(clean_price)

# Join order and shipping logs
order_shipping_mdf = spd.merge(order_history_mdf, shipping_logs_mdf, on='order_id', how='inner')

# Analyze product orders
product_counts_mdf = order_shipping_mdf.groupby('product_name').size().reset_index(name='order_count')
product_counts_mdf = product_counts_mdf.sort_values('order_count', ascending=False)

# Pivot by delivery status
product_status_pivot_mdf = order_shipping_mdf.pivot_table(
    index='product_name',
    columns='status',
    values='order_id',
    aggfunc='count',
    fill_value=0
)
product_status_pivot_mdf['Total_Orders'] = product_status_pivot_mdf.sum(axis=1)
product_status_pivot_mdf = product_status_pivot_mdf.sort_values('Total_Orders', ascending=False)

# Snowpark Python: Load customer reviews
customer_reviews_sdf = session.table('customer_reviews')
product_sentiment_sdf = customer_reviews_sdf.group_by('PRODUCT') \
    .agg(F.round(F.avg('SENTIMENT_SCORE'), 2).alias('AVG_SENTIMENT_SCORE')) \
    .sort(F.col('AVG_SENTIMENT_SCORE').desc())

product_sentiment_sdf.write.save_as_table('PRODUCT_SENTIMENT_ANALYSIS', mode='overwrite')
