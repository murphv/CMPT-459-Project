import pandas as pd
### TODO: The logic should work for any input of year and quarter

def main():

    data_path = '../data/'

    stocks_data = pd.read_csv(f'{data_path}stocks_data.csv')
    stocks_data['Date'] = pd.to_datetime(stocks_data['Date'])

    index_data = pd.read_csv(f'{data_path}index_data.csv')
    index_data['Date'] = pd.to_datetime(index_data['Date'])
    index_data['month'] = index_data['Date'].dt.month

    # Computing index changes
    dow_avg_monthly = index_data[index_data['Company'] == '^DJI'].drop(columns=['Company', 'Date']).groupby(by=['month']).mean().reset_index()
    sp_avg_monthly = index_data[index_data['Company'] == '^GSPC'].drop(columns=['Company', 'Date']).groupby(by=['month']).mean().reset_index()
    nasdaq_avg_monthly = index_data[index_data['Company'] == '^IXIC'].drop(columns=['Company', 'Date']).groupby(by=['month']).mean().reset_index()
    
    dow_start = dow_avg_monthly[dow_avg_monthly['month'] <= 3].drop(columns='month').mean()['Close']
    dow_end = dow_avg_monthly[dow_avg_monthly['month'] == 6].drop(columns='month').mean()['Close']
    dow_change = ((dow_end - dow_start) / dow_start) * 100

    sp_start = sp_avg_monthly[sp_avg_monthly['month'] <= 3].drop(columns='month').mean()['Close']
    sp_end = sp_avg_monthly[sp_avg_monthly['month'] == 6].drop(columns='month').mean()['Close']
    sp_change = ((sp_end - sp_start) / sp_end) * 100

    nas_start = nasdaq_avg_monthly[nasdaq_avg_monthly['month'] <= 3].drop(columns='month').mean()['Close']
    nas_end = nasdaq_avg_monthly[nasdaq_avg_monthly['month'] == 6].drop(columns='month').mean()['Close']
    nas_change = ((nas_end - nas_start) / nas_start) * 100

    # Adding index change over the quarter to the stocks data
    stocks_data['dow_change'] = dow_change
    stocks_data['sp_change'] = sp_change
    stocks_data['nas_change'] = nas_change


    #Computing the Stock change over the quarter and adding it as column
    stocks_data['stock_change'] = 0
    companies = stocks_data['Company'].unique()

    for company in companies:
        company_df = stocks_data[stocks_data['Company'] == company].copy()
        company_df['month'] = company_df['Date'].dt.month
        
        company_avg_monthly = company_df.drop(columns=['Company', 'Date']).groupby(by=['month']).mean().reset_index()
        
        company_start = company_avg_monthly[company_avg_monthly['month'] <= 3].drop(columns='month').mean()['Close']
        company_end = company_avg_monthly[company_avg_monthly['month'] == 6].drop(columns='month').mean()['Close']
        
        # Compute the stock change
        stock_change = ((company_end - company_start) / company_start) * 100
        
        # Assign the computed stock_change to all records of the company
        stocks_data.loc[stocks_data['Company'] == company, 'stock_change'] = stock_change

    # Now that we dont need the stocks information for the other months except the quarter. we will remove that from our dataset
    stocks_data = stocks_data[stocks_data['Date'].dt.month <= 3]
    stocks_data.to_csv(f'{data_path}stocks_vs_index_data.csv', index=False)
    



if __name__ == "__main__":
    main()