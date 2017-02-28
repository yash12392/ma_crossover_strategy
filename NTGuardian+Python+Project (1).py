
# coding: utf-8

# In[1]:

import pandas as pd
import pandas.io.data as web   # Package and modules for importing data; this code may change depending on pandas version
import datetime
 
# We will look at stock prices over the past year, starting at January 1, 2016
start = datetime.datetime(2016,1,1)
end = datetime.date.today()
 
# Let's get Apple stock data; Apple's ticker symbol is AAPL
# First argument is the series we want, second is the source ("yahoo" for Yahoo! Finance), third is the start date, fourth is the end date
apple = web.DataReader("AAPL", "yahoo", start, end)


# In[2]:

apple.head()


# In[3]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'pylab inline')
pylab.rcParams['figure.figsize'] = (15, 9)

apple['Adj Close'].plot(grid = True)


# In[4]:

microsoft = web.DataReader('MSFT','yahoo',start,end)
google = web.DataReader('GOOG','yahoo',start,end)

stocks = pd.DataFrame({"AAPL": apple['Adj Close'], 'MSFT': microsoft['Adj Close'], 'GOOG': google['Adj Close']})
stocks.head()
stocks.plot(grid = True)


# In[5]:

stocks.plot(secondary_y = ["AAPL", "MSFT"], grid = True)


# In[6]:

stocks_return = stocks.apply(lambda x: x/x[0])
stocks_return.head()


# In[7]:

stocks_return.plot(grid = True).axhline(y = 1, color = "black", lw = 2)


# In[8]:

apple["20d"] = np.round(apple["Close"].rolling(window = 20, center = False).mean(), 2)


# In[9]:

start = datetime.datetime(2010,1,1)
apple = web.DataReader("AAPL", "yahoo", start, end)
apple["20d"] = np.round(apple["Close"].rolling(window = 20, center = False).mean(), 2)
apple["50d"] = np.round(apple["Close"].rolling(window = 50, center = False).mean(), 2)
apple["200d"] = np.round(apple["Close"].rolling(window = 200, center = False).mean(), 2)


# In[10]:

apple['20d-50d'] = apple['20d'] - apple['50d']
apple.tail()


# In[11]:

# np.where() is a vectorized if-else function, where a condition is checked for each component of a vector, and the first argument passed is used when the condition holds, and the other passed if it does not
apple['Regime'] = np.where(apple['20d-50d']>0,1,0)
apple['Regime'] = np.where(apple['20d-50d']<0,-1,apple['Regime'])
apple["Regime"].plot(ylim = (-2,2)).axhline(y = 0, color = "black")


# In[12]:

apple['Regime'].value_counts()


# In[13]:

# To ensure that all trades close out, I temporarily change the regime of the last row to 0
regime_orig = apple.ix[-1,"Regime"]
apple.ix[-1,"Regime"] = 0
apple['Signal'] = np.sign(apple['Regime'] - apple['Regime'].shift(1))
#Reversing the situation
apple.ix[-1,"Regime"] = regime_orig
apple.tail()


# In[14]:

apple["Signal"].plot(ylim = (-2, 2))


# In[15]:

apple["Signal"].value_counts()


# In[16]:

apple.loc[apple['Signal'] == 1, "Close"]


# In[17]:

# Create a DataFrame with trades, including the price at the trade and the regime under which the trade is made.
trade_signals = pd.concat([pd.DataFrame({"Price" : apple.loc[apple['Signal'] == 1, "Close"], "Regime": apple.loc[apple['Signal'] == 1, "Regime"], "Signal" : "Buy"}),
                          pd.DataFrame({"Price": apple.loc[apple['Signal'] == -1, "Close"], "Regime": apple.loc[apple['Signal'] == -1, "Regime"], "Signal": "Sell"})])
trade_signals.sort_index(inplace = True)
trade_signals


# In[18]:

# Let's see the profitability of long trades
apple_long_profits = pd.DataFrame({
        "Price": trade_signals.loc[(trade_signals["Signal"] == "Buy") &
                                  trade_signals["Regime"] == 1, "Price"],
        "Profit": pd.Series(trade_signals["Price"] - trade_signals["Price"].shift(1)).loc[
            trade_signals.loc[(trade_signals["Signal"].shift(1) == "Buy") & (trade_signals["Regime"].shift(1) == 1)].index
        ].tolist(),
        "End Date": trade_signals["Price"].loc[
            trade_signals.loc[(trade_signals["Signal"].shift(1) == "Buy") & (trade_signals["Regime"].shift(1) == 1)].index
        ].index
    })
apple_long_profits


# In[19]:

from matplotlib.dates import DateFormatter, WeekdayLocator,    DayLocator, MONDAY
from matplotlib.finance import candlestick_ohlc
 
def pandas_candlestick_ohlc(dat, stick = "day", otherseries = None):
    """
    :param dat: pandas DataFrame object with datetime64 index, and float columns "Open", "High", "Low", and "Close", likely created via DataReader from "yahoo"
    :param stick: A string or number indicating the period of time covered by a single candlestick. Valid string inputs include "day", "week", "month", and "year", ("day" default), and any numeric input indicates the number of trading days included in a period
    :param otherseries: An iterable that will be coerced into a list, containing the columns of dat that hold other series to be plotted as lines
 
    This will show a Japanese candlestick plot for stock data stored in dat, also plotting other series if passed.
    """
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    dayFormatter = DateFormatter('%d')      # e.g., 12
 
    # Create a new DataFrame which includes OHLC data for each period specified by stick input
    transdat = dat.loc[:,["Open", "High", "Low", "Close"]]
    if (type(stick) == str):
        if stick == "day":
            plotdat = transdat
            stick = 1 # Used for plotting
        elif stick in ["week", "month", "year"]:
            if stick == "week":
                transdat["week"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[1]) # Identify weeks
            elif stick == "month":
                transdat["month"] = pd.to_datetime(transdat.index).map(lambda x: x.month) # Identify months
            transdat["year"] = pd.to_datetime(transdat.index).map(lambda x: x.isocalendar()[0]) # Identify years
            grouped = transdat.groupby(list(set(["year",stick]))) # Group by year and other appropriate variable
            plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                            "High": max(group.High),
                                            "Low": min(group.Low),
                                            "Close": group.iloc[-1,3]},
                                           index = [group.index[0]]))
            if stick == "week": stick = 5
            elif stick == "month": stick = 30
            elif stick == "year": stick = 365
 
    elif (type(stick) == int and stick >= 1):
        transdat["stick"] = [np.floor(i / stick) for i in range(len(transdat.index))]
        grouped = transdat.groupby("stick")
        plotdat = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []}) # Create empty data frame containing what will be plotted
        for name, group in grouped:
            plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0,0],
                                        "High": max(group.High),
                                        "Low": min(group.Low),
                                        "Close": group.iloc[-1,3]},
                                       index = [group.index[0]]))
 
    else:
        raise ValueError('Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')
 
 
    # Set plot parameters, including the axis object ax used for plotting
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
    else:
        weekFormatter = DateFormatter('%b %d, %Y')
    ax.xaxis.set_major_formatter(weekFormatter)
 
    ax.grid(True)
 
    # Create the candelstick chart
    candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                      plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                      colorup = "black", colordown = "red", width = stick * .4)
 
    # Plot other series (such as moving averages) as lines
    if otherseries != None:
        if type(otherseries) != list:
            otherseries = [otherseries]
        dat.loc[:,otherseries].plot(ax = ax, lw = 1.3, grid = True)
 
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
 
    plt.show()
 
pandas_candlestick_ohlc(apple)


# In[20]:

pandas_candlestick_ohlc(apple, stick = 45, otherseries = ["20d", "50d", "200d"])


# In[21]:

def ohlc_adj(dat):
    return pd.DataFrame({"Open": dat['Open'] * dat['Adj Close']/dat["Close"],
                        "High": dat['High'] * dat['Adj Close']/dat["Close"],
                        "Low": dat['Low'] * dat['Adj Close']/dat["Close"],
                        "Close": dat['Adj Close']})

apple_adj = ohlc_adj(apple)


# In[22]:

# This next code repeats all the earlier analysis we did on the adjusted data
 
apple_adj["20d"] = np.round(apple_adj["Close"].rolling(window = 20, center = False).mean(), 2)
apple_adj["50d"] = np.round(apple_adj["Close"].rolling(window = 50, center = False).mean(), 2)
apple_adj["200d"] = np.round(apple_adj["Close"].rolling(window = 200, center = False).mean(), 2)
 
apple_adj['20d-50d'] = apple_adj['20d'] - apple_adj['50d']
# np.where() is a vectorized if-else function, where a condition is checked for each component of a vector, and the first argument passed is used when the condition holds, and the other passed if it does not
apple_adj["Regime"] = np.where(apple_adj['20d-50d'] > 0, 1, 0)
# We have 1's for bullish regimes and 0's for everything else. Below I replace bearish regimes's values with -1, and to maintain the rest of the vector, the second argument is apple["Regime"]
apple_adj["Regime"] = np.where(apple_adj['20d-50d'] < 0, -1, apple_adj["Regime"])
# To ensure that all trades close out, I temporarily change the regime of the last row to 0
regime_orig = apple_adj.ix[-1, "Regime"]
apple_adj.ix[-1, "Regime"] = 0
apple_adj["Signal"] = np.sign(apple_adj["Regime"] - apple_adj["Regime"].shift(1))
# Restore original regime data
apple_adj.ix[-1, "Regime"] = regime_orig
 
# Create a DataFrame with trades, including the price at the trade and the regime under which the trade is made.
apple_adj_signals = pd.concat([
        pd.DataFrame({"Price": apple_adj.loc[apple_adj["Signal"] == 1, "Close"],
                     "Regime": apple_adj.loc[apple_adj["Signal"] == 1, "Regime"],
                     "Signal": "Buy"}),
        pd.DataFrame({"Price": apple_adj.loc[apple_adj["Signal"] == -1, "Close"],
                     "Regime": apple_adj.loc[apple_adj["Signal"] == -1, "Regime"],
                     "Signal": "Sell"}),
    ])
apple_adj_signals.sort_index(inplace = True)
apple_adj_long_profits = pd.DataFrame({
        "Price": apple_adj_signals.loc[(apple_adj_signals["Signal"] == "Buy") &
                                  apple_adj_signals["Regime"] == 1, "Price"],
        "Profit": pd.Series(apple_adj_signals["Price"] - apple_adj_signals["Price"].shift(1)).loc[
            apple_adj_signals.loc[(apple_adj_signals["Signal"].shift(1) == "Buy") & (apple_adj_signals["Regime"].shift(1) == 1)].index
        ].tolist(),
        "End Date": apple_adj_signals["Price"].loc[
            apple_adj_signals.loc[(apple_adj_signals["Signal"].shift(1) == "Buy") & (apple_adj_signals["Regime"].shift(1) == 1)].index
        ].index
    })
 
pandas_candlestick_ohlc(apple_adj, stick = 45, otherseries = ["20d", "50d", "200d"])


# In[23]:

apple_adj_long_profits

# We need to get the low of the price during each trade.
tradeperiods = pd.DataFrame({"Start": apple_adj_long_profits.index,
                            "End": apple_adj_long_profits["End Date"]})
apple_adj_long_profits["Low"] = tradeperiods.apply(lambda x: min(apple_adj.loc[x["Start"]:x["End"], "Low"]), axis = 1)
apple_adj_long_profits


# In[24]:

# Now we have all the information we need to simulate this strategy
cash = 1000000
apple_backtest = pd.DataFrame({"Start Portfolio Value": [], "End Portfolio Value": [], "End Date": [], "Shares": [], "Share Price": [],
                              "Trade Value": [], "Profit per share": [], "Stop Loss Triggered": [], "Total Profit": []})


# In[25]:

port_value = 0.1 #Max only 10% of the portfolio is bet
batch = 100
stoploss = 0.2 #Downside is max at 20%

for index,row in apple_adj_long_profits.iterrows():
    batches = np.floor(cash*port_value) // np.ceil(batch*row['Price'])
    trade_val = batches*batch*row['Price'] #How much money is invested
    if row["Low"] < (1 - stoploss)*row['Price']:
        share_profit = np.round((1-stoploss)*row['Price'],2)
        stop_trig = True
    else:
        share_profit = row['Profit']
        stop_trig = False

    profit = batches*batch*share_profit
    #Add a row to the backtest data to contain the results
    apple_backtest = apple_backtest.append(pd.DataFrame({"Start Portfolio Value": cash, "End Portfolio Value": cash+profit, "End Date": row["End Date"], "Shares": batch*batches, "Share Price": row["Price"],
                              "Trade Value": trade_val, "Profit per share": share_profit, "Stop Loss Triggered": profit, "Total Profit": stop_trig},index = [index]))
    cash = max(0,cash+profit)

apple_backtest
    


# In[26]:

apple_backtest["End Portfolio Value"].plot()


# In[27]:

(apple_backtest.ix[-1,"End Portfolio Value"] - (cash*0.1))/(cash*0.1)


# In[48]:

def ma_crossover_orders(stocks,fast,slow):
    """:param stocks: A list of tuples, the first argument in each tuple being a string containing the ticker symbol of each stock (or however you want the stock represented, so long as it's unique), and the second being a pandas DataFrame containing the stocks, with a "Close" column and indexing by date (like the data frames returned by the Yahoo! Finance API)
    :param fast: Integer for the number of days used in the fast moving average
    :param slow: Integer for the number of days used in the slow moving average
 
    :return: pandas DataFrame containing stock orders
 
    This function takes a list of stocks and determines when each stock would be bought or sold depending on a moving average crossover strategy, returning a data frame with information about when the stocks in the portfolio are bought or sold according to the strategy
    """
    
    fast_str = str(fast) + 'd'
    slow_str = str(slow) + 'd'
    ma_diff_str = fast_str + '-' + slow_str
    
    trades = pd.DataFrame({"Price": [], "Regime": [], "Signal": []})
    for s in stocks:
        s[1][fast_str] = np.round(s[1]["Close"].rolling(window = fast, center = False).mean(),2)
        s[1][slow_str] = np.round(s[1]["Close"].rolling(window = slow, center = False).mean(),2)
        s[1][ma_diff_str] = s[1][fast_str] - s[1][slow_str]
        
        s[1]["Regime"] = np.where(s[1][ma_diff_str] > 0, 1,0)
        s[1]["Regime"] = np.where(s[1][ma_diff_str] < 0, -1, s[1]["Regime"])
        regime_orig = s[1].ix[-1, "Regime"]
        s[1]["Signal"] = s[1]["Regime"] - s[1]["Regime"].shift(1)
        s[1].ix[-1,"Regime"] = regime_orig
        
        #Get Signals from trade
        signals = pd.concat([
                pd.DataFrame({"Price": s[1].loc[s[1]["Signal"] == 1, "Close"],
                                          "Regime": s[1].loc[s[1]["Signal"] == 1, "Regime"],
                                            "Signal": "Buy"}),
                pd.DataFrame({"Price": s[1].loc[s[1]["Signal"] == -1, "Close"],
                                         "Regime": s[1].loc[s[1]["Signal"] == -1, "Regime"],
                                         "Signal": "Sell"}),])
        signals.index = pd.MultiIndex.from_product([signals.index,[s[0]]],names = ["Date", "Symbol"])
        trades = trades.append(signals)
        
        trades.sort_index(inplace = True)
        trades.index = pd.MultiIndex.from_tuples(trades.index, names = ["Date","Symbol"])
        
        return trades


# In[60]:

def backtest(signals, cash, port_value = .1, batch = 100):
    SYMBOL = 1 # Constant for which element in index represents symbol
    portfolio = dict()    # Will contain how many stocks are in the portfolio for a given symbol
    port_prices = dict()  # Tracks old trade prices for determining profits
    # Dataframe that will contain backtesting report
    results = pd.DataFrame({"Start Cash": [],
                            "End Cash": [],
                            "Portfolio Value": [],
                            "Type": [],
                            "Shares": [],
                            "Share Price": [],
                            "Trade Value": [],
                            "Profit per Share": [],
                            "Total Profit": []})
 
    for index, row in signals.iterrows():
        # These first few lines are done for any trade
        shares = portfolio.setdefault(index[SYMBOL], 0)
        trade_val = 0
        batches = 0
        cash_change = row["Price"] * shares   # Shares could potentially be a positive or negative number (cash_change will be added in the end; negative shares indicate a short)
        portfolio[index[SYMBOL]] = 0  # For a given symbol, a position is effectively cleared
 
        old_price = port_prices.setdefault(index[SYMBOL], row["Price"])
        portfolio_val = 0
        for key, val in portfolio.items():
            portfolio_val += val * port_prices[key]
 
        if row["Signal"] == "Buy" and row["Regime"] == 1:  # Entering a long position
            batches = np.floor((portfolio_val + cash) * port_value) // np.ceil(batch * row["Price"]) # Maximum number of batches of stocks invested in
            trade_val = batches * batch * row["Price"] # How much money is put on the line with each trade
            cash_change -= trade_val  # We are buying shares so cash will go down
            portfolio[index[SYMBOL]] = batches * batch  # Recording how many shares are currently invested in the stock
            port_prices[index[SYMBOL]] = row["Price"]   # Record price
            old_price = row["Price"]
        elif row["Signal"] == "Sell" and row["Regime"] == -1: # Entering a short
            pass
            # Do nothing; can we provide a method for shorting the market?
        #else:
            #raise ValueError("I don't know what to do with signal " + row["Signal"])
 
        pprofit = row["Price"] - old_price   # Compute profit per share; old_price is set in such a way that entering a position results in a profit of zero
 
        # Update report
        results = results.append(pd.DataFrame({
                "Start Cash": cash,
                "End Cash": cash + cash_change,
                "Portfolio Value": cash + cash_change + portfolio_val + trade_val,
                "Type": row["Signal"],
                "Shares": batch * batches,
                "Share Price": row["Price"],
                "Trade Value": abs(cash_change),
                "Profit per Share": pprofit,
                "Total Profit": batches * batch * pprofit
            }, index = [index]))
        cash += cash_change  # Final change to cash balance
 
    results.sort_index(inplace = True)
    results.index = pd.MultiIndex.from_tuples(results.index, names = ["Date", "Symbol"])
 
    return results
 
# Get more stocks
microsoft = web.DataReader("MSFT", "yahoo", start, end)
google = web.DataReader("GOOG", "yahoo", start, end)
facebook = web.DataReader("FB", "yahoo", start, end)
twitter = web.DataReader("TWTR", "yahoo", start, end)
netflix = web.DataReader("NFLX", "yahoo", start, end)
amazon = web.DataReader("AMZN", "yahoo", start, end)
yahoo = web.DataReader("YHOO", "yahoo", start, end)
sony = web.DataReader("SNY", "yahoo", start, end)
nintendo = web.DataReader("NTDOY", "yahoo", start, end)
ibm = web.DataReader("IBM", "yahoo", start, end)
hp = web.DataReader("HPQ", "yahoo", start, end)


# In[61]:

signals = ma_crossover_orders([("AAPL", ohlc_adj(apple)),
                              ("MSFT",  ohlc_adj(microsoft)),
                              ("GOOG",  ohlc_adj(google)),
                              ("FB",    ohlc_adj(facebook)),
                              ("TWTR",  ohlc_adj(twitter)),
                              ("NFLX",  ohlc_adj(netflix)),
                              ("AMZN",  ohlc_adj(amazon)),
                              ("YHOO",  ohlc_adj(yahoo)),
                              ("SNY",   ohlc_adj(yahoo)),
                              ("NTDOY", ohlc_adj(nintendo)),
                              ("IBM",   ohlc_adj(ibm)),
                              ("HPQ",   ohlc_adj(hp))],fast = 50,slow = 20)
signals


# In[62]:

bk = backtest(signals,1000000)
bk


# In[65]:

bk["Portfolio Value"].groupby(level=0).apply(lambda x: x[-1]).plot()


# In[66]:

spyder = web.DataReader("SPY", "yahoo", start, end)
spyder.iloc[[0,-1],:]


# In[71]:

batches = 1000000//np.ceil(100*spyder.ix[0,"Adj Close"])
trade_val = batches*batch*spyder.ix[0,"Adj Close"]
final_val = batches*batch*spyder.ix[-1,"Adj Close"] + (1000000 - trade_val)
final_val


# In[72]:

axis_bench = (spyder["Adj Close"]/spyder.ix[0, "Adj Close"]).plot(label="SPY")
axis_bench = (bk["Portfolio Value"].groupby(level=0).apply(lambda x: x[-1]/1000000).plot(ax = axis_bench, label="Portfolio"))
axis_bench.legend(axis_bench.get_lines(),[l.get_label() for l in axis_bench.get_lines()],loc = 'best')
axis_bench


# In[ ]:



