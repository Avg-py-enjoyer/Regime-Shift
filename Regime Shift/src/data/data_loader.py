import yfinance as yf


class IndianMarketDataLoader:

    def __init__(self, tickers, start_date, end_date):

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def download_asset_prices(self):

        prices = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True
        )["Close"]

        return prices.dropna()

    def download_india_vix(self):

        vix = yf.download(
            "^INDIAVIX",
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True
        )["Close"]

        return vix.dropna()

    def download_usdinr(self):

        usdinr = yf.download(
            "INR=X",
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True
        )["Close"]

        return usdinr.dropna()

    def load_all_data(self):

        prices = self.download_asset_prices()
        india_vix = self.download_india_vix()
        usdinr = self.download_usdinr()

        return prices, india_vix, usdinr