from IPython.display import display
from datetime import datetime

now = datetime.now()
date_time = now.strftime("%m-%d-%Y")

import os
os.environ["RD_LIB_CONFIG_PATH"] = "./"
import refinitiv.data as rd
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


rd.open_session(config_name="./tests/refinitiv-data.config.json")

ticker = "AAPL.O"

hist = rd.get_history(universe="US5YT=RR", start="2013-04-02", end="2023-04-02")

display(hist)

df = rd.get_data(universe=[ticker], fields=["TR.WACCBeta", "TR.F.CashFlowOpBefChgInWkgCap", "TR.F.MktCap", "TR.F.EBIT", "TR.F.EBITDA", "TR.F.DeprAmortSuppl", "TR.F.CAPEXNetCF"])

display(df)



class DCF:

  def __init__(self, symbol, proj_growth, tgr):
    self._ticker = symbol
    # projected growth rates for operating cash flow
    self._proj_growth = proj_growth
    # forecasting period
    self._period = len(self._proj_growth)
    # terminal growth rate
    self._tgr = tgr

    self._info = rd.get_history(universe=ticker, start="2013-04-02", end=date_time)
    # reverse col order (ascending FYE)
    self._financials = rd.get_data(universe=[self._ticker], fields=["TR.F.IncomeStatement.fieldname","TR.F.IncomeStatement.fielddescription", "TR.F.IncomeStatement"])
    self._balance_sheet = rd.get_data(universe=[self._ticker], fields=["TR.F.BalanceSheet.fieldname","TR.F.BalanceSheet.fielddescription", "TR.F.BalanceSheet"])
    self._cash_flow = rd.get_data(universe=[self._ticker], fields=["TR.F.CashflowStatement.fieldname","TR.F.CashflowStatement.fielddescription", "TR.F.CashflowStatement"])

    self._wacc = self.wacc()
    # round all figures to 2 decimal places
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

  def prep(self):
    
    op_cf = rd.get_data(universe=[ticker], fields=["TR.F.CashFlowOpBefChgInWkgCap"])
    capex = rd.get_data(universe=[ticker], fields=["TR.F.CAPEXTot"])
    # CapEx as % of operating cash flow
    pcnts_capex_op_cf = capex.multiply(-1).div(op_cf)
    avg_pcnt = pcnts_capex_op_cf.mean()
    df_prep = [rd.get_data(universe=[ticker], fields=["TR.F.CashFlowOpBefChgInWkgCap"])]
    df_prep.loc['CapEx'] = capex.multiply(-1)
    for i in range(self._period):
      proj_op_cf = df_prep.iat[0, -1] * (1 + self._proj_growth[i])
      proj_capex = avg_pcnt * proj_op_cf
      df_prep[i + 1] = [proj_op_cf, proj_capex]
    df_prep.loc['Free Cash Flow'] = df_prep.sum()
    return df_prep

  def dcf(self):
    df_dcf = self.prep()
    wacc = self._wacc
    df_dcf.loc['Present Value of FCF'] = 0
    for i in range(self._period):
      curr_fcf = df_dcf.at['Free Cash Flow', i + 1] 
      df_dcf.at['Present Value of FCF', i + 1] = curr_fcf / (1 + wacc)**(i + 1)
    return df_dcf

  def share_price(self):
    df_dcf = self.dcf()
    wacc = self._wacc
    last_fcf = df_dcf.at['Free Cash Flow', self._period]
    tv = (last_fcf * (1 + self._tgr)) / (wacc - self._tgr)
    pv_tv = tv / (1 + wacc)**(self._period)
    enterprise_value = df_dcf.loc['Present Value of FCF'].sum() + pv_tv
    cash = self._balance_sheet.loc['Cash'][-1]
    debt = self._balance_sheet.loc['Long Term Debt'][-1]
    equity_value = enterprise_value + cash - debt
    shares = self._info.get('sharesOutstanding')
    share_price = equity_value / shares
    d = [tv, pv_tv, enterprise_value, cash, debt, equity_value, shares, 
         share_price]
    df_sp = pd.DataFrame(data=d, columns=['All numbers in dollars'])
    df_sp.index = ['Terminal Value', 'Present Value of Terminal Value',
                   'Enterprise Value', 'Cash', 'Debt', 'Equity Value', 'Shares',
                   'Implied Share Price']
    return df_sp

  def wacc(self):
    # treasury yield for risk free rate
    tnx = rd.get_history(universe="US5YT=RR", start="2013-04-02", end=date_time)
    rfr = tnx.info.get('previousClose') * 0.01
    # beta
    b = rd.get_data(universe=[ticker], fields=["TR.WACCBeta"])
    # equity risk premium
    erp = 0.056

    # calculate mean tax rate
    taxes = self._financials.loc['Income Tax Expense'].abs()
    ebit = self._financials.loc['Ebit']
    tax_rates = taxes.div(ebit)
    tc = tax_rates.mean()

    # calculate cost of equity
    cost_equity = rfr + b * (erp - rfr)
    # market value of equity (market capitalization)
    e = rd.get_data(universe=[self._ticker], fields=["TR.F.MktCap"])


    # calculate cost of debt
    interests = self._financials.loc['Interest Expense'].multiply(-1)
    debts = self._balance_sheet.loc['Long Term Debt']
    int_rates = interests.div(debts)
    avg_int_rate = int_rates.mean()
    cost_debt = avg_int_rate * (1 - tc)
    # market value of debt (most recent debt figure)
    d = debts[-1]

    # for ratios
    v = e + d

    # equation
    wacc = (e/v * cost_equity) + (d/v * cost_debt * (1 - tc))
    return wacc


if __name__ == "__main__":
    symbol = 'NPIFF'
    proj_growth = [0.15, 0.10, 0.05] # operating cash flow
    tgr = 0.025 # terminal growth rate
    stock = DCF(symbol, proj_growth, tgr)
    print(stock.wacc())
    print(stock.prep())
    print(stock.dcf())
    print(stock.share_price())


rd.close_session()

