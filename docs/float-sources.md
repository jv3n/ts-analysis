# Where to get a stock's float

As of 2026-11-05. Inventory for [#30](https://github.com/jv3n/ts-analysis/issues/30): the float of
each picked ticker **as it was known on the day**, not today's. Today's float on a past day hides the
dilution that came since.

- **Float**: the shares free to trade (shares outstanding minus insiders, large holders, locked-up
  shares). Every provider computes it its own way.
- **Shares outstanding**: every share issued. Reported by the issuer, so dated and checkable, but
  larger than the float.
- What the POC needs: a **dated value**, a **history**, **API access** usable locally, and a licence
  that allows it.

## At a glance

No free source gives a **float history in shares**. The free sources with a history give **shares
outstanding**; a dated float through an API is paid (Ortex, sec-api.io), or current only (Massive,
Alpha Vantage, EODHD).

| Source | Gives | Dated history | Access | Price |
| --- | --- | --- | --- | --- |
| [SEC EDGAR XBRL](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | Shares outstanding (shares), public float (USD) | Yes, with filing date, from ~2009–2011 | API, bulk zip | Free |
| [Massive ticker overview](https://massive.com/docs/rest/stocks/tickers/ticker-overview) | Shares outstanding | Yes through `date`, 2 years on Basic | API | In Basic |
| [Massive float](https://massive.com/docs/rest/stocks/fundamentals/float.md) | Float (shares, %) | No, current value | API | In Basic |
| [Alpha Vantage](https://www.alphavantage.co/documentation/) | Current float, quarterly shares outstanding | Shares outstanding from 2008, no filing date | API | Free 25 req/day |
| [Sharadar SF1](https://sharadar.com/docs/fundamentals) | Shares outstanding | Yes, indexed to the filing date, delisted included | API | $19–39/month |
| [EODHD](https://eodhd.com/financial-apis/stock-etfs-fundamental-data-feeds) | Current float, quarterly shares outstanding | Shares outstanding only | API | $49.99–59.99/month |
| FMP | Float, historical float | To check | API | To check |
| [sec-api.io](https://sec-api.io/docs/outstanding-shares-float-api) | Shares outstanding, public float (USD) | Yes, with filing time, from 2011 | API | From $199/month |
| [Ortex](https://docs.ortex.com/reference/stock_reference_free_float_retrieve.md) | Float (shares, %) | Yes, `point_in_time` option | API, bulk | Credits, price to check |
| [Finviz](https://finviz.com/elite) | Float, shares outstanding | No | Site, Elite CSV export | Elite $39.50/month |
| [Yahoo Finance](https://finance.yahoo.com/quote/AAPL/key-statistics/) | Float, shares outstanding | No | Site only | Free |
| [StockAnalysis](https://stockanalysis.com/stocks/aapl/statistics/) | Float, shares outstanding | No | Site only | Free |
| [TradingView](https://in.tradingview.com/support/solutions/43000564727) | Yearly float, shares outstanding | Inside Pine only | Pine Script, no API | Per subscription |

## Free primary sources

**[SEC EDGAR XBRL](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)** is
the best free dated source: every value carries its filing date (`filed`), so we know when it was
known.

- `dei:EntityCommonStockSharesOutstanding`: shares outstanding, from the cover of every 10-Q and 10-K,
  so quarterly. The count is dated a few days before the filing (e.g. as of 2026-10-23, filed
  2026-11-03).
- `dei:EntityPublicFloat`: the public float in **dollars**, not shares, once a year in the 10-K,
  measured at the last day of Q2. Too old for a small cap that dilutes.
- Endpoints: `companyconcept`, `companyfacts`, `frames` on `data.sec.gov`, and `companyfacts.zip`
  rebuilt every night. No key.
- Rules ([FAQ](https://www.sec.gov/os/webmaster-faq)): 10 requests a second at most, and a declared
  User-Agent (a name and a contact e-mail) is required.
- Limit: between two quarterly filings, an offering does not show. The 424B and 8-K filings announce
  it, but without a structured share count.

**OTC Markets** shows, on each stock's page, the authorized, outstanding and restricted shares and the
float, each with its own date. The pages are rendered in JavaScript and its internal API refused
access: nothing could be checked beyond search snippets. It mostly covers OTC names, not the Nasdaq
and NYSE names we pick.

## Market data APIs

**Massive**, our provider already, has two useful endpoints, both in the Basic plan:

- [Ticker overview](https://massive.com/docs/rest/stocks/tickers/ticker-overview)
  (`/v3/reference/tickers/{ticker}`): `share_class_shares_outstanding` and
  `weighted_shares_outstanding`, no float. The `date` parameter gives the value "as of that date",
  2 years back on Basic. Careful: for SEC data the date is set against the report's period end, not
  its filing date, so a value can show before it was published.
- [Float](https://massive.com/docs/rest/stocks/fundamentals/float.md) (`/stocks/vX/float`,
  experimental since 2025-12-23): `free_float`, `free_float_percent`, `effective_date`, up to 5,000
  tickers a page. **Current value only**:
  [per Massive](https://massive.com/knowledge-base/article/does-massive-have-historical-float-data),
  there is no history, and its advice is to poll it daily to build your own.

**Other APIs**, cheapest first:

| Service | Float | Dated shares outstanding | Price |
| --- | --- | --- | --- |
| [Alpha Vantage](https://www.alphavantage.co/premium/) | Current `SharesFloat` (`OVERVIEW`) | Quarterly `SHARES_OUTSTANDING`, no filing date | Free 25 req/day, from $49.99/month |
| [Finnhub](https://finnhub.io/docs/api/company-profile2) | Not found | Current `shareOutstanding`, unit undocumented | Free, limits to check |
| [Sharadar SF1](https://sharadar.com/subscribe) | No | Yes, indexed to the filing date, from the 1990s, delisted included | $19–39/month |
| [EODHD](https://eodhd.com/pricing) | Current `SharesFloat` | Yearly and quarterly | Not in the free plan, $49.99–59.99/month |
| FMP | Float and historical float announced | To check | To check, docs unreachable |
| [Benzinga](https://docs.benzinga.com/ar/api-reference/fundamentals/get-share-class-profile) | Not documented | `sharesOutstanding` with `asOf` | To check |
| [Intrinio](https://docs.intrinio.com/documentation/javascript/shares_outstanding_by_company_v2) | A public float endpoint exists | From the 10-K and 10-Q | To check |
| [sec-api.io](https://sec-api.io/pricing) | Public float in USD | Yes, filing time, from 2011 | 100 free calls, then from $199/month |
| [Ortex](https://docs.ortex.com/reference/stock_reference_free_float_retrieve.md) | Float in shares and %, one row per change | Yes, `point_in_time` option | Credits, plan prices to check |

Ortex is the only checked API that gives a **float history in shares, dated from when it was known**.
FMP announces one too, but its documentation refused access: to check on the site directly.

## Sites to look up by hand

Useful to check a ticker by eye, not to feed the POC: they all give today's value, with no history
and no official API.

| Site | Shows | Stated source |
| --- | --- | --- |
| [Finviz](https://finviz.com/quote.ashx?t=AAPL) | `Shs Float`, `Shs Outstand`, `Insider Own`; CSV export with [Elite](https://finviz.com/elite) ($39.50/month) | Not stated |
| [Yahoo Finance](https://finance.yahoo.com/quote/AAPL/key-statistics/) | Float, Shares Outstanding, Implied Shares Outstanding | Latest filed report, for shares outstanding |
| [StockAnalysis](https://stockanalysis.com/stocks/aapl/statistics/) | Float, shares outstanding, yearly and quarterly change | S&P Global Market Intelligence |
| [Fintel](https://fintel.io/so/us/aapl) | Shares outstanding behind a paywall, no float shown | Capital IQ |
| [TradingView](https://in.tradingview.com/support/solutions/43000564727) | Yearly `FLOAT_SHARES_OUTSTANDING` in Pine Script | Not stated |
| [Koyfin](https://www.koyfin.com/pricing) | Fundamentals, free to $299/month | Not stated |

Not checkable (robots refused): MarketWatch and Barron's, Dilution Tracker (often cited for small-cap
dilution; Trade-Ideas uses it as a source), Macrotrends, and Interactive Brokers' fundamentals.

## Pitfalls

- **Each its own float.** Massive leaves out insiders, holders above 5 % and government stakes, but
  keeps funds and ETFs; Yahoo removes restricted stock. On AAPL, StockAnalysis shows a float (14.57 B)
  **larger** than the shares outstanding (14.53 B). Two sources never mix in one measure.
- **Today's value reads the future.** Today's float on a day in May hides every dilution that came
  since, which is precisely what sinks a small cap after a gap.
- **Period date against filing date.** A value dated at its period end (Massive with `date`, Alpha
  Vantage) can be used before it was published. Cut on the filing date: SEC `filed`, Sharadar,
  sec-api.io `reportedAt`, Ortex `point_in_time`.
- **Quarterly is slow.** Between two 10-Qs, an offering (424B, 8-K) can double the share count. SEC
  shares outstanding alone miss those jumps.
- **Licences.** Finviz, Yahoo, StockAnalysis and the other sites have no open API; scraping them
  generally breaks their terms. Paid plans often limit redistribution (sec-api.io: Enterprise to
  redistribute). For local use, the official APIs are enough.

## Recommendation

Start free, with the SEC's dated shares outstanding, and pay for a float history only if splitting
the score by float shows something.

1. **The past: SEC EDGAR.** For each picked day, take `EntityCommonStockSharesOutstanding` from the
   last 10-Q or 10-K **filed before** that day. Free, dated by filing, and it covers the six months of
   the backfill.
2. **From now on: Massive float, every day.** Add to `ts-fetch` a daily call to `/stocks/vX/float`
   (5,000 tickers a page, so a handful of requests) and store the value: a real float history from now
   on, for free.
3. **If the score by float bucket reacts: Ortex or FMP.** A point-in-time float history, to cover the
   past in tradable shares. Check Ortex's price and FMP's real access first.

Open:

- [ ] Check on the Massive account that `/stocks/vX/float` answers on Basic
- [ ] Pick the float buckets to split the score by (e.g. under 5 M, 5–20 M, over 20 M)
- [ ] Carry these choices into [#30](https://github.com/jv3n/ts-analysis/issues/30)
