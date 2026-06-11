from __future__ import annotations

import unittest

from src.fetchers.taifex_fetcher import TaifexFetcher


class TaifexFetcherTest(unittest.TestCase):
    def test_selects_highest_volume_tx_contract(self):
        content = (
            "交易日期,契約,到期月份(週別),開盤價,最高價,最低價,收盤價,成交量\n"
            "20260601,TX,202606,22000,22100,21900,22050,100000\n"
            "20260601,TX,202607,22020,22120,21920,22070,5000\n"
            "20260601,MTX,202606,22000,22100,21900,22050,1000\n"
        )

        result = TaifexFetcher().parse_futures_csv(content)

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "symbol"], "TX")
        self.assertEqual(result.loc[0, "close"], 22050)
        self.assertEqual(result.loc[0, "volume"], 100000)

    def test_parses_official_put_call_ratio_table(self):
        html = """
        <table>
          <tr>
            <th>日期</th><th>賣權成交量</th><th>買權成交量</th>
            <th>買賣權成交量比率%</th><th>賣權未平倉量</th>
            <th>買權未平倉量</th><th>買賣權未平倉量比率%</th>
          </tr>
          <tr>
            <td>2026/6/11</td><td>225,901</td><td>214,908</td>
            <td>105.12</td><td>132,390</td><td>99,293</td><td>133.33</td>
          </tr>
        </table>
        """

        result = TaifexFetcher().parse_put_call_ratio_html(html)

        self.assertEqual(len(result), 2)
        values = dict(zip(result["dataset"], result["value"]))
        self.assertAlmostEqual(values["put_call_volume_ratio"], 1.0512)
        self.assertAlmostEqual(values["put_call_open_interest_ratio"], 1.3333)
        self.assertTrue((result["source"] == "taifex").all())


if __name__ == "__main__":
    unittest.main()
