# Clustering report

Training texts: **2245**, chosen **k = 10**

- Silhouette at k=10: **0.267**
- Agreement with LLM event types: ARI **0.577**, NMI **0.725**
- PCA: 2 components explain **32.1%** of the variance

## k scan

~~~
 k  inertia  silhouette
 2 1208.289       0.217
 3 1090.723       0.257
 4 1011.164       0.210
 5  937.657       0.234
 6  896.634       0.237
 7  854.378       0.250
 8  815.276       0.245
 9  784.186       0.256
10  753.787       0.267
11  730.064       0.270
12  710.775       0.284
13  693.807       0.282
14  674.102       0.294
15  667.557       0.290
~~~

## Top title words per cluster

~~~
cluster  0 (  96 rows): appointment, director, directors, chief, change, officer, election, resignation
cluster  1 ( 348 rows): year, results, financial, ended, june, half, fund, period
cluster  2 ( 160 rows): information, material, disclosure, miscellaneous, form, revoked, limited, industries
cluster  3 ( 176 rows): board, meeting, progress, notice, closed, directors, period, rescheduled
cluster  4 ( 242 rows): general, meeting, annual, notice, resolutions, passed, certified, copy
cluster  5 ( 123 rows): dividend, cash, credit, interim, final, notice, daily, book
cluster  6 ( 291 rows): transmission, report, quarterly, period, ended, annual, year, june
cluster  7 ( 368 rows): quarter, results, financial, ended, september, june, sep, audited
cluster  8 ( 350 rows): limited, shares, right, disclosure, company, notice, movement, unusual
cluster  9 (  91 rows): briefing, corporate, session, notice, presentation, recording, video, sessions
~~~

## Cluster vs event type

~~~
event_type  agm_egm  board_meeting  corporate_action  dividend_payout  financial_results  management_change  market_news  material_information  other  regulatory_compliance
cluster
0                19              1                 1                0                  2                 67            0                     1      1                      4
1                 1              5                 0                0                336                  0            0                     0      0                      6
2                 5              0                24                1                  1                  9            0                   118      0                      2
3                 0            174                 0                0                  0                  1            0                     0      0                      1
4               239              3                 0                0                  0                  0            0                     0      0                      0
5                 1              0                 1              112                  0                  0            0                     0      0                      9
6                 1              0                 0                0                  4                  0            0                     0      1                    285
7                 0              0                 0                0                363                  0            0                     0      0                      5
8                50             12               111                1                 10                  1           22                    16     13                    114
9                 0              0                 0                0                  0                  0            0                     0     91                      0
~~~

## Cluster vs label

~~~
sentiment  negative  neutral  positive
cluster
0                 0       95         1
1               129       80       139
2                 8       99        53
3                 0      176         0
4                 0      242         0
5                 0      106        17
6                 0      289         2
7               152       67       149
8                 8      330        12
9                 2       86         3
~~~
