# LLM label quality

Human-verified items (blind): **291**

- Accuracy (LLM vs human): **93.5% ± 2.8%** (95% CI)
- Cohen's kappa: **0.846**

## Confusion matrix

~~~
                 llm: positive  llm: negative  llm: neutral
human: positive             37              0             4
human: negative              1             30             7
human: neutral               5              2           205
~~~

## Per-class (human = truth)

~~~
              precision    recall  f1-score   support

    positive       0.86      0.90      0.88        41
    negative       0.94      0.79      0.86        38
     neutral       0.95      0.97      0.96       212

    accuracy                           0.93       291
   macro avg       0.92      0.89      0.90       291
weighted avg       0.94      0.93      0.93       291

~~~

## Accuracy by LLM self-reported confidence

~~~
                accuracy    n
confidence_bin
(0.6, 0.8]         0.922  129
(0.8, 0.9]         0.991  112
(0.9, 1.0]         1.000   33
(-0.001, 0.6]      0.529   17
~~~

## Accuracy by event type (LLM's tag)

~~~
                       accuracy   n
event_type
financial_results         0.838  80
regulatory_compliance     0.947  57
agm_egm                   0.974  38
board_meeting             1.000  37
dividend_payout           0.952  21
corporate_action          1.000  18
management_change         1.000  16
material_information      1.000  12
other                     1.000  10
market_news               0.500   2
~~~

## Accuracy by text source

~~~
             accuracy    n
text_source
pdf             0.971  139
ocr             0.895  133
title_only      0.947   19
~~~
