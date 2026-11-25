# Labelling guidelines

These rules are used by both the LLM (they're inserted into its prompt) and by you during manual
verification, so both label against the same definition. If you change them, re-run the labelling.

## Task

Label each text with the **short-term sentiment for a current shareholder of the named company**.
For general market or economy news with no single company, judge it for PSX investors in general.

## Labels

**positive**: likely good news for shareholders.
- Profit or EPS up versus the comparable previous period
- Cash dividend, bonus shares, or a higher payout than before
- New contracts, expansion, capacity increase, credit rating upgrade
- Favourable court or regulatory decisions, debt successfully restructured
- For market news: index gains, rate cuts, inflows, improving macro data

**negative**: likely bad news for shareholders.
- Loss, or profit/EPS down versus the comparable previous period
- Dividend cut or skipped when one was paid in the same period last year
- Plant shutdown, default, penalties, fines, delisting or suspension notices
- Credit rating downgrade, major litigation against the company, key resignations amid problems
- For market news: index falls, rate hikes, outflows, worsening macro data

**neutral**: routine, procedural or unclear impact.
- Notices of AGM/EGM, board meeting dates, book closures, transmission of reports
- Routine compliance filings, changes of address or registrar
- Results with no comparison figures and no clear direction
- Mixed news where the good and bad roughly balance
- Not enough information to judge

## Tie-breakers

1. Judge only from the text given. Don't use outside knowledge of later price moves.
2. If figures are given, the direction of profit/EPS versus last period decides.
3. When genuinely unsure between neutral and another label, choose **neutral**.
4. The text may contain OCR errors from scanned documents. Read past obvious misread characters.
