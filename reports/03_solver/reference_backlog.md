# Reference Backlog

This backlog records sources and citation decisions that still need human or advisor review after
the solver-validation reference integration pass. Items in this file were not guessed into the
shared bibliography when metadata was uncertain or when the source was not essential to the
current solver-validation reports.

## Metadata Requiring Human Review

- Brennan-Schwartz American put article: Crossref returned a verified JSTOR record for
  `10.2307/2326779` with authors Michael J. Brennan and Eduardo S. Schwartz, and a separate Wiley
  record for `10.1111/j.1540-6261.1977.tb03284.x` whose author metadata includes Robert C.
  Merton. The shared bibliography uses the verified JSTOR record and does not rely on the
  conflicting Wiley author metadata. An advisor may prefer a publisher-specific DOI after checking
  the journal archive directly.
- Merton rational option pricing page range: Crossref verified the article DOI and starting page
  for `10.2307/3003143`. The shared bibliography does not expand the page range beyond verified
  metadata.
- Cottle, Pang, and Stone LCP reference: Crossref verified the SIAM electronic record with DOI
  `10.1137/1.9780898719000` and year 2009. If the project needs the original print-year citation,
  verify the original edition metadata separately before changing the entry key or year.
- Tavella and Randall finite-difference option-pricing text: Crossref search surfaced a book
  review rather than a verified book record. The source was considered but not added.

## Sources Considered but Not Added

- Additional computational finance textbooks beyond Wilmott, Howison, and Dewynne were not added
  because the current reports only need light theory grounding rather than a full literature
  review.
- Longstaff-Schwartz, Haugh-Kogan, PINN, DGM, deep American option, and neural free-boundary
  sources are included in `references.bib` because their metadata was verified and they support
  future-work context, but they are cited sparingly. A later literature-review pass should decide
  whether each belongs in the final paper.
- Recent fractional Black-Scholes, jump-diffusion, and process-informed neural-network leads from
  the literature map were not added because the solver-validation reports do not need them yet.

## Citation Placement Requiring Advisor Review

- Ticket reports now cite external sources in theory or methodology sections only. An advisor may
  want denser citation coverage in a future paper-style literature review.
- Boundary extraction and Greek diagnostics remain implementation diagnostics. The added citations
  ground the numerical ideas, but they do not turn threshold-based boundaries or finite-difference
  Greeks into production-grade estimates.
- Ticket 01 still contains historical raw git-status text from its original study report. This pass
  did not clean unrelated prose because the scope was reference integration only.

## Open Follow-Up

- Decide whether the final research paper should use a different citation style than `plainnat`.
- Decide whether to replace study-report bibliographies with a consolidated final bibliography in
  a later paper/report assembly step.
- Verify any additional sources requested by the project supervisor before adding them to
  `references.bib`.
