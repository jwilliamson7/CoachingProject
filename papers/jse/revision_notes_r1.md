# JSE-26-0210 Round 1: literature check, venue costs, economics reframe

Compiled 2026-10-28. Reviews: `papers/jse/feedback/v1/` (gitignored). Downloaded papers:
`literature/jse_round1/` (PDFs gitignored). "Full text" = read in full (some via working-paper
versions); "abstract" = abstract and secondary descriptions only.

## A. Overlap verdict on our three novelty claims

| Claim | Verdict | What changes |
|---|---|---|
| (a) First to measure how foreseeable NFL head-coach FIRING is from hiring-day information | Holds in narrow form | Say "involuntary separation" and "NFL" explicitly. Cite neighbours that link pre-hire information to later *performance*: Skinner & Foreman 2025, Foreman et al. 2026, Roach 2016, Kilduff et al. 2016, Schuckers & Hayes 2025 (college, arXiv). Desai et al. 2018 is the closest *design* (hire-time information + Cox, EPL). |
| (b) Internal promotions are fired faster | New for the NFL, NOT new in sport | Desai et al. 2018 (EPL): outside hires have HR ~0.19, i.e. insiders ~5x hazard, Cox with in-tenure controls. Reword "direction unsettled" to "evidence conflicts across settings": outside CEOs dismissed sooner (Zhang 2008); EPL insiders dismissed sooner (Desai); college alumni coaches protected early (Holmes 2011). |
| (c) The pre-hire on-field record does not forecast firing | Holds, and is now better supported | Fee et al. 2006 (outside hires are selected on unit record, so the record is range-restricted among hires), Desai (prior top-job experience no effect), Holmes (career record no effect), Longley & Wong 2011 (minor-league stats weak even with the same task), Benson, Li & Shue 2019 (Peter Principle). |

## B. Errors in our current text found by the review

1. Line 70 cites `farnell2021` for "a largely internal promotion ladder". Neither Farnell version splits internal vs external promotion. Cite Fee, Hadlock & Pierce (2006) instead; Fee actually stresses substantial movement between teams.
2. Line 341 attributes "firing risk rises with age" to Farnell et al. True in the 2021 WP; in the 2025 version the baseline age effect is not significant (only when exits above 60 to 65 are dropped).
3. Lines 68 and 395: "an asymmetric-information account expects [internal tenure] to be longer". Desai et al. argue the opposite version: boards know *less* about an insider's ability in the top job, while outsiders often bring a visible top-job record. That predicts our direction. The theory paragraph must present both.
4. Line 395: "external hires are typically coordinators making the same step up". Our data: 34% of external hires had a prior NFL head-coach stint (stable across eras); Fee reports 55% (likely counting college head coaches). State the number; test the premium among first-time head coaches.
5. Update `farnell2021` to the 2025 version: Farnell, Berri, O'Sullivan & Simmons (2025), Maynooth WP N321-25 (the referee omitted O'Sullivan).
6. Line 303: "roughly 30% reduction in mean squared error" is measured against the 0.25 coin flip. Against Kaplan-Meier, skill is ~1.8% (non-nested) and ~0 (fully nested). Replace.
7. Blinding: the LinkedIn post "Predicting the Average Two-Year Win Probability and Hire Tenure of NFL Head Coach Hires: Three Approaches" appears in web searches on this topic.

## C. Paper-by-paper (referee 1's list, plus literatures referees 2 and 1 alluded to)

**Fee, Hadlock & Pierce (2006)**, J. Business 79(2) 821-850, doi:10.1086/499139. Full text (2002 SSRN WP). Not cited by us; should be.
- NFL head coaches and coordinators 1970-2001 (369 people); logits of promotion, demotion and head-coach separation.
- Outside promotion tracks the unit's record (25th to 75th pct of unit performance: 0.98% to 5.04% chance of an outside head job); team record adds nothing.
- When an opening occurs, poor team performance pushes toward OUTSIDE replacement (coef -4.13 vs -2.02, diff sig. at 1%). This cuts against "distressed franchises promote from within" and supports our selection probe.
- ~30% of head coaches came from the same team; 55% of outside hires were former head coaches.
- Staff continuity: 74% of coordinators leave with the head coach; 86% if an outsider replaces him, 46% if an insider does. Candidate mechanism: insiders inherit the failed regime's staff.
- Never compares survival of inside vs outside head coaches. (Check the 2006 published version; small residual risk.)

**Desai, Lockett & Paton (2018)**, Brit. J. Management 29(1) 26-42, doi:10.1111/1467-8551.12210. Full text (Warwick accepted manuscript).
- 164 new EPL managers 1996-2014, first season and the next only; Cox, manager-clustered SEs, season dummies; 67 dismissals; no competing risks.
- Outside hire coef -1.647 (HR ~0.19, p<0.01): insiders ~5x hazard, with in-tenure league position and form controlled.
- Predecessor left voluntarily: +0.875 (p<0.10). Longer-serving predecessor raises hazard. Prior top-job experience: no effect. Returning managers +1.638; former players of the club -1.261.
- Theory: information asymmetry about ability *in the top role*; Zhang 2008 and Shen & Cannella 2002 on the corporate side.

**Holmes (2011)**, JSE 12(2) 157-178, doi:10.1177/1527002510378820. Full text of the 2009 dissertation chapter; published version not read.
- Division I-A football 1983-2006, 2,507 coach-seasons, 196 performance firings; discrete-time logit hazard motivated by a Bayesian learning model (Beta-binomial).
- Current and lagged winning protect; the school's historical winning *raises* firing risk (expectations reference point). Hazard rises with tenure, levels off ~year 5.
- Alumni coaches protected in years 1-3 only. Career record, age, awards: no effect.
- No variable for an assistant promoted to head coach ("insider-ness" = alumni). A secondary source's race result differs from the dissertation's; read the published version.

**Seebruck & Savage (2020)**, Social Problems 67(3) 546-564, doi:10.1093/socpro/spz030. Abstract only.
- Every internal-promotion opportunity in D-I men's basketball 2008-2013, probits. Same-race assistant/head-coach pairs promoted more; internal promotion more likely when staff composition matches the departing head coach. Use for the selection discussion; supports adding predecessor exit type.

**Farnell, Berri, O'Sullivan & Simmons (2025)**, Maynooth WP N321-25. Full text.
- All NFL coaches 1989-2020 (1,259 coaches, 10,502 coach-years); head-coach exit logit with team FE.
- Exits driven by current win% (OR 0.001) and tenure (1.61). Black OR 0.718, n.s.; no Rooney Rule effect (2021's marginal Rule effect is gone). The race gap sits at offensive position-coach promotion (RRR 0.216).
- Never splits internal vs external. No overlap.

**Rider, Wade, Swaminathan & Schwab (2023)**, AJS 129(1) 227-275, doi:10.1086/725389. Full text (SSRN accepted version, read in browser, not saved).
- NFL coaches 1985-2015 (~1,317 coaches); exponential event-history models; Fine-Gray robustness; year and franchise FE; 52 performance metrics.
- White coaches promoted 1.72x as often; similar for internal (1.74x) and external (1.58x) promotion. Promotion to head coach not significant once current position is controlled. No glass cliff; no racial difference in first-season win%. Head-coach firing not modeled.

**Foreman, Soebbing, Seifried & Agyemang (2018)**, IJSM 19, pp. 313/315-338 (sources disagree; issue number unconfirmed). Full text of near-identical dissertation chapter.
- 12,536 coach-seasons 1984-2016; Black coaches less likely to be promoted, concentrated among position coaches; centrality raises promotion; no Rooney effect; no disparity at head-coach rank.

**Foreman, Skinner & Agyemang (2026)**, JSM advance online, doi:10.1123/jsm.2025-0066. Abstract only.
- Bridging social capital raises interviews; Black coaches who coordinated under minority head coaches get more jobs and later outperform. Pre-hire network predicts performance (not firing). Motivates a coaching-tree covariate.

**Skinner & Foreman (2025)**, Am. Behavioral Scientist, online first, doi:10.1177/00027642251366034. Abstract only. HIGHEST-PRIORITY LIBRARY PULL.
- NFL 2015-2022; 435 candidate observations (interviews, hires), "582 head coaches" for win% and against-the-spread models.
- Finds hiring practices misaligned with determinants of success; attributes it to herd behavior. Same question family as ours, on performance not firing. Pre-empts part of our practical message (lines 399, 403).
- Theory: herding (Banerjee 1992; Scharfstein & Stein 1990).

**Kilduff, Crossland, Tsai & Bowers (2016)**, AMJ 59(1) 352-375, doi:10.5465/amj.2014.0239. Authors' LSE summary only.
- NFL coaches 1980-2010; acolytes of celebrated head coaches 52% more likely to be promoted, most when little else is known; afterwards acolytes' next moves are more often demotions (54% vs 43%): ex post settling up (Fama 1980).
- A referee may ask why coaching-tree status is not a pre-hire feature.

**Longley & Wong (2011)**, MDE 32(3) 193-204, doi:10.1002/mde.1526. Abstract only.
- 1,200+ pitchers, ~20 years: minor-league stats have limited value for predicting MLB performance despite identical task and easy measurement. Strong analogue: the coordinator-to-head-coach step changes the task, so it should predict even less.

**Rooney Rule economics** (referee 2's uncited claim):
- Solow, Solow & Walker (2011), Labour Econ 18(3) 332-337: no race effect on coordinator-to-head-coach promotion; no Rule effect on minority head-coach counts.
- DuBois (2016), ALER 18(1) 208-233: DiD; minority candidates 19-21% more likely to fill NFL head-coach vacancies after the Rule.
- **Madden & Ruther (2011), JSE 12(2) 127-142**, doi:10.1177/1527002510379641 (full text of WP). Black head coaches outperformed in 1990-2002 and were fired more given performance (+0.892, t=2.15); both gaps vanished in 2003-09. No Black coordinator promoted internally to head coach through 2009.
- Pitts, Evans & Johnson (2022), Rev. Black Political Economy 51(1): Black coordinators less likely to become head coaches 2003-2020, especially from 2018.

**Succession and scapegoating** (referee 1's ter Weel comment): Grusky (1963) AJS; Gamson & Scotch (1964) AJS "ritual scapegoating"; Brown (1982) ASQ (NFL); Rowe et al. (2005) Leadership Quarterly (NHL); van Ours & van Tuijl (2016) Econ Inquiry (no causal gain from in-season dismissal, counterfactual design); **De Paola & Scoppa (2012) JSE 13(2)**; Besters, van Ours & van Tuijl (2016) De Economist; Soebbing & Washington (2011) JSM. Keep ter Weel; add van Ours & van Tuijl and De Paola & Scoppa; ground theory in Gamson & Scotch with Brown as the NFL case.

**GM change and payroll** (referee 2's uncited claims):
- Foster et al. (2024), Sport, Business and Management 14(1) 1-18 (open access): NFL 1987-2017, coach exits co-occur with GM exits (p~1.5e-5). "Hired alongside a new GM" is knowable at hire; GM change during tenure is post-hire.
- Fredrickson, Hambrick & Baumrin (1988) AMR: dismissal likelier when the board did not select the CEO.
- Payroll: Mondello & Maxcy (2009) Management Decision; Borghesi (2008) J. Econ & Business; Gregory-Smith (2021) Econ Inquiry (cap tied up in injured players). No peer-reviewed dead-money study found. Keefer (2017) JSE on sunk costs in NFL playing time.

**Other precedents to cite:** Bryson, Buraimo, Farnell & Simmons (2021) "Time To Go?" De Economist 169(1) 81-105 (head-coach quits and dismissals as competing risks, four European leagues); Benson, Li & Shue (2019) QJE (Peter Principle); Fama (1980) JPE; Holmström (1999) REStud.

**Race control, likely effect:** literature finds no race effect on firing given performance after 2003; omitted-variable bias on the internal-hire HR is probably small. But check internal x Black: several recent Black internal or interim-to-permanent hires were fired after about one season (agent recollection, verify: Hue Jackson 2011, Crennel 2012, Lovie Smith 2022, Pierce 2024, Mayo 2024).

**Library pulls (paywalled; abstract-level so far):** Skinner & Foreman 2025; Holmes 2011 (published); Fee et al. 2006 (published); Foreman et al. 2026; Seebruck & Savage 2020; Kilduff et al. 2016; Longley & Wong 2011; Madden & Ruther 2011 and Rider et al. 2023 (read, not saved); Bryson et al. 2021.

## D. Venue costs (checked 2026-10-28)

- **JSA**: now SAGE (from IOS Press, Jan 2025). APC $1,600, gold OA only, after acceptance. SAGE lists possible waivers. The UM-SAGE 10% discount needs a UM-affiliated corresponding author; Jon has graduated, so it does not apply: **$1,600**.
- **IJCSS**: IACSS society journal, EUR 320 APC (one aggregator says EUR 210, likely stale), OA only, no waivers. Scopus-indexed, CiteScore 2.6, reviews ~1 month.
- **JSE**: subscription journal, no APC on the standard route.

## F. Papers to download (no institutional access)

Save into `literature/jse_round1/` with the suggested filename. Free routes, best first: the free
copy listed; Google Scholar "All versions"; the Unpaywall browser extension; a ResearchGate or
email request to the corresponding author (usually answered within days); JSTOR free
"Register & Read"; a public-library account (databases or interlibrary loan).

| Pri | Paper | DOI | Have now | Best free route | Save as |
|---|---|---|---|---|---|
| 1 | Skinner & Foreman (2025), Am. Behavioral Scientist | 10.1177/00027642251366034 | abstract | Email author (Foreman) or ResearchGate | skinner2025_herd.pdf |
| 1 | Holmes (2011), JSE 12(2) | 10.1177/1527002510378820 | 2009 dissertation | Author request; published race result differs | holmes2011_published.pdf |
| 1 | Fee, Hadlock & Pierce (2006), J. Business 79(2) | 10.1086/499139 | 2002 SSRN WP | JSTOR Register & Read (J. Business is on JSTOR) | fee2006_published.pdf |
| 2 | Madden & Ruther (2011), JSE 12(2) | 10.1177/1527002510379641 | read, not saved | Penn repository working paper (free) | madden2011_rooney.pdf |
| 2 | Rider et al. (2023), AJS 129(1) | 10.1086/725389 | read, not saved | SSRN accepted version (free) | rider2023_valuative.pdf |
| 2 | Foreman, Skinner & Agyemang (2026), JSM | 10.1123/jsm.2025-0066 | abstract | Author request | foreman2026_socialcapital.pdf |
| 2 | Bryson, Buraimo, Farnell & Simmons (2021), De Economist 169(1) | search title "Time To Go?" | none | Check Springer page for open access; else author | bryson2021_timetogo.pdf |
| 2 | van Ours & van Tuijl (2016), Economic Inquiry 54(1) | 10.1111/ecin.12280 | none | IZA/CEPR working-paper version | vanours2016_inseason.pdf |
| 2 | De Paola & Scoppa (2012), JSE 13(2) | search title | none | RePEc working-paper version | depaola2012_coach.pdf |
| 3 | Kilduff et al. (2016), AMJ 59(1) | 10.5465/amj.2014.0239 | LSE blog summary | Author request | kilduff2016_acolyte.pdf |
| 3 | Seebruck & Savage (2020), Social Problems 67(3) | 10.1093/socpro/spz030 | abstract | Author request | seebruck2020_promote.pdf |
| 3 | Longley & Wong (2011), MDE 32(3) | 10.1002/mde.1526 | abstract | Author request | longley2011_minors.pdf |
| 3 | Foster et al. (2024), Sport, Business & Mgmt 14(1) | 10.1108/SBM-09-2022-0081 | none | Open access (CC BY) | foster2024_gm.pdf |
| 3 | Benson, Li & Shue (2019), QJE 134(4) | search title "Promotions and the Peter Principle" | none | NBER working paper | benson2019_peter.pdf |

Already saved: desai2018_dismissal.pdf (accepted manuscript), fee2006 WP, holmes2011 dissertation,
farnell2021 and farnell2025 WPs, kilduff2016 LSE summary.

## E. Economics reframe: outline

### E1. The economic question (replaces "limits of foresight")
In labor economics a match's value is partly a *search good* (assessable before hiring) and partly an *experience good* (revealed only by production). The NFL is an extreme case on the search side: a candidate's prior output is public, charted and standardized. We ask (i) how much of the duration of a head-coach match is foreseeable when it is formed, and (ii) whether *how* it is formed, promotion from within vs an outside hire, changes how long it lasts, and why.

Candidate titles: "Hiring From Within and the Limits of Foresight: Employer Learning in the NFL Head-Coach Market"; "What Teams Can Know When They Hire: Internal Promotion and the Duration of NFL Head-Coaching Matches".

### E2. Why time-to-firing (answers "why not wins" and "just regress duration")
- Firing is the employer's own verdict on the match *relative to its aspirations* (Holmes 2011's expectations reference point; Allen & Chadwick's relative performance). Win% measures output, not value against what this team expected. This turns the JQAS AE's objection ("success differs by team") into the justification.
- 92% of completed spells end in firing, so the firing hazard *is* the duration regression R2 asks for; voluntary exits are a competing risk (the JQAS AE's other objection).
- Keep S5 (tenure tracks win%) in the main text as one sentence plus a supplement figure.

### E3. Conceptual framework (new, ~1.5 pages, no new theory needed)
A Holmes-style learning model: head-coach ability in the role, theta, has a hiring-day prior whose mean and precision depend on the hiring channel j (internal/external) and observables. Each season the team observes a noisy signal; it fires when the posterior falls below a reservation value r_j. Predictions:
- **P1 Ceiling.** If hires are already selected on the public record (Fee 2006), the record is range-restricted among hires and its residual information is small relative to on-the-job signals: low concordance, and the unit block near chance. Peter Principle (Benson, Li & Shue 2019) and Longley & Wong 2011 for non-transfer.
- **P2 Channel via match quality.** Chan (1996) tilted contest: outsiders hired only if distinctly better, so mu_ext > mu_int. Desai et al. (2018) top-job information asymmetry: insiders' top-job ability is less known, so prior variance is higher, giving more early firings. Both predict the internal premium is *mediated by in-tenure performance* and concentrated early.
- **P3 Channel via the firing threshold.** Lower patience for insiders at equal performance: smaller sunk search cost and contract guarantees (escalation of commitment; Keefer 2017 JSE on sunk costs in the NFL), weaker owner allegiance (Fredrickson et al. 1988), blame-sharing for consensus outside hires (Scharfstein & Stein 1990). Predicts the premium *persists conditional on in-tenure performance*.
- **P4 Succession context.** An insider who replaces a fired coach inherits the failed regime's staff (Fee: coordinators retained 54% under insiders vs 14% under outsiders). Predicts the premium is concentrated in internal promotions after a firing; Desai finds a voluntary predecessor exit raises hazard.
- **P5 Selection works against us.** Fee shows poor teams lean toward outside hires, so distress-driven selection would bias *against* finding the premium. Firm-specific capital (Lazear 2009) predicts the opposite sign; our estimate rejects it.

### E4. Tests that separate the mechanisms (mostly existing data)
| Test | Separates | Data |
|---|---|---|
| In-tenure win%/SRS as time-varying covariate | P2 (mediated) vs P3 (persists) | Have it (Teams/) |
| Predecessor exit type x internal | P4 | Have it (event labels) |
| First-time head coaches only / control prior HC experience | Chan vs Desai top-job record | Have it |
| Hazard by tenure year for internal vs external | P2 early concentration | Have it |
| Franchise-stratified Cox | Persistent franchise churn | Have it; fix 4 miscoded hires first |
| Hired alongside a new GM | Owner/GM allegiance (P3) | Hand-code |
| Race, with pre/post-2003 interaction; internal x Black | Literature control; recent-era confound | Hand-code |

### E5. Paper structure (JSE style)
1. Introduction: economic question first; three findings; one paragraph on why firing duration.
2. Institutional background: how hires happen, the internal ladder (Fee), Rooney Rule, owner/GM roles, turnover (22.3% of teams per season, 1970-2025).
3. Conceptual framework (E3), ending in P1-P5.
4. Data: sample and event coding, variable definitions in the main text (internal-hire rule with R2's example), missingness (24% of hires have no prior NFL unit seasons), descriptive table by internal vs external.
5. Empirical strategy: cause-specific Cox, coach clustering, franchise strata, era strata (1994 cap, 2003 Rooney Rule); predictability measured as concordance and Brier skill relative to Kaplan-Meier.
6. Results: 6.1 how much is foreseeable; 6.2 the hiring channel; 6.3 mechanisms (E4); 6.4 robustness (frailty, observed-unit subsample or missing indicator, post-2003, recoded hires, corrected boundary labels, Fine-Gray, race).
7. Implications for franchises (hold internal promotions to a higher bar; staff continuity) and for candidates (insiders face shorter expected tenure: negotiate guarantees).
8. Conclusion.
Supplement: stability selection, model bake-off, Nadeau-Bengio, feature dictionary, event coding.

Standard-format tables: columns adding blocks (channel only; + credentials; + inherited situation; + franchise strata; + era; + race), with N, events, clusters, log-likelihood and FE rows; long descriptions in table notes.

### E6. Data and text fixes required regardless of venue
- Hand-code terminal outcomes for the 32 boundary stints (no re-crawl).
- Recode the 4 outside midseason hires coded internal (Coryell 1979, Levy 1987, Berry 1985, Meyer 1987); report both codings.
- Disclose and handle structural missingness of the unit block.
- Fix B1 to B7 above.
