# Feature inventory and audit (2026-10-28)

Scope: every modeled column of `data/master_data.csv` (columns 2 to -2): **102 columns**
(51 base from `create_data.py`, 51 engineered in `engineer_career_features.py`). The
redundancy prune (`survival_methods.DROP_REDUNDANT_FEATURES`) removes 21, leaving **81
candidate features** for selection. The submitted supplement documents 74 candidates; the
difference is the 7 structural indicators added in this revision. Every definition below
was checked against the code that computes it. Per-column statistics (coverage, range,
outliers, strongest correlate) were computed on the 364-stint build.

## 1. Findings

### Fixed in code
1. **Roster retention was wrong at the source.** Coach_WAR's position-group retention
   mapped only generic position codes, so starters' side-specific codes (RDE, LCB, LOLB,
   RILB, ...) fell into an unmapped group, and a player counted as retained only inside
   the same group. Result: false 0% values (242 team-seasons for the defensive line, 194
   for corners, 117 for linebackers, 76 for the offensive line). Example: the 1984->85
   Vikings show 0% in all four defensive groups, but 13 of 29 1984 defenders were on the
   1985 roster (45%). Retention is now computed here from the raw PFR rosters
   (`roster_retention`): a player is retained if he is on the next season's roster at any
   position; his earlier position only assigns the offense/defense split. (Correction to
   an earlier note: the `_crosstab` columns in Coach_WAR's combined file are STARTERS'
   retention, picked up by a merge suffix, not a second computation of roster retention;
   their low correlation with roster retention is expected.)
   **Fixed in Coach_WAR too (uncommitted):** completed position maps in
   `calculate_roster_turnover_crosstab.py` and `calculate_starters_turnover_crosstab.py`
   (the latter lacked LDT/RDT and LLB/RLB), report an empty prior-season group as missing
   rather than 0%, and regenerated the turnover files and `combined_final_dataset.csv`.
   Roster-retention zeros fell from 242 to 1 (DL), 194 to 2 (CB), 117 to 2 (LB), 76 to 0
   (OL); starters' zeros from 384 to 122 (DL) and 664 to 151 (LB). Only turnover columns
   changed. **Coach_WAR's own models must be rerun to pick this up.**
2. **`hire_roster_turnover` was a retention rate under a turnover label** (it averaged
   retention percentages; higher meant MORE players kept). Renamed `hire_roster_retention`;
   the supplement's "share of the roster turned over" had the sign backwards.
3. **Premium draft window.** First-round picks used the three drafts before the hire
   (Y-1, Y-2, Y-3) but premium picks used Y-2, Y-3, Y-4, skipping the most recent draft.
   Both now use Y-1..Y-3.

### Verified, no change
- **Negative starting-QB value** (5 hires, minimum -3) is real: PFR's raw AV is negative
  for a few very poor QB seasons (Hoying 1998 Eagles -3, Bob Lee 1974 Falcons -3,
  Skelton 2012 Cardinals -2).
- **Outliers beyond 5 SD are all genuine:** predecessor tenure 29 and 26 (Landry and
  Shula, for Jimmy Johnson's two hires) and 24 (Belichick, for Mayo); 18 years at the
  hiring organization (Burns, Vikings 1968-85); 25 NFL position-coach seasons (Culley);
  gaps of 14 years (Vermeil 1997, Wilkinson 1978).
- No count or percentage falls outside its possible range; negatives occur only in
  z-scores, SRS values, trajectories and the QB AV above.
- No candidate is constant; all 0/1 columns are intended indicators.

### Supplement definitions that are wrong (fix in the revised supplement)
| Supplement entry | Supplement says | Code actually measures |
|---|---|---|
| Prior OC / DC / position coach / head coach | "ever" held the role | the coach's **most recent** prior role (`cf_pre_*`) |
| Prior offensive-side role | any prior offensive experience | most recent role was on offense |
| Prior NFL experience | any prior NFL coaching | most recent role was at the NFL (or AFL) level |
| Roster value, Starter value | total AV | **average** AV per player / per starter |
| Starting-QB value | AV of the starting QB | average AV of the season's listed starting QB(s) |
| Turnovers forced / committed | turnovers over two seasons | two-season mean of season **z-scores** |
| Roster turnover | share turned over | now `hire_roster_retention`, share **retained** |
| Years at hiring org | seasons spent there | **consecutive** seasons immediately before the hire |
| Premium draft capital | prior three drafts | now true (was Y-2..Y-4) |

### Definitions that changed with this revision's data policies
- NFL experience counts, NFL employers, NFL share, age at first NFL job, ever NFL OC/DC,
  prior HC stints: the AFL (1960-69) now counts as NFL-level.
- Age at first NFL job: equals age at hire when the hire is the coach's first NFL job.
- Recent-form features and ranked seasons exclude interim titles.
- Unit trajectory: set to 0 (and flagged) when only one ranked season exists.
- Predecessor tenure: the prior season's season-opening coach; the 1996 Ravens inherit
  the 1995 Browns; the 1999 Browns inherit nothing.
- Internal hire: midseason takeovers are coded by insider status (four outside
  appointments recoded external; `analysis/midseason_takeovers_summary.md`).
- Inherited-situation features for former AFL franchises use their AFL seasons.

### Redundancy among the 81 candidates (|r| >= 0.85)
Prior HC stints ~ NFL HC years (0.86, kept on purpose: stints vs. accumulated years);
turnovers ~ turnover rate (0.86); passing yards ~ passing first downs (0.90); rush
attempts ~ rushing yards (0.87). Retention overall ~ offense/defense: 0.81 / 0.83.

## 2. Inventory

Columns marked *pruned* are in the data but not selection candidates. "Flag" names the
structural indicator that covers the column's missing values (no column has gaps that
are not covered by one).

### Core experience (base, 8; all candidates)
| Column | Definition |
|---|---|
| Feature 1 `age` | Age at hire (as of the anchored hire season) |
| Feature 2 `num_times_hc` | Prior NFL/AFL head-coaching stints |
| Feature 3 `num_yr_col_pos` | College position-coach seasons |
| Feature 4 `num_yr_col_coor` | College coordinator seasons |
| Feature 5 `num_yr_col_hc` | College head-coach seasons |
| Feature 6 `num_yr_nfl_pos` | NFL/AFL position-coach seasons |
| Feature 7 `num_yr_nfl_coor` | NFL/AFL coordinator seasons, special teams included |
| Feature 8 `num_yr_nfl_hc` | NFL/AFL non-interim head-coach seasons (a re-anchored takeover's partial season counts) |

### Pooled prior-unit performance (base, 33; 26 candidates)
Each statistic is standardized within its season (and within league before 1970), signed
so larger is better, and pooled over the coach's prior OC, DC and non-interim HC seasons
(HC seasons contribute both sides). Flags: `cf_no_unit_record` (all 33 missing) and
`cf_no_drive_record` (the 13 drive and situational statistics, recorded from ~1999).
*Pruned:* points scored, total yards, interceptions, net yards per attempt, rushing first
downs, penalty yards, points per drive. Statistics: listed in supplement Table S2.

### Hiring-team results (base, 10; 3 candidates)
Prior two seasons of the hiring franchise, as two-season means of season z-scores.
Candidates: `turnovers_forced`, `turnovers_committed`, `num_playoff_appearances` (a count,
0-2). *Pruned:* win %, points scored/allowed, yards gained/allowed, yards per play
gained/allowed. Flag: `org_expansion`.

### Career structure (engineered, 20; 19 candidates)
| Column | Definition |
|---|---|
| `cf_num_nfl_employers` | Distinct NFL/AFL franchises before the hire (relocation-aware) |
| `cf_num_college_employers` | Distinct college employers before the hire |
| `cf_nfl_share` | Share of prior coaching seasons at NFL/AFL level |
| `cf_age_first_job` | Age at first coaching job |
| `cf_age_first_nfl` | Age at first NFL/AFL coaching job (age at hire if none) |
| `cf_gap_before_hire` | Hire year minus last prior coaching season minus 1 |
| `cf_internal_hire` | Any coaching role with the hiring organization in the season before the hire (midseason rule above) |
| `cf_years_at_hiring_org` | Consecutive seasons with the hiring organization immediately before the hire |
| `cf_pre_OC`, `cf_pre_DC`, `cf_pre_POS`, `cf_pre_HC` | Most recent prior role was OC / DC / position coach / head coach (any level) |
| `cf_pre_side_off` | Most recent prior role was on offense (*`cf_pre_side_def` pruned*) |
| `cf_pre_level_nfl` | Most recent prior role was at NFL/AFL level |
| `cf_ever_nfl_oc`, `cf_ever_nfl_dc` | Ever an NFL/AFL offensive / defensive coordinator |
| `cf_ever_interim_hc` | **New.** Held an interim head-coach title in a prior NFL/AFL season |
| `cf_no_unit_record` | **New.** Never led an NFL/AFL unit (no OC, DC or non-interim HC season) |
| `cf_no_drive_record` | **New.** Has a unit record but none from the drive-stat era |

### Recent form (engineered, 7; 6 candidates)
From PFR's ranks table, prior non-interim ranked seasons. Flag: `cf_no_unit_record`.
| Column | Definition |
|---|---|
| `rf_final_unit_pctl` | League percentile of the coach's own unit in his last ranked season |
| `rf_avg_unit_pctl` | Mean of those unit percentiles |
| `rf_unit_traj` | Slope of unit percentile over the last three ranked seasons; 0 if only one |
| `rf_avg_winpct_pctl` | Mean league percentile of team win % over ranked seasons (*`rf_final_winpct_pctl` pruned for a PH violation*) |
| `rf_n_seasons_rank` | Number of prior ranked seasons |
| `rf_no_trend` | **New.** Exactly one ranked season, so no trajectory exists |

### Organizational instability (engineered, 4; all candidates)
| Column | Definition |
|---|---|
| `org_prev_coach_tenure` | Consecutive seasons of the coach replaced (prior season's season-opening HC) |
| `org_unique_hc_10yr` | Distinct head coaches in the prior ten seasons (the hire's own partial takeover season excluded) |
| `org_expansion` | **New.** First head coach of an expansion franchise (known list of 6) |
| `org_prev_fired` | **New.** Predecessor's stint ended in involuntary non-retention (same protocol as the outcome; 26 pre-window predecessors hand-coded in `analysis/predecessor_exits_prewindow.csv`) |

### Inherited team quality (engineered, 6; 4 candidates)
PFR team ratings for the season before the hire (former AFL franchises: AFL seasons).
Candidates: `tq_osrs`, `tq_dsrs`, `tq_sos`, `tq_srs_traj` (SRS slope over the three prior
seasons). *Pruned:* `tq_srs`, `tq_mov` (linear identities of the kept components).

### Inherited roster (engineered, 14; 11 candidates)
Hiring franchise's season before the hire (the 1996 Ravens use the 1995 Browns).
Flags: `org_expansion`, `hire_no_roster_data` (1970 hires), `hire_no_retention_data`
(1970-71 hires; rosters begin in 1970).
| Column | Definition |
|---|---|
| `hire_qb_av` | Average AV of the listed starting QB(s) |
| `hire_roster_av` / `hire_starter_av` | Average AV per rostered player / per starter |
| `hire_starter_exp` / `hire_roster_exp` | Average seasons of experience, starters / roster (*ages pruned*) |
| `hire_recent_r1_picks` | First-round picks in the three drafts before the hire |
| `hire_premium_draft` | First- plus second-round picks in the same three drafts |
| `hire_roster_retention` | % of the roster two seasons before the hire still on the prior-season roster (*pruned: pooled version of the split below, r = 0.81 / 0.83*) |
| `hire_off_retention` / `hire_def_retention` | Same, for players listed on offense / defense |
| `hire_no_roster_data`, `hire_no_retention_data` | **New.** Coverage flags |

## 3. Status
- Supplement rewritten (2026-10-28): Table S1 from Section 2 (81 candidates; the `cf_pre_*`
  indicators relabeled "Hired from ..." since they measure the most recent role, which is the
  intended concept), new Section S3 on data construction and coding decisions, updated
  voluntary-exit and censored lists. Values that depend on the analysis rerun carry
  `% TODO(rerun)` markers in `papers/jse/supplement.tex`.
- Coach_WAR retention fixed and regenerated (uncommitted); its models still need a rerun.
- Main-text Table 1 (feature families and counts) still to update with the paper revision.
