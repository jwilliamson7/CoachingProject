# Mid-season takeovers: internal-hire coding (supplement source)

Compiled 2026-10-28. Covers every hire in the sample whose stint is anchored at the first
full season after a mid-season takeover in which Pro-Football-Reference lists the coach
only as "Head Coach" for the partial season (16 cases). PFR records one role per partial
season and drops whatever the coach did earlier that season, so the data alone cannot say
whether he was an insider. Each case was therefore resolved from contemporaneous sources.

**Coding rule.** A hire is internal if the coach was employed by the organization (coaching
staff or front office) before the team chose him. For a mid-season takeover the stint is
anchored at the next season-opening year and the partial season is folded into prior
experience; the prior-season rule then codes the takeover internal. Outside mid-season
appointments are recoded external through `MIDSEASON_OUTSIDE_HIRES` in
`scripts/data/engineer_career_features.py`.

Detail and sources: `analysis/midseason_hires_period_press.md` (part 1: Coryell, Berry,
Levy, Meyer) and `analysis/midseason_takeovers_period_press_part2.md` (part 2: the six
insiders). Interim-versus-permanent status is recorded for completeness; it does not
change the internal coding, which depends only on whether the coach was already inside.

| Coach | Takeover | Anchored | Team | Before taking over | Interim or permanent | Coded | Evidence |
|---|---|---|---|---|---|---|---|
| Jim Ringo | 1976 | 1977 | BUF | Bills offensive line coach (1975) | not needed | internal | On staff the prior season (PFR) |
| Al Saunders | 1986 | 1987 | SD | Chargers receivers coach (1985) | not needed | internal | On staff the prior season (PFR) |
| Art Shell | 1989 | 1990 | LAR | Raiders offensive line coach (1988) | not needed | internal | On staff the prior season (PFR) |
| Richard Williamson | 1990 | 1991 | TB | Buccaneers WR coach / asst. HC (1989) | not needed | internal | On staff the prior season (PFR) |
| Jason Garrett | 2010 | 2011 | DAL | Cowboys asst. HC / OC (2009) | not needed | internal | On staff the prior season (PFR) |
| Leslie Frazier | 2010 | 2011 | MIN | Vikings asst. HC / DC (2009) | not needed | internal | On staff the prior season (PFR) |
| Ed Khayat | 1971 | 1972 | PHI | Eagles DL coach, hired 1971 offseason | unresolved | internal | Eagles 1971 media guide (pre-season staff list) |
| Harland Svare | 1971 | 1972 | SD | Chargers general manager since Feb 1971 | interim, made permanent 1971-12-19 | internal | UPI 1971-11-23; Chargers 1973 guide |
| Sid Gillman | 1973 | 1974 | HOU | Oilers EVP / general manager since 1973-03-02 | interim in practice ("for the rest of year") | internal | SI 1973-11-12; Oilers 1974 guide |
| John McVay | 1976 | 1977 | NYG | Giants assistant ("Research and Development") since Mar 1976 | contested; rehired on a two-year contract 1976-12-14 | internal | Harvard Crimson 1976-10-30; NYT 1976-12-15 |
| Tommy Hudspeth | 1976 | 1977 | DET | Lions coordinator of personnel and scouting (1975-76) | interim; three-year contract 1977-02-09 | internal | AP 1976-10-06; Lions 1977 guide |
| Homer Rice | 1978 | 1979 | CIN | Bengals quarterbacks coach since Apr 1978 | unresolved, leaning interim | internal | Dayton Daily News 1978-10-03; Bengals 1979 guide |
| Don Coryell | 1978 | 1979 | SD | Fired by the Cardinals (Feb 1978); no Chargers role | permanent (Chargers traded a draft pick for his release) | **external** | Washington Post 1978-09-26 and 1978-10-10 |
| Raymond Berry | 1984 | 1985 | NE | Out of football 1982-84 | permanent ("multi-year pact") | **external** | UPI 1984-10-26 (full text) |
| Marv Levy | 1986 | 1987 | BUF | Out of coaching; no Bills coaching role | unresolved, leaning permanent | **external** | UPI 1986-11-09; no period source says interim |
| Ron Meyer | 1986 | 1987 | IND | Fired by the Patriots (1984); no Colts role | permanent ("long-term commitment") | **external** | UPI 1986-12-02 |

**Why reference works disagree.** Many sources, including the Pro Football Researchers
Association's 2004 article "Interim Coaches", label any mid-season replacement "interim" by
timing alone. That timing-based label is not used here: the question for the internal-hire
flag is where the coach came from, not what title he held for the rest of the season.

**Open hand checks (optional; none would change the coding).** The interim-versus-permanent
status of Khayat, Rice and Levy rests on snippets; the pages are listed in the two detail
files. Several quotes passed through a summarizing fetch tool and should be checked against
the original pages before being quoted in the paper.
