# LaTeX Conversion of Coach WAR Paper

## Status: Nearly Complete

This folder contains the LaTeX conversion of the Coach WAR analysis paper for submission to the Journal of Quantitative Analysis in Sports (JQAS).

## Files

- `2026-Williamson-Jon-Portfolio-Coach-WAR.tex` - Main LaTeX document
- `figures/` - Directory containing figure files referenced in the paper

## Compilation

To compile the document:
```bash
pdflatex 2026-Williamson-Jon-Portfolio-Coach-WAR.tex
bibtex 2026-Williamson-Jon-Portfolio-Coach-WAR
pdflatex 2026-Williamson-Jon-Portfolio-Coach-WAR.tex
pdflatex 2026-Williamson-Jon-Portfolio-Coach-WAR.tex
```

Or use your preferred LaTeX editor (TeXworks, Overleaf, etc.).

## Template

The document uses the De Gruyter/JQAS LaTeX template (`dgruyter.sty`). You'll need to copy the template files from:
- `../JQAS_LaTeX-Template-for-Authors/dgruyter.sty`
- `../JQAS_LaTeX-Template-for-Authors/dgruyter.ist` (if using index)

Into this directory or ensure they're in your LaTeX path.

## Figures Status

### Completed Figures (10/13)
The following figures have been copied from `analysis/outputs/png/`:

1. ✓ `coach_2024_matrix.png` - 2024 coaches average WAR vs career length
2. ✓ `coach_2024_trajectories.png` - Cumulative WAR trajectories for 2024 coaches
3. ✓ `coach_2024_single_year_bar.png` - 2024 single-year WAR by coach
4. ✓ `coach_background_from_history_15seasons.png` - Average cumulative WAR by coach background
5. ✓ `coaching_war_persistence_scatter.png` - Year-to-year WAR persistence scatter
6. ✓ `coaching_regression_to_mean_survivorship_adjusted.png` - WAR quintile regression to mean
7. ✓ `coaching_survivorship_bias_magnitude.png` - Year-over-year WAR changes by quintile
8. ✓ `win_pct_persistence_scatter.png` - Win percentage persistence (Appendix F)
9. ✓ `coaching_war_persistence_by_background.png` - Single-year persistence by background (Appendix G)
10. ✓ `coaching_war_multiyear_persistence_scatter.png` - Multi-year persistence by background (Appendix G)

### Missing Figures (3/13)
The following figures need to be created:

1. **`coach_trajectories_oconnell_shula_eberflus.png`** (Figure 1)
   - Shows cumulative WAR trajectories for Kevin O'Connell, Don Shula, and Matt Eberflus
   - Data available in: `../data/final/coach_war_trajectories.csv`
   - Could be generated from Python script or extracted from existing HTML visualization
   - Similar visualization exists: `../analysis/outputs/html/coach_war_trajectory.html`

2. **`coach_career_distributions.png`** (Figure 2)
   - Scatter plot showing average WAR per season vs career length
   - Should show quadrants based on medians
   - Highlights market efficiency and inefficiency
   - Data available in coach career statistics files

3. **`dashboard_placeholder.png`** (Figure 10)
   - Screenshot of interactive dashboard
   - This is explicitly marked as placeholder in the paper
   - Could be a screenshot from the HTML dashboard files in `../analysis/outputs/html/`

## Before Submission

1. **Update Author Information** (Line 45-46):
   ```latex
   \affil[1]{\protect\raggedright
   University/Institution, Department, City, State, e-mail: author@email.com}
   ```

2. **Update Date Fields** (Lines 28-30):
   ```latex
   \received{October DD, 2025}
   \revised{Month DD, YYYY}
   \accepted{Month DD, YYYY}
   ```

3. **Create Missing Figures**:
   - Generate the three missing figures listed above
   - Ensure all figures are high resolution (at least 300 DPI for print)
   - Figures should be in PNG format (already specified in LaTeX)

4. **Copy Template Files**:
   - Copy `dgruyter.sty` and related files to this directory

5. **Test Compilation**:
   - Compile the full document to verify all references work
   - Check that all figures display correctly
   - Verify bibliography formatting

## Figure Generation Scripts

To create the missing figures, you can:

1. **For coach trajectories**: Modify `../analysis/coach_war_trajectory.py` to output the specific comparison
2. **For career distributions**: Create a scatter plot from coach career summary data
3. **For dashboard**: Take a screenshot of the HTML dashboard

## Notes

- All tables have been converted to LaTeX booktabs format
- Bibliography is in JQAS format (numbered citations)
- Appendices A-G are complete with proper table and figure formatting
- The comprehensive feature list in Appendix B is abbreviated due to space (368 features)
- Document follows JQAS template structure and formatting guidelines
