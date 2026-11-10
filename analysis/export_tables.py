"""Export the verified regression and comparison results as English workbooks."""
from pathlib import Path
import pandas as pd
from analysis._config import ROOT, DATA_DIR, SOURCE_DATA
S=SOURCE_DATA
TABLES={
 'Table_S1_Predictive_contributions.xlsx':{
  'Model comparison':S/'table_results/main_model_comparison.csv','Group contributions':S/'table_results/main_group_contributions.csv',
  'All coefficients':S/'predictive_coefficients/complete_coefficients.csv','Refit summary':S/'predictive_coefficients/refit_model_summary.csv'},
 'Table_S2_Collaborator_regressions.xlsx':{
  'Model summary':S/'table_results/main_collaborator_model_summary.csv','All coefficients':S/'table_results/main_collaborator_coefficients.csv'},
 'Table_S3_Reference_characteristics.xlsx':{'Adjusted comparisons':S/'table_results/main_reference_comparisons.csv'},
 'Table_S4_Absolute_alpha_robustness.xlsx':{
  'Score states':S/'absolute_alpha_states.csv','Model comparison':S/'table_results/absolute_model_comparison.csv',
  'Group contributions':S/'table_results/absolute_group_contributions.csv','Collaborators':S/'table_results/absolute_collaborator_summary.csv'},
 'Table_S5_Delayed_adoption.xlsx':{
  'Samples':S/'trajectories/sample_flow.csv','Group comparisons':S/'trajectories/reference_group_contrasts.csv',
  'Period changes':S/'trajectories/reference_early_late_change_contrasts.csv','All reference coefficients':S/'trajectories/reference_complete_coefficients.csv',
  'Later non-resistance':S/'trajectories/transition_mechanisms.csv','Continuous R check':S/'trajectories/knowledge_continuous_R_sensitivity.csv','All model coefficients':S/'trajectories/mechanism_complete_coefficients.csv'},
 'Table_S6_Publication_sequence_and_language.xlsx':{
  'Publication blocks':S/'sequence_language/block_persistence.csv','Same author sensitivity':S/'sequence_language/same_author_width_sensitivity.csv',
  'Next publication':S/'sequence_language/next_publication_summary.csv','Language retention':S/'sequence_language/language_cohort_retention.csv','Language comparison':S/'sequence_language/language_adjusted_differences.csv',
  'Language interactions':S/'sequence_language/language_joint_interactions.csv','Language coefficients':S/'sequence_language/language_complete_coefficients.csv'}
}


def export(output=None):
    from openpyxl.styles import Font,PatternFill,Alignment
    dest=Path(output or ROOT/'tables');dest.mkdir(parents=True,exist_ok=True)
    for filename,sheets in TABLES.items():
        with pd.ExcelWriter(dest/filename,engine='openpyxl') as writer:
            for label,path in sheets.items():
                frame=pd.read_csv(path);frame.to_excel(writer,index=False,sheet_name=label)
                sheet=writer.sheets[label];sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
                for cell in sheet[1]:cell.font=Font(name='Arial',bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='36566F')
                for col in sheet.columns:
                    values=[str(c.value) for c in list(col)[:120] if c.value is not None];sheet.column_dimensions[col[0].column_letter].width=min(55,max(14,max(map(len,values),default=14)+2))
                for row in sheet.iter_rows(min_row=2):
                    for cell in row:
                        cell.font=Font(name='Arial',size=10)
                        if isinstance(cell.value,float):cell.number_format='0.000'
                sheet.sheet_view.showGridLines=False
        print('Exported',filename,flush=True)
    return {'workbooks':len(TABLES)}
