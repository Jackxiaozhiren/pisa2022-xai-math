# Variable Plan

The code starts with a broad candidate list and keeps only variables actually present in the provided PISA 2022 files. The final variable set must be frozen after the first complete data audit.

## Required Identifiers and Survey Variables

| Role | Default variable |
|---|---|
| Country/economy | `CNT` |
| Student ID | `CNTSTUID` |
| School ID | `CNTSCHID` |
| Final student weight | `W_FSTUWT` |
| Replicate weights | `W_FSTURWT1` ... `W_FSTURWT80` |
| Mathematics plausible values | `PV1MATH` ... `PV10MATH` |

## Candidate Feature Groups

Student background:

- `ST004D01T`: gender.
- `AGE`: age.
- `GRADE`: grade.
- `ESCS`: economic, social, and cultural status.
- `IMMIG`: immigrant background.
- `HISEI`, `PAREDINT`, `HOMEPOS`: family socioeconomic and home resources.

Mathematics attitudes and learning:

- `ANXMAT`: mathematics anxiety.
- `MATHEFF` or `MATHEF21`: mathematics self-efficacy.
- `MATHPERS`: mathematics persistence.
- `BELONG`: sense of belonging.
- `BULLIED`, `FEELSAFE`, `SCHRISK`: school safety and risk climate.

Family and support:

- `FAMCON`: family connection/support.
- `FAMSUP`: family support if present.
- `PQSCHOOL`, `PASCHPOL`: parent/school relationship variables if present.

School environment:

- `DISCLIM`: disciplinary climate.
- `TEACHSUP`: teacher support.
- `PERFEED`: perceived feedback.
- `STUBEHA`, `TEACHBEHA`: student/teacher behavior.
- `EDUSHORT`, `STAFFSHORT`: educational resource and staff shortage.

Digital learning:

- `ICTRES`: ICT resources.
- `ICTHOME`: ICT availability/use at home if present.
- `ICTSCH`: ICT availability/use at school if present.
- `ICTEFFIC`: ICT self-efficacy.
- `ICTDISTR`: digital distraction if present.
- `LEARNRES`, `DISTICT`, `STUDYHMW`: learning resources, distance learning, homework/study indicators when present.

## Variable Audit Decisions

After `scripts/01_prepare_data.py` runs, inspect `data/processed/prepare_data_report.json`.

Freeze the final model feature set by:

1. Removing variables missing from the data.
2. Removing variables with extreme missingness unless theoretically essential.
3. Documenting every removed variable in the Methods supplement.
4. Keeping the final set consistent across regression and classification tasks.

## Risk Notes

- PISA variable names can differ across files and questionnaire forms. Do not assume every index in this candidate list exists in every public use file.
- Some constructs are derived OECD indices. If an index is missing, use the codebook to identify its item-level alternatives.
- Country fixed effects may dominate prediction in global models. Report both global and within-country/region robustness results.
