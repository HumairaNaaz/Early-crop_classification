# Baseline notes - Gené-Mola et al. (2026)
**Evidence level:** the uploaded `Baseline_paper.htm` is only the ScienceDirect landing page (title, authors, highlights, abstract, keywords, data-availability line). **The full text (methods, tables, results, references) was NOT in the file.** Everything not on that page is marked *Needs verification*.
Citation: Gené-Mola J., Pamies-Sans M., Minuesa C., Casadesús J., Bellvert J., "Early crop type classification and mapping combining remote sensing and ancillary data," *Computers and Electronics in Agriculture*, vol. 253, art. 112138, Oct 2026 (online 8 Jul 2026), DOI 10.1016/j.compag.2026.112138. Open access (CC licence). Code+data: github.com/UEA-Irta/CatCrops.

| Heading | Verified from the abstract page | Needs verification (read full PDF) |
|---|---|---|
| Objective | Classify crop type **early in the season** by combining Sentinel-2-type remote-sensing time series with ancillary info | Exact stated research questions |
| Research problem | Most approaches use spectral time series and focus on end-of-season classification; early maps would help water allocation, production forecasting, price stabilisation | Specific prior-work limitations the authors cite |
| Dataset | About 130,000 fields, Lleida, Catalonia (Spain), "representative Mediterranean agricultural landscape" | Exact field count (plan says 137,408 from 2024 slides), 18 classes, years, class list, label source, train/test split |
| Input features | Remote-sensing time series; ancillary: cloud-cover %, previous-year crop type, irrigation type | Bands, indices, elevation/slope use, preprocessing, normalisation, handling of cloud gaps |
| Methodology | "Novel AI framework"; Transformer-based, self-attention, multi-modal fusion (keywords/highlights); designed for year-round classification | Architecture details, how ancillary data enter the model (the plan says they are appended to the time-series table - **not confirmed**) |
| Experimental setup | - | Training procedure, loss, optimiser, splits, metrics, test dates |
| Key results | ~70% accuracy at season onset using previous-season RS data; +10% from ancillary data -> ~80% early; 88% by season end; multi-year RS data improves early accuracy by 30% (highlights) | Per-date tables, per-class results, rotated-field result |
| Limitations | Under-represented classes remain hard because of class imbalance; authors call for better dataset-balancing techniques | Any other limitations; rotation analysis |
| Research gap (supported) | Class imbalance / rare crops (authors say so) | Fusion design and rotation weakness are **our hypotheses** until the text confirms them |
| Relevance to us | Our baseline, dataset and protocol source; our class-balanced focal loss targets the stated imbalance problem | - |

## Discrepancies to resolve before writing
1. Plan says rotated-field accuracy falls **97% -> ~56%** early season; not on the abstract page. Confirm figure, date and subset definition.
2. Plan quotes "about 0.70 early to 0.81 late" (2024 slides); the paper says **88% at season end**. Different versions - use only the paper.
3. Field count: ~130,000 (paper abstract) vs 137,408 (slides).
4. Surname spelling: "Casadesús" on the page vs "Casadésus" in the plan -> use the page.
5. Plan note: "multi-year RS data improves early accuracy by 30%" - careful: highlights say 30%, abstract says ~70% at onset. Confirm what 30% is relative to.
