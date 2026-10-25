# Interface contract (draft from team plan - CONFIRM at next meeting, then freeze)
| Tensor | Shape | Notes |
|---|---|---|
| temporal input | (B, T, 12) + mask (B, T) | 10 S2 bands + cloud prob + day-of-year |
| static input | continuous (B, n_cont); irrigation id (B,); municipality id (B,) | elevation, slope; irrigation type, municipality |
| previous-year crop | (B,) long | class id 0..17 |
| h_temp, h_stat, e_prior | (B, 64) | |
| gate value g | (B, 1) in [0,1] | |
| fused vector | (B, 64) | s(h_stat)*h_temp + o(h_stat) + g*e_prior |
| logits | (B, 18) | |
Open: how D_rot(t) is aggregated over time (mean / last step)? Cross-attention placement? (Owners: Deepasha, Anshika)
