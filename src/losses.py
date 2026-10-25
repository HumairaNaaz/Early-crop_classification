"""OWNER: Gate & Loss Lead (Deepasha). Class-Balanced Focal Loss with cross-entropy switch.
  weights w_c = (1 - beta_cb) / (1 - beta_cb ** N_c)   (Cui et al., 2019)
  loss = - w_c * (1 - p_c) ** gamma_f * log p_c          (Lin et al., 2017)
Do NOT stack inverse-frequency weights on top of balanced sampling (see plan, Theme 4)."""


def build_loss(name: str, class_counts, beta_cb: float = 0.999, gamma_f: float = 2.0):
    raise NotImplementedError("TODO(Deepasha)")
