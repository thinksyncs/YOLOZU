def final_score(score_det, score_tmp_sym, sigma_z, sigma_rot, weights):
    w_det = weights.get("det", 1.0)
    w_tmp = weights.get("tmp", 1.0)
    w_unc = weights.get("unc", 1.0)
    return w_det * score_det + w_tmp * score_tmp_sym - w_unc * (sigma_z + sigma_rot)


def passes_template_gate(score_tmp_sym, enabled, tau):
    if not enabled:
        return True
    return score_tmp_sym >= tau


def passes_low_fp_gate(score_tmp_sym, enabled, tau):
    if not enabled:
        return True
    return score_tmp_sym >= tau
