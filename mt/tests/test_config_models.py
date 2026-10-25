from we_care_mt.config import load_config
from we_care_mt.constants import NLLB_SRC_LANG, NLLB_TGT_LANG


def test_model_registry_licenses_and_language_codes(project_root):
    cfg = load_config(project_root / "configs" / "default.yaml")
    lang = cfg.models["language"]
    assert lang["source_code"] == NLLB_SRC_LANG
    assert lang["target_code"] == NLLB_TGT_LANG
    models = cfg.models["models"]
    for key in ("dsfsi_600m", "dsfsi_1_3b", "nllb_baseline_600m"):
        assert key in models
        assert models[key]["clinically_validated"] is False
        assert models[key]["hf_id"]
        assert models[key]["license"]
    assert models["nllb_baseline_600m"]["license"].startswith("CC-BY-NC")
    assert models["dsfsi_600m"]["license"].startswith("Apache")
    assert models["dsfsi_1_3b"]["recommended_vram_gb"] >= models["dsfsi_600m"]["recommended_vram_gb"]
