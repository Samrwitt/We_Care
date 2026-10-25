"""One NLLB model for Amharic→English questions and English→Amharic replies.

The DSFSI checkpoints are English→Amharic fine-tunes, so the question hop uses
the multilingual facebook/nllb-200-distilled-600M checkpoint (CC-BY-NC).
"""

from __future__ import annotations

from we_care_nlu.translation.protect import protect, restore


class NLLBTurnTranslator:
    can_translate_reply = True
    backend = "nllb"

    def __init__(
        self,
        hf_id: str = "facebook/nllb-200-distilled-600M",
        device: str = "auto",
        num_beams: int = 4,
    ):
        self.hf_id = hf_id
        self.device = device
        self.num_beams = num_beams
        self._model = None
        self._tokenizer = None
        self._resolved_device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        requested = (self.device or "auto").lower()
        if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
            self._resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._resolved_device = "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
        kwargs = {"torch_dtype": torch.float16} if self._resolved_device == "cuda" else {}
        model = AutoModelForSeq2SeqLM.from_pretrained(self.hf_id, **kwargs)
        model.to(self._resolved_device)
        model.eval()
        self._model = model

    def _translate(self, text: str, src: str, tgt: str) -> str:
        protected, mapping = protect(text)
        self._load()
        assert self._model is not None and self._tokenizer is not None
        import torch

        tokenizer = self._tokenizer
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = src
        encoded = tokenizer(
            protected,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        encoded = {key: value.to(self._resolved_device) for key, value in encoded.items()}
        bos = tokenizer.convert_tokens_to_ids(tgt)
        with torch.inference_mode():
            output = self._model.generate(
                **encoded,
                forced_bos_token_id=bos,
                num_beams=self.num_beams,
                max_new_tokens=200,
            )
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()
        return restore(decoded, mapping)

    def to_english(self, text: str) -> str:
        return self._translate(text, "amh_Ethi", "eng_Latn")

    def to_amharic(self, text: str) -> str:
        return self._translate(text, "eng_Latn", "amh_Ethi")
