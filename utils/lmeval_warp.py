import logging

from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from utils.modify_llama import QH2OLlamaForCausalLM

logger = logging.getLogger(__name__)


class LMEvalLM(HFLM):
    def __init__(
            self,
            model_name: str,
            config: AutoConfig = None,
            device: str = "cuda:0",
            dtype: str = "float16",
            batch_size: int = 1,
            load_compressed_weight: bool = False,
    ) -> None:
        super().__init__(
            pretrained="meta-llama/Llama-2-7b-chat-hf",
            backend="default",
            revision="main",
            subfolder=None,
            tokenizer=None,
            truncation=False,
            max_length=None,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            max_batch_size=64,
            trust_remote_code=False,
            use_fast_tokenizer=True,
            add_bos_token=False,
            prefix_token_id=None,
            parallelize=False,
            # device_map_option="auto",
            max_memory_per_gpu=10,
            max_cpu_memory=10,
            offload_folder="./offload",
            peft=None,
            delta=None,
            autogptq=False,
        )

        self._model.to("cpu")
        self._model = None
        # call GC
        import gc
        gc.collect()

        self._model = QH2OLlamaForCausalLM.from_pretrained(model_name, config=config).half().eval().cuda()

        self._model = self._model.half()
        self._model.to(self.device)
