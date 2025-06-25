# simpo/tests/test_utils.py (或者您的 test_utils.py 所在的正确路径)

# 确保导入了 transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
import torch.nn as nn
from unittest.mock import MagicMock
import pytest
from multiprocessing import Queue # 确保导入Queue

# 假设您的_run_model_inference_worker在simpo.utils模块中
from simpo.utils import _run_model_inference_worker

# 引入日志（如果您的_run_model_inference_worker使用了logger）
import logging
logger = logging.getLogger(__name__)


@pytest.fixture
def mock_tokenizer(mocker):
    """
    Mock a transformers AutoTokenizer instance.
    """
    mock_tokenizer_instance = MagicMock(spec=PreTrainedTokenizerBase)
    mock_tokenizer_instance.name_or_path = "mock/tokenizer"
    mock_tokenizer_instance.pad_token_id = 0 # 假设pad_token_id是0
    mock_tokenizer_instance.eos_token_id = 10 # 假设eos_token_id是10

    # ！！！关键修改 ！！！
    # 模拟 tokenizer 自身的调用行为
    # 当 tokenizer(...) 被调用时，它需要返回一个模拟的 BatchEncoding 对象
    mock_batch_encoding = MagicMock()
    # 模拟 prompt 的 input_ids，长度与您的mock_model.generate的prompt部分一致
    mock_batch_encoding.input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    mock_batch_encoding.attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
    
    # 确保返回的对象有 .to() 方法
    mock_batch_encoding.to.return_value = mock_batch_encoding # .to() 返回自身，简化模拟

    # 当 mock_tokenizer_instance(...) 被调用时，返回这个模拟的 BatchEncoding 对象
    mock_tokenizer_instance.return_value = mock_batch_encoding

    # 模拟 tokenizer.decode 或 batch_decode (确保 _run_model_inference_worker 不会用到它)
    # 实际上，_run_model_inference_worker 并没有直接使用 decode，它把 response_ids 放入队列
    # 但为了完整性，如果其他测试用到，可以保留：
    def mock_decode(token_ids, **kwargs):
        return f"mocked text from tokens {token_ids.tolist()}"
    mock_tokenizer_instance.decode.side_effect = mock_decode
    mock_tokenizer_instance.batch_decode.side_effect = lambda token_ids_list, **kwargs: [mock_decode(t) for t in token_ids_list]

    # patch AutoTokenizer.from_pretrained
    # 假设您的_run_model_inference_worker没有从头加载tokenizer，而是直接接收
    # 如果有，这里的路径要和实际加载的地方匹配
    mocker.patch('simpo.utils.AutoTokenizer.from_pretrained', return_value=mock_tokenizer_instance)

    return mock_tokenizer_instance

@pytest.fixture
def mock_model(mocker):
    """
    Mock a transformers AutoModelForCausalLM instance.
    """
    mock_model_instance = MagicMock(spec=AutoModelForCausalLM)
    mock_model_instance.device = torch.device("cuda:0") # 假设在cuda:0上
    mock_model_instance.name_or_path = "mock/model-7b"

    # 显式地模拟 generate 方法，并设置其返回值
    mocker.patch.object(mock_model_instance, 'generate', autospec=True)
    mock_model_instance.generate.return_value = torch.tensor([[
        1, 2, 3, 4,   # mock prompt_ids (length 4)
        5, 6, 7, 8, 9 # mock generated_ids (length 5)
    ]], dtype=torch.long, device="cuda:0")

    # 模拟 model() 方法的返回值 (用于计算对数似然)
    mock_logits = torch.randn(1, 9, 1000, device="cuda:0") # batch=1, seq_len=9, vocab_size=1000
    mock_outputs = MagicMock()
    mock_outputs.logits = mock_logits
    mock_model_instance.return_value = mock_outputs # 当调用 model(...) 时返回

    # 模拟 model.config
    # 尽量模拟真实的config对象，如果可以直接实例化一个，或者更详细地模拟其属性
    # 简单起见，这里假设只需要pad_token_id
    mock_model_instance.config = MagicMock()
    mock_model_instance.config.pad_token_id = 0 # 假设pad_token_id是0

    # patch AutoModelForCausalLM.from_pretrained (这里路径要和实际加载的地方匹配)
    mocker.patch('simpo.utils.AutoModelForCausalLM.from_pretrained', return_value=mock_model_instance)

    return mock_model_instance


# 您原来的测试函数
def test_run_model_inference_worker(mock_model, mock_tokenizer):
    """
    Tests the _run_model_inference_worker function in an isolated environment.
    """
    model_name = "mock/model-7b"
    alias = "MockModel"
    tokenizer_name = "mock/tokenizer"
    prompt_text_list = ["Once upon a time, in a land far, far away,"]
    num_generations = 3

    response_queue = Queue() # multiprocessing.Queue 实例

    # 添加调试打印语句
    logger.info("Calling _run_model_inference_worker...")
    
    _run_model_inference_worker(
        model=mock_model,
        tokenizer=mock_tokenizer,
        prompt=prompt_text_list,
        response_queue=response_queue,
        max_new_tokens=50,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    
    logger.info(f"After worker call: Queue empty = {response_queue.empty()}")
    logger.info(f"After worker call: Queue size = {response_queue.qsize()}")

    # Assertions
    assert not response_queue.empty()

    # 如果队列不为空，还可以进一步断言其内容
    # result = response_queue.get()
    # assert result[0] == mock_model.name_or_path
    # assert torch.equal(result[1], torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long, device="cuda:0"))