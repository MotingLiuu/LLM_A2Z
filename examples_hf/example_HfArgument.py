from dataclasses import dataclass, field # dataclass，快速创建只包含数据（属性）的类，自动生成init方法等。 field用于定义类属性的默认值和元数据。元数据是一个字典，可以包含帮助信息等。
from transformers import HfArgumentParser 

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune,
    or train from scratch.
    """
    model_name_or_path: str | None = field( 
        default=None,
        metadata={"help": "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."},
    )
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={"help": "The name or path to the tokenizer."},
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: str | None = field(
        default=None, metadata={"help": "The input training data file (a jsonl file)."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )

@dataclass
class SFTConfig:
    """
    Arguments pertaining to the SFT training configuration.
    """
    learning_rate: float = field(
        default=2e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Total number of training epochs to perform."}
    )
    output_dir: str | None = field( 
        default=".", metadata={"help": "Where to store the final model."}
    )

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, sft_config = parser.parse_args_into_dataclasses() # 解析命令行参数为三个数据类的实例

    print(f"Model path: {model_args.model_name_or_path}")
    print(f"Train file: {data_args.train_file}")
    print(f"Learning rate: {sft_config.learning_rate}")
    print(f"type of model_args: {type(model_args)}")
    
'''
python example_HfArgument.py \
    --model_name_or_path bert-base-uncased \
    --tokenizer_name_or_path bert-base-uncased \
    --train_file /path/to/your/training_data.jsonl \
    --max_seq_length 256 \
    --learning_rate 1e-4 \
    --num_train_epochs 5 \
    --output_dir ./my_sft_output
'''