from transformers import WhisperProcessor, WhisperForConditionalGeneration
import re
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
from utils import compute_metrics, DataCollatorSpeechSeq2SeqWithPadding, DataCollatorSpeechSeq2SeqWithPadding_from_npy
import os
import argparse
import evaluate
def define_argparser():
    """Function to define the command line arguments
    
    Returns:
        argparse.Namespace: Command line arguments
    """
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', type=str, default='./data/Bingsu_zeroth-korean')
    p.add_argument('--model_address', type=str, default='openai/whisper-medium')
    p.add_argument('--model_save_path', type=str, default='./models_zoo/openai_whisper-medium/')
    p.add_argument('--gradient_accumulation_steps', type=int, default=1)
    p.add_argument('--batch_size_per_device', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--total_step', type=int, default=20000)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=225)
    p.add_argument('--remove_text', type=str, default='n')
    p.add_argument('--deepspeed', type=str)
    p.add_argument('--local_rank', type=int)
    config = p.parse_args()

    return config

def remove_not_text(batch, tokenizer):
    labels = re.sub("o/|c/|n/|N/|u/|l/|b/|\*|\+|/", " ", 
                    tokenizer.decode(batch['labels'], skip_special_tokens=True))
    
    # encode target text to label ids     
    batch["labels"] = tokenizer(labels).input_ids
    return batch

def main(config):
    """Main function to train the language model

    Args:
        config (argparse.Namespace): Command line arguments
    """
    dataset = load_from_disk(config.data_path)
    processor = WhisperProcessor.from_pretrained(config.model_address, language="ko", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(config.model_address)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids()
    model.config.suppress_tokens = []
    
    if os.path.basename(config.data_path[:-1]) == "Bingsu_zeroth-korean":
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    else:
        data_collator = DataCollatorSpeechSeq2SeqWithPadding_from_npy(processor=processor)
        dataset = dataset.rename_column("npy_path",'input_features')
        if config.remove_text == 'y': # For clena data, These texts are already removed in denoised data.
            dataset = dataset.map(lambda x: remove_not_text(x, processor.tokenizer), num_proc=8)
        
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    print(
        '|train| =', len(dataset['train']),
        '|valid| =', len(dataset['val']),
    )
    
    # total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    # n_total_iterations = int(len(dataset['train']) / total_batch_size * config.n_epochs)
    n_total_iterations = config.total_step
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(config.model_save_path, 'checkpoints'),
        # num_train_epochs=config.n_epochs,
        max_steps=n_total_iterations,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=n_warmup_steps,
        fp16=True,
        learning_rate=config.lr,
        gradient_checkpointing=True,
        evaluation_strategy='steps',
        save_strategy ='steps',
        report_to=["tensorboard"],
        logging_steps=25,
        save_steps=n_total_iterations // 5,
        eval_steps=n_total_iterations // 5,
        predict_with_generate=True,
        generation_max_length=config.max_length,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=16,
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, wer_metric, cer_metric, 
                                                processor.tokenizer),
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    
    trainer.model.save_pretrained(os.path.join(config.model_save_path, 'model_weights'))
    #tokenizer.save_pretrained(os.path.join(config.model_save_path, 'tokenizer'))

if __name__ == '__main__':
    config = define_argparser()
    main(config)