# install transformer from source ---
import gc
import os
import time
from copy import deepcopy

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BitsAndBytesConfig

try:
    from r_gemma.lmsys_dataset import LMSYSDataset
    from r_gemma.lmsys_loader import LMSYSCollator, show_batch
    from r_gemma.lmsys_model import GemmaForLMSYS
    from r_gemma.lmsys_optimizer import get_optimizer
    from utils.metric_utils import get_score
    from utils.train_utils import (
        AverageMeter,
        as_minutes,
        get_custom_cosine_schedule_with_warmup,
        get_lr,
        label_smoothing,
        log_gradient_norms,
        setup_training_run,
    )


except Exception as e:
    print(e)
    raise ImportError

logger = get_logger(__name__)


def to_list(t):
    return t.float().cpu().numpy().tolist()


def run_evaluation(accelerator, model, valid_dl, valid_ids, id2topic=None, id2language=None, id2chat_type=None):
    model.eval()

    all_predictions = []
    all_truths = []

    progress_bar = tqdm(range(len(valid_dl)), disable=not accelerator.is_local_main_process)

    for step, batch in enumerate(valid_dl):
        with torch.no_grad():
            labels = batch.pop("labels").to(torch.long)
            outputs = model(**batch)

        logits = outputs.logits
        predictions = F.softmax(logits, dim=-1)
        predictions, references = accelerator.gather_for_metrics((predictions, labels.reshape(-1)))
        predictions, references = to_list(predictions), to_list(references)

        all_predictions.extend(predictions)
        all_truths.extend(references)

        progress_bar.update(1)
    progress_bar.close()

    # compute oof dataframe ---
    oof_df = pd.DataFrame(all_predictions)
    oof_df.columns = ["winner_model_a", "winner_model_b", "winner_tie"]
    oof_df["id"] = valid_ids
    oof_df = oof_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]]
    oof_df = oof_df.reset_index(drop=True)

    true_df = pd.DataFrame(all_truths)
    true_df.columns = ["label"]
    true_df["id"] = valid_ids
    true_df = true_df[["id", "label"]]
    true_df = true_df.reset_index(drop=True)

    # compute metric ---
    eval_dict = get_score(true_df, oof_df)

    result_df = oof_df.copy()
    result_df["label"] = all_truths

    # label smoothing ---
    smooth_df = deepcopy(oof_df)
    cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    for col in cols:
        smooth_df[col] = smooth_df[col].apply(label_smoothing)
    log_loss_smooth = get_score(true_df, smooth_df)["lb"]

    # compute score on topic subsets ---
    subset_scores = []
    subset_metrics = {}

    if id2topic is not None:
        result_df["topic"] = result_df["id"].map(id2topic)
        topics = result_df["topic"].unique()

        for topic in topics:
            topic_df = result_df[result_df["topic"] == topic]
            t = topic_df[["id", "label"]].copy()
            p = topic_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].copy()
            s = get_score(t, p)["lb"]
            subset_scores.append({"subset": f"Topic: {topic}", "score": s, "support": len(topic_df)})
            subset_metrics[topic] = s

    if id2language is not None:
        result_df["language"] = result_df["id"].map(id2language)
        languages = result_df["language"].unique()
        for language in languages:
            language_df = result_df[result_df["language"] == language]
            t = language_df[["id", "label"]].copy()
            p = language_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].copy()
            s = get_score(t, p)["lb"]
            subset_scores.append({"subset": f"Language: {language}", "score": s, "support": len(language_df)})
            subset_metrics[language] = s

    if id2chat_type is not None:
        result_df["chat_type"] = result_df["id"].map(id2chat_type)
        chat_types = result_df["chat_type"].unique()
        for chat_type in chat_types:
            chat_type_df = result_df[result_df["chat_type"] == chat_type]
            t = chat_type_df[["id", "label"]].copy()
            p = chat_type_df[["id", "winner_model_a", "winner_model_b", "winner_tie"]].copy()
            s = get_score(t, p)["lb"]
            subset_scores.append({"subset": f"Chat Type: {chat_type}", "score": s, "support": len(chat_type_df)})
            subset_metrics[chat_type] = s

    subset_df = pd.DataFrame(subset_scores)

    to_return = {
        "lb": eval_dict["lb"],
        "log_loss_smooth": log_loss_smooth,
        "result_df": result_df,
        "oof_df": oof_df,
        "subset_df": subset_df,
        "subset_metrics": subset_metrics,
    }

    return to_return


@hydra.main(version_base=None, config_path="../conf/r_gemma", config_name="conf_r_gemma")
def run_training(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    accelerator = setup_training_run(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit * 50 + suffix)

    # ------- load data -----------------------------------------------------------------#
    print_line()
    if cfg.pretrain:
        lmsys_train_df = pd.read_parquet(os.path.join(cfg.input_dir, "train.parquet"))
        pair_df = pd.read_parquet(os.path.join(cfg.input_dir, "pair_corpus.parquet"))
        pair_df = pair_df.sample(cfg.n_pair_samples).reset_index(drop=True)

        train_df = pd.concat([lmsys_train_df, pair_df]).reset_index(drop=True)
        valid_df = pd.read_parquet(os.path.join(cfg.input_dir, "valid.parquet"))
    else:
        train_df = pd.read_parquet(os.path.join(cfg.input_dir, "train.parquet"))
        valid_df = pd.read_parquet(os.path.join(cfg.input_dir, "valid.parquet"))

    # mappings for fine-grained evaluations ---
    if "topic" not in valid_df.columns:
        valid_df["topic"] = "common"
        valid_df["language"] = "en"

    id2topic = dict(zip(valid_df["id"], valid_df["topic"]))
    id2language = dict(zip(valid_df["id"], valid_df["language"]))

    # multi-turn
    valid_df["chat_type"] = valid_df["prompt"].apply(lambda x: "multi-turn" if len(x) > 1 else "single-turn")
    id2chat_type = dict(zip(valid_df["id"], valid_df["chat_type"]))
    valid_df = valid_df.drop(columns=["chat_type"])

    if cfg.use_99:
        # train_df = train_df[~train_df["kfold"].isna()].copy()
        train_df = train_df[train_df["kfold"] == 99].copy()
        train_df = train_df.reset_index(drop=True)
        accelerator.print("Using kfold=99 data for training")
        accelerator.print(f"shape of train data: {train_df.shape}")

    else:
        # train_df = train_df[train_df["kfold"] != 99].copy()
        train_df = train_df.reset_index(drop=True)
        accelerator.print("Using host for training")
        accelerator.print(f"shape of train data: {train_df.shape}")

    if cfg.debug:
        n = min(1024, len(train_df))
        train_df = train_df.head(n).reset_index(drop=True)

    if cfg.full_fit:  # train on full data
        train_cols = list(train_df.columns)
        add_df = deepcopy(valid_df)
        add_df = add_df[train_cols].copy()

        def convert_label(x):
            l = int(x["label"])
            K = 3.0
            epsilon = 0.03

            oh = [0.0, 0.0, 0.0]
            oh[l] = 1.0
            x = np.array(oh)
            x = (1 - epsilon) * x + (epsilon / K)
            return list(x)

        add_df["label"] = add_df.apply(convert_label, axis=1)

        train_df = pd.concat([train_df, add_df]).reset_index(drop=True)
        valid_df = valid_df.head(156).reset_index(drop=True)
        accelerator.print(valid_df["label"].value_counts())
        id2topic = None
        id2language = None
        id2chat_type = None

    accelerator.print(f"shape of train data: {train_df.shape}")
    accelerator.print("Examples from train data:")
    accelerator.print(f"{train_df.sample(3)}")
    accelerator.print(f"shape of validation data: {valid_df.shape}")
    print_line()

    # with accelerator.main_process_first():
    dataset_creator = LMSYSDataset(cfg)

    train_ds = dataset_creator.get_dataset(train_df)
    valid_ds = dataset_creator.get_dataset(valid_df)

    tokenizer = dataset_creator.tokenizer

    valid_ds = valid_ds.sort("length")
    valid_ids = valid_ds["id"]

    data_collator = LMSYSCollator(tokenizer=tokenizer, pad_to_multiple_of=64)
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=1,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train_params.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    accelerator.print("data preparation done...")
    print_line()

    del train_df
    gc.collect()

    # --- show batch -------------------------------------------------------------------#
    print_line()
    for b in train_dl:
        break
    show_batch(b, tokenizer, task="training", print_fn=accelerator.print)
    print_line()

    for b in valid_dl:
        break
    show_batch(b, tokenizer, task="validation", print_fn=accelerator.print)

    # --- model -------------------------------------------------------------------------#
    print_line()

    if cfg.model.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["classification_head"],
        )

        base_model = GemmaForLMSYS.from_pretrained(
            cfg.model.backbone_path,
            num_labels=cfg.model.num_labels,  # 3
            quantization_config=bnb_config,
            ignore_mismatched_sizes=True,
        )
    else:
        base_model = GemmaForLMSYS.from_pretrained(
            cfg.model.backbone_path,
            num_labels=cfg.model.num_labels,  # 3
            torch_dtype=torch.bfloat16,
        )

    ###
    base_model.config.pretraining_tp = 1

    if cfg.model.use_gradient_checkpointing:
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # # --- reinit lstm weights ---
    print_line()
    accelerator.print("before reinit weights")
    for n, p in base_model.classification_head.named_parameters():
        accelerator.print(n, p)
        accelerator.print(f"Min: {p.min()}, Max: {p.max()}, Shape: {p.shape}")
    print_line()

    if cfg.model.reinit_head:
        base_model.reinit_head()
        print_line()
        accelerator.print("after reinit weights")
        for n, p in base_model.classification_head.named_parameters():
            accelerator.print(n, p)
            accelerator.print(f"Min: {p.min()}, Max: {p.max()}, Shape: {p.shape}")
        print_line()

    # lora ---
    peft_config = LoraConfig(
        use_dora=cfg.model.lora.use_dora,
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.lora_alpha,
        lora_dropout=cfg.model.lora.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        target_modules=cfg_dict["model"]["lora"]["target_modules"],
        modules_to_save=cfg_dict["model"]["lora"]["modules_to_save"],
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    accelerator.wait_for_everyone()

    # --- optimizer ---------------------------------------------------------------------#
    print_line()
    optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)

    # ------- Prepare -------------------------------------------------------------------#

    model, optimizer, train_dl, valid_dl = accelerator.prepare(model, optimizer, train_dl, valid_dl)

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    num_epochs = cfg.train_params.num_train_epochs
    grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl) // grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct * num_training_steps)

    accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    # ------- training setup --------------------------------------------------------------#
    best_lb = 1e6

    patience_tracker = 0
    current_iteration = 0

    # ------- training  --------------------------------------------------------------------#
    start_time = time.time()
    accelerator.wait_for_everyone()
    progress_bar = None

    for epoch in range(num_epochs):
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
        loss_meter = AverageMeter()

        # Training ------
        model.train()

        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):  # gives sync vs no sync context manager
                outputs = model(**batch)
                loss = outputs.loss
                # loss = loss / grad_accumulation_steps
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if cfg.use_wandb:
                        log_gradient_norms(accelerator, model, current_iteration)
                    accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                loss_meter.update(loss.item())  # tracks loss in each batch, no accumulation

            if accelerator.sync_gradients:
                progress_bar.set_description(
                    f"STEP: {current_iteration+1:5}/{num_training_steps:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    accelerator.log({"train_loss": round(loss_meter.avg, 5)}, step=current_iteration)  # only on main process
                    accelerator.log({"lr": get_lr(optimizer)}, step=current_iteration)

            # ------- evaluation  -------------------------------------------------------#
            if (accelerator.sync_gradients) & (current_iteration % cfg.train_params.eval_frequency == 0):
                # set model in eval mode ---
                model.eval()
                eval_response = run_evaluation(accelerator, model, valid_dl, valid_ids, id2topic, id2language, id2chat_type)

                lb = eval_response["lb"]
                result_df = eval_response["result_df"]
                oof_df = eval_response["oof_df"]
                subset_df = eval_response["subset_df"]

                print_line()
                et = as_minutes(time.time() - start_time)
                accelerator.print(f">>> Epoch {epoch+1} | Step {step} | Total Step {current_iteration} | Time: {et}")
                print_line()
                accelerator.print(f">>> Current LB (Log Loss) = {round(lb, 4)}")
                accelerator.print(f">>> Smoothed Log Loss = {round(eval_response['log_loss_smooth'], 4)}")

                print_line()

                is_best = False
                if lb <= best_lb:
                    best_lb = lb
                    is_best = True
                    patience_tracker = 0

                else:
                    patience_tracker += 1

                if is_best:  # do in main process
                    oof_df.to_csv(os.path.join(cfg.outputs.model_dir, "oof_df_best.csv"), index=False)
                    result_df.to_csv(os.path.join(cfg.outputs.model_dir, "result_df_best.csv"), index=False)
                    subset_df.to_csv(os.path.join(cfg.outputs.model_dir, "subset_df_best.csv"), index=False)
                else:
                    accelerator.print(f">>> patience reached {patience_tracker}/{cfg.train_params.patience}")
                    accelerator.print(f">>> current best score: {round(best_lb, 4)}")

                oof_df.to_csv(os.path.join(cfg.outputs.model_dir, "oof_df_last.csv"), index=False)
                result_df.to_csv(os.path.join(cfg.outputs.model_dir, "result_df_last.csv"), index=False)
                subset_df.to_csv(os.path.join(cfg.outputs.model_dir, "subset_df_last.csv"), index=False)

                # saving -----
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)

                if cfg.save_model:
                    unwrapped_model.save_pretrained(
                        f"{cfg.outputs.model_dir}/last",
                        state_dict=accelerator.get_state_dict(model),
                        save_function=accelerator.save,
                    )

                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(f"{cfg.outputs.model_dir}/last")

                # logging ----
                if cfg.use_wandb:
                    accelerator.log({"lb": lb}, step=current_iteration)
                    accelerator.log({"best_lb": best_lb}, step=current_iteration)
                    for k, v in eval_response["subset_metrics"].items():
                        accelerator.log({f"lb_{k}": round(v, 4)}, step=current_iteration)

                # -- post eval
                model.train()
                torch.cuda.empty_cache()
                print_line()

                # early stopping ----
                if patience_tracker >= cfg.train_params.patience:
                    print("stopping early")
                    model.eval()
                    accelerator.end_training()
                    return

    # --- end training
    accelerator.end_training()


if __name__ == "__main__":
    run_training()
