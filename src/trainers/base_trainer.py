"""Trainer class for training models with Next Token Prediction"""

import datetime
import os
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional

import json5
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

import wandb
from models import model_shell
from models.generator import StandardGenerator
from trainers.evaluator import train_eval
from trainers.utils import (
    aggregate_value,
    print_evaluation_results,
    profilize,
    set_seed,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseTrainer:
    """Base Trainer Class

    Uses subcomponents: optimizer, scheduler,
    model, dataloader, loss functions, logger
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        model: model_shell.ModelShell,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        loss_fn: callable,
        max_epochs: float,
        max_iters: int,
        is_iters_based: bool,
        iters_per_epoch: int,
        dataset_size: int,
        gpu_id: Optional[int] = None,
        lr_scheduler: Optional[Any] = None,
        dropout_scheduler: Optional[Any] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize BaseTrainer with proper validation and setup."""
        # Validate critical requirements early
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA must be available for training")

        # Store core components
        self.cfg = cfg
        self.model = model
        self.gpu_id = gpu_id
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.dropout_scheduler = dropout_scheduler
        self.total_model_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total model parameters: {self.total_model_params:,}")

        # Setup distributed training
        self.dist = gpu_id is not None
        if self.dist:
            self.DDP_model = DDP(self.model, device_ids=[gpu_id])
        else:
            self.DDP_model = model

        # Setup data loaders
        self.train_dataloader_iter = iter(train_dataloader)
        self.val_dataloader = val_dataloader
        self.dataset_size = dataset_size

        # Store training configuration parameters (calculated in build_trainer)
        training_cfg = cfg["trainer"]["training"]
        self.batch_size = training_cfg["batch_size"]
        self.max_epochs = max_epochs
        self.max_iters = max_iters
        self.is_iters_based = is_iters_based
        self.iters_per_epoch = iters_per_epoch
        self.perform_injection = cfg["trainer"]["inject"]["perform_injection"]

        # Calculate gradient accumulation steps
        base_grad_steps = training_cfg["gradient_accumulation_steps"]
        if torch.cuda.is_available() and self.dist:
            self.gradient_accumulation_steps = (
                base_grad_steps // torch.cuda.device_count()
            )
        else:
            self.gradient_accumulation_steps = base_grad_steps

        # Initialize training state
        self.scaler = None
        self.run_id = None
        if checkpoint is not None:
            self.epoch_start = checkpoint["epoch"]
            self.iter_start = checkpoint["iteration"]
            self.run_id = checkpoint.get("wandb_run_id", None)
            logger.info(
                f"Resuming training from epoch {self.epoch_start}, iteration {self.iter_start}"
            )
        else:
            self.epoch_start = 0
            self.iter_start = 1
            logger.info("Starting training from scratch")

        # Setup logging configuration
        self.use_wandb = cfg["general"]["logging"]["wandb_log"]
        self.checkpoint_dir = cfg["general"]["paths"]["checkpoint_dir"]
        self.table = wandb.Table(
            columns=["epoch", "iteration", "text"], log_mode="MUTABLE"
        )
        injected_prompts_path = os.path.join(
            cfg["general"]["paths"]["data_dir"],
            "inject",
            cfg["trainer"]["inject"]["injected_prompts"],
        )
        if os.path.exists(injected_prompts_path):
            with open(injected_prompts_path, "r") as f:
                self.injected_prompts = json5.load(f)

        # Setup training context (moved to separate method - this IS complex)
        self.ctx = self._setup_ctx(checkpoint=checkpoint)

        # Handle initialization that should only run on main process
        if self._is_main_process():
            if self.use_wandb:
                self._setup_logging()

            if training_cfg.get("run_profiler", False):
                logger.info("Running profiler and exiting...")
                self.run_profile()
                raise SystemExit("Profiling completed")

    def _is_main_process(self) -> bool:
        """Check if this is the main process for logging/profiling."""
        return self.gpu_id == 0 or not self.dist

    def format_number(self, num: int) -> str:
        """Format a number with appropriate suffix (K, M, B, T).

        Args:
            num: The number to format

        Returns:
            Formatted string with appropriate suffix
        """
        if num < 1000:
            return str(num)
        dict_format = {
            0: "",
            1: "K",
            2: "M",
            3: "B",
            4: "T",
        }
        num_div = 0
        aux = num
        while True:
            aux = aux / 1000
            if aux < 1:
                break
            num = aux
            num_div += 1
        return f"{num:.2f}{dict_format[num_div]}"

    def _setup_logging(self):
        """Setup wandb logging with comprehensive run naming."""
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        iters_or_epochs = "iters" if self.is_iters_based else "epochs"
        max_value = self.max_iters if self.is_iters_based else self.max_epochs
        run_name = (
            f"{current_time}"
            f"_{self.cfg.trainer['dataset']}"
            f"_{self.format_number(self.total_model_params)}_params"
            f"_{self.format_number(self.dataset_size)}_tokens"
            f"_{self.format_number(max_value)}_{iters_or_epochs}"
        )
        if self.run_id is not None:
            wandb.init(
                project=self.cfg.general.logging.wandb_project,
                config=OmegaConf.to_container(self.cfg),
                name=run_name,
                id=self.run_id,
                resume="must",
            )
            logger.info(f"Resuming Weights & Biases run with ID: {self.run_id}")
        else:
            run = wandb.init(
                project=self.cfg.general.logging.wandb_project,
                config=OmegaConf.to_container(self.cfg),
                name=run_name,
            )
            self.run_id = run.id
        logger.info("Weights & Bias initialized.")

    def _setup_ctx(self, checkpoint=None):
        """Get the context manager"""
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        self.scaler = self._setup_scaler(dtype)
        if checkpoint is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
            logger.info("Loaded scaler state from checkpoint.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # torch.backends.cuda.matmul.fp32_precision = "tf32"
        # torch.backends.cudnn.conv.fp32_precision = "tf32"

        ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

        return ctx

    def print_with_rank(self, rank, *arg):
        print(f"[RANK {rank}]", *arg)

    def _setup_scaler(self, dtype=torch.float16):
        """Setup the scaler"""
        scaler = torch.amp.GradScaler(device="cuda", enabled=dtype == torch.float16)
        return scaler

    def _get_scheduler_state(self, scheduler):
        """Get the state of a scheduler for checkpointing.

        Args:
            scheduler: The scheduler object to get state from

        Returns:
            dict: Scheduler state dictionary, or None if scheduler is None
        """
        if scheduler is None:
            return None

        # Try to get state_dict if available (for PyTorch schedulers)
        if hasattr(scheduler, "state_dict"):
            return scheduler.state_dict()

        # For custom schedulers, save their instance variables
        state = {}
        for attr_name in dir(scheduler):
            if not attr_name.startswith("_") and not callable(
                getattr(scheduler, attr_name)
            ):
                try:
                    attr_value = getattr(scheduler, attr_name)
                    # Only save simple types that can be serialized
                    if isinstance(
                        attr_value, (int, float, str, bool, list, tuple, dict)
                    ):
                        state[attr_name] = attr_value
                except Exception:
                    # Skip attributes that can't be accessed or serialized
                    continue

        return state

    def _load_scheduler_state(self, scheduler, state):
        """Load state into a scheduler from checkpoint.

        Args:
            scheduler: The scheduler object to load state into
            state: The state dictionary to load
        """
        if scheduler is None or state is None:
            return

        # Try to use load_state_dict if available (for PyTorch schedulers)
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(state)
            return

        # For custom schedulers, restore their instance variables
        for attr_name, attr_value in state.items():
            if hasattr(scheduler, attr_name):
                try:
                    setattr(scheduler, attr_name, attr_value)
                except Exception:
                    # Skip attributes that can't be set
                    continue

    @torch.no_grad()
    def estimate_performance(
        self, eval_iters: int = None, verbose: bool = False
    ) -> tuple[dict, dict]:
        """Estimate the loss and perplexity on the validation set, plus evaluator metrics.

        Args:
            eval_iters (int, optional): Number of evaluation iterations. Defaults to config value.
            verbose (bool, optional): If True, logs detailed info. Defaults to False.

        Returns:
            tuple[dict, dict]: (eval_results, evaluator_results)
        """
        if verbose:
            logger.info("estimate_performance: called")
        if eval_iters is None:
            eval_iters = self.cfg.trainer.training.eval_iters
        if verbose:
            logger.info(f"estimate_performance: eval_iters={eval_iters}")
        eval_results: dict = {}
        self.model.eval()

        # eval on val set
        losses = []
        perplexities = []

        for i, (x, y) in enumerate(self.val_dataloader):
            if verbose:
                logger.info(f"estimate_performance: batch {i}")
            x = x.to(self.gpu_id if self.gpu_id is not None else self.model.device)
            y = y.to(self.gpu_id if self.gpu_id is not None else self.model.device)
            with self.ctx:
                output, _ = self.model(x)
                loss = self.loss_fn(output, y)
                if verbose:
                    logger.info(f"estimate_performance: loss={loss.item()}")
                losses.append(loss.item())
                perplexity = torch.exp(loss)
                if verbose:
                    logger.info(f"estimate_performance: perplexity={perplexity.item()}")
                perplexities.append(perplexity.item())
            if i >= eval_iters:
                if verbose:
                    logger.info(
                        "estimate_performance: reached eval_iters limit, breaking"
                    )
                break

        avg_loss = aggregate_value(np.mean(losses), self.cfg.general.device)

        if verbose:
            logger.info(f"estimate_performance: avg_loss={avg_loss}")
        eval_results["Loss"] = avg_loss

        avg_perplexity = aggregate_value(np.mean(perplexities), self.cfg.general.device)

        if verbose:
            logger.info(f"estimate_performance: avg_perplexity={avg_perplexity}")
        eval_results["Perplexity"] = avg_perplexity

        evaluator_results: dict = {}
        for evaluator_cfg in self.cfg.trainer["eval"]:
            if verbose:
                logger.info(
                    f"estimate_performance: running evaluator {evaluator_cfg['evaluator']}"
                )
            evaluator_results[evaluator_cfg["evaluator"]] = train_eval(
                evaluator_cfg, self.model
            )
            relabeled_results = {}
            for metric in evaluator_results[evaluator_cfg["evaluator"]]:
                relabeled_results[f"{evaluator_cfg['evaluator']}/{metric}"] = (
                    evaluator_results[evaluator_cfg["evaluator"]][metric]
                )
            evaluator_results[evaluator_cfg["evaluator"]] = relabeled_results

        self.model.train()

        if verbose:
            logger.info("estimate_performance: returning results")

        return eval_results, evaluator_results

    def run_profile(self):
        """Run the profiler"""
        profilize(self.model)
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for i in range(10):
                if i <= 3:
                    self._run_step()  ## set the 'epoch' to ensure shuffle
                else:
                    with record_function("_run_step"):
                        self._run_step()  ## set the 'epoch' to ensure shuffle
            # place profile in dictionary
        backwards_prof = prof.key_averages().table(sort_by="self_cpu_time_total")
        print(backwards_prof)
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            self.estimate_performance(eval_iters=1)
            with record_function("estimate_performance"):
                self.estimate_performance(eval_iters=10)
            # place profile in dictionary
        forwards_prof = prof.key_averages().table(sort_by="self_cpu_time_total")
        print(forwards_prof)

    def save_checkpoint(self, iteration: int, epoch: int, verbose: bool = True) -> None:
        """Save a comprehensive checkpoint for resuming training.

        Args:
            iteration (int): The current training iteration number.
            verbose (bool, optional): If True, logs checkpoint saving info. Defaults to True.
        """
        checkpoint = {
            # Model state
            "model": self.model.state_dict(),
            # Optimizer state (includes momentum, learning rate history, etc.)
            "optimizer": self.optimizer.state_dict(),
            # Training progress
            "iteration": iteration + 1,  # Next iteration to run
            "epoch": epoch,
            # Schedulers state
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "dropout_scheduler": self.dropout_scheduler.state_dict(),
            # Configuration
            "config": self.cfg,
            # Scaler state for mixed precision training
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            # Run ID for wandb resumption
            # "wandb_run_id": self.run_id if self.use_wandb else None,
        }

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        iters_or_epochs = "iters" if self.is_iters_based else "epochs"
        save_value = iteration if self.is_iters_based else epoch

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        checkpoint_path = (
            f"{self.checkpoint_dir}/"
            f"{current_time}"
            f"_{self.cfg.trainer['dataset']}"
            f"_{save_value}_{iters_or_epochs}.pt"
        )

        if verbose:
            logger.info(f"Saving comprehensive checkpoint to {checkpoint_path}")

        torch.save(checkpoint, checkpoint_path)

        if verbose:
            logger.info(
                f"Checkpoint saved successfully at {'iteration' if self.is_iters_based else 'epoch'} {save_value}"
            )

    def run_prompting_table(self, prompt_cfg) -> str:
        """Generate answers for a set of prompts using the model.

        Returns them as a formatted string.

        Args:
            prompt_cfg (dict): Configuration containing 'generator' settings and
                'input_prompts' list.

        Returns:
            str: A formatted string containing prompts and their generated answers.
        """
        generator = StandardGenerator(
            model=self.model, generate_cfg=prompt_cfg["generator"]
        )
        generated = ""
        for prompt_num, prompt in enumerate(prompt_cfg["input_prompts"], start=1):
            generated_text, messages = generator.default_generate(
                input_text=prompt["sentence"]
            )
            probs, perplexity = generator.evaluate_perplexity(
                prompt["sentence"],
                prompt["answer"],
                temperature=prompt_cfg["generator"]["temperature"],
                top_k=prompt_cfg["generator"]["top_k"],
            )
            generated += (
                f"Question {prompt_num}\n\n"
                f"Prompt:\n{prompt['sentence']}\n\n"
                f"Generated:\n{generated_text[0]}\n\n"
                f"Answer:\n{prompt['answer']}\n\n"
                f"Probability of correct answer: {probs}\n\n"
                f"Perplexity of correct answer: {perplexity:.4f}\n\n"
            )
            generated += generator._format_messages(
                messages, prompt_cfg["generator"]["steps_to_log"]
            )
            generated += "=" * 30 + "\n\n"
        return generated

    def _run_step(self):
        """Run a single step of training with gradient accumulation."""
        self.optimizer.zero_grad()  # Clear gradients at the start of accumulation

        accumulated_loss = 0
        for i in range(self.gradient_accumulation_steps):
            # get the next batch
            x, y = next(self.train_dataloader_iter)
            x = x.to(self.gpu_id if self.gpu_id is not None else self.model.device)
            y = y.to(self.gpu_id if self.gpu_id is not None else self.model.device)

            # Enable or disable gradient synchronization based on the need for accumulation
            if self.dist and hasattr(self.DDP_model, "no_sync"):
                context_manager = (
                    self.DDP_model.no_sync()
                    if i != self.gradient_accumulation_steps - 1
                    else nullcontext()
                )
            else:
                context_manager = nullcontext()

            with context_manager:
                with self.ctx:
                    output, aux_loss = self.DDP_model(x)
                    loss = self.loss_fn(output, y)
                    if aux_loss is not None:
                        loss += aux_loss

                # Scale loss to simulate larger effective batch size
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()

        # once graidents are accumulated, step
        if self.cfg.trainer.optimizer.grad_clip > 0:
            # Unscale the gradients of the optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # Clip the gradients with normalization
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.trainer.optimizer.grad_clip
            )

        # Perform a single optimization step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()  # Reset gradients after update

        return accumulated_loss

    def _should_log(self, iter_num: int, interval: int) -> bool:
        """Check if we should log at this iteration based on the training mode."""
        if not self._is_main_process():
            return False
        elif interval <= 0:
            return False
        # elif iter_num == self.iter_start:
        #     return True
        elif self.is_iters_based:
            return not iter_num % interval
        else:
            return not iter_num % (self.iters_per_epoch * interval)

    def _log_training_progress(
        self,
        iter_num: int,
        epoch: int,
        lossf: float,
        lr: float,
        dropout: float,
        step_time: float,
        elapsed_time: float,
    ):
        """Log training progress to console and wandb."""
        lossf = aggregate_value(lossf, self.cfg.general.device)
        if self._is_main_process():
            elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            logger.info(
                f"All GPU(s): Epoch {epoch}/{self.max_epochs:.0f} | "
                f"Step {iter_num} | Loss: {lossf:.4f} | LR: {lr:.1e} | "
                f"Dropout: {dropout:.2f} | Step time: {step_time:.2f}s | "
                f"Total time: {elapsed_time_str}"
            )
            if self.use_wandb:
                return {
                    # "epoch": epoch,
                    # "iter": iter_num,
                    "loss": lossf,
                    "lr": lr,
                    "dropout": dropout,
                }
            else:
                return {}
        return {}

    def _handle_prompting(self, epoch, iteration: int):
        """Handle periodic prompting if configured."""
        if self.use_wandb and self._is_main_process():
            logger.info(f"Running prompting at epoch {epoch}, iteration {iteration}")
            generated = self.run_prompting_table(self.cfg.trainer.prompt)
            self.table.add_data(epoch, iteration, generated)
            # wandb.log({"prompt_answer_table": self.table})
            return {"prompt_answer_table": self.table}
        else:
            return {}

    def _handle_evaluation(self, iter_num: int):
        """Handle periodic evaluation if configured."""
        self.print_with_rank(f"DEBUG: Starting evaluation at iter {iter_num}")
        eval_results, benchmark_results = self.estimate_performance(verbose=True)
        self.print_with_rank(f"DEBUG: Evaluation completed at iter {iter_num}")
        if self._is_main_process():
            print_evaluation_results(
                iter_num=iter_num,
                eval_results=eval_results,
                benchmark_results=benchmark_results,
            )
            if self.use_wandb:
                log_dict = {**eval_results}
                log_dict.update(benchmark_results)
                # logger.info(f"Logging evaluation results to wandb: {log_dict}")
                # wandb.log(log_dict)
                return log_dict
            else:
                return {}
        return {}

    def _handle_checkpointing(self, iter_num: int, epoch: int):
        """Handle periodic checkpointing if configured."""
        if self._is_main_process():
            self.save_checkpoint(iter_num, epoch)

    def run_injected_evaluation(self, generator_cfg):
        generator = StandardGenerator(model=self.model, generate_cfg=generator_cfg)

        res = {"injected": {}}
        for type in self.injected_prompts.keys():
            # logger.info(f"Evaluating injected prompts of type: {type}")
            type_name = (
                "injected/" + type
            )  # So that the section 'injected' is separate in wandb

            res["injected"][type_name] = {}
            ranks = []
            perplexities = []

            if self.injected_prompts[type] == []:
                logger.warning(f"No injected prompts found for type: {type}")
                continue

            for prompt in self.injected_prompts[type]:
                _, perplexity = generator.evaluate_perplexity(
                    prompt["prompt"],
                    prompt["completion"],
                    temperature=generator_cfg["temperature"],
                    top_k=generator_cfg["top_k"],
                )
                _, avg_rank = generator.evaluate_rank(
                    prompt["prompt"],
                    prompt["completion"],
                    temperature=generator_cfg["temperature"],
                    top_k=generator_cfg["top_k"],
                )
                ranks.append(avg_rank)
                perplexities.append(perplexity)

                # logger.info(
                #     f"Prompt: {prompt['prompt']}\n"
                #     f"Completion: {prompt['completion']}\n"
                #     f"Perplexity: {perplexity:.4f}\n"
                #     f"Average Rank: {avg_rank}\n"
                # )

            # logger.info(ranks)
            # logger.info(perplexities)

            avg_rank = sum(ranks) / len(ranks)
            avg_perplexity = sum(perplexities) / len(perplexities)

            res["injected"][type_name]["rank_average"] = avg_rank
            res["injected"][type_name]["perplexity_average"] = avg_perplexity

            logger.info(
                f"Type: {type} - Average Rank: {avg_rank:.4f}, Average Perplexity: {avg_perplexity:.4f}"
            )

        return res

    def _handle_injected_evaluation(self):
        """Run evaluation on injected prompts."""
        if self._is_main_process() and self.use_wandb:
            res = self.run_injected_evaluation(self.cfg.trainer.prompt.generator)
            # logger.info(f"Injected evaluation results: {res}")
            # logger.info(f"Logging injected evaluation results to wandb: {res}")
            # wandb.log(res)
            return res
        else:
            return {}

    def run_training_loop(self):
        """Execute the main training loop with periodic evaluation, checkpointing and logging."""
        epoch = self.epoch_start
        elapsed_time = 0.0

        for iter_num in tqdm(
            range(self.iter_start, self.max_iters + 1), desc="Training"
        ):
            self.print_with_rank(f"DEBUG: Starting iteration {iter_num}, epoch {epoch}")
            start_time = time.time()

            if self.lr_scheduler is not None:
                lr = self.lr_scheduler.step(self.optimizer, iter_num - 1)
                self.print_with_rank(f"DEBUG: LR scheduler stepped, lr = {lr}")
            else:
                lr = self.optimizer.param_groups[0]["lr"]
                self.print_with_rank(f"DEBUG: No LR scheduler, lr = {lr}")

            dropout = self.dropout_scheduler.step(self.model, iter_num - 1)
            self.print_with_rank(
                f"DEBUG: Dropout scheduler stepped, dropout = {dropout}"
            )

            # Training step
            lossf = self._run_step()
            self.print_with_rank(f"DEBUG: Training step completed, lossf = {lossf}")
            end_time = time.time()
            step_time = end_time - start_time
            elapsed_time += step_time
            print(
                f"DEBUG: Step time = {step_time:.2f}s, elapsed_time = {elapsed_time:.2f}s"
            )

            if self.iters_per_epoch > 0 and not iter_num % self.iters_per_epoch:
                epoch += 1
                self.print_with_rank(f"DEBUG: Epoch incremented to {epoch}")

            master_log_dict = {"epoch": epoch, "iter": iter_num}
            self.print_with_rank(
                f"DEBUG: Master log dict initialized: {master_log_dict}"
            )

            # Periodic logging
            if self._should_log(iter_num, self.cfg.trainer.training.log_interval):
                self.print_with_rank(
                    f"DEBUG: Should log training progress at iter {iter_num}"
                )
                train_metrics = self._log_training_progress(
                    iter_num, epoch, lossf, lr, dropout, step_time, elapsed_time
                )
                master_log_dict.update(train_metrics)
                self.print_with_rank(f"DEBUG: Train metrics updated: {train_metrics}")

            # Periodic evaluation
            if self._should_log(iter_num, self.cfg.trainer.training.eval_interval):
                self.print_with_rank(f"DEBUG: Should evaluate at iter {iter_num}")
                eval_metrics = self._handle_evaluation(iter_num)
                master_log_dict.update(eval_metrics)
                self.print_with_rank(f"DEBUG: Eval metrics updated: {eval_metrics}")

            # Periodic prompting
            if self._should_log(iter_num, self.cfg.trainer.training.prompt_interval):
                self.print_with_rank(
                    f"DEBUG: Should handle prompting at iter {iter_num}"
                )
                prompt_metrics = self._handle_prompting(epoch, iter_num)
                master_log_dict.update(prompt_metrics)
                self.print_with_rank(f"DEBUG: Prompt metrics updated: {prompt_metrics}")

            # Periodic injected evaluation
            if self._should_log(
                iter_num, self.cfg.trainer.training.injected_eval_interval
            ):
                self.print_with_rank(
                    f"DEBUG: Should handle injected evaluation at iter {iter_num}"
                )
                injected_metrics = self._handle_injected_evaluation()
                master_log_dict.update(injected_metrics)
                self.print_with_rank(
                    f"DEBUG: Injected metrics updated: {injected_metrics}"
                )

            # Log to wandb if enabled
            if self.use_wandb and self._is_main_process():
                self.print_with_rank(f"DEBUG: Logging to wandb: {master_log_dict}")
                wandb.log(master_log_dict)

            # Periodic checkpointing
            if self._should_log(
                iter_num, self.cfg.trainer.training.checkpoint_interval
            ):
                self.print_with_rank(f"DEBUG: Should checkpoint at iter {iter_num}")
                self._handle_checkpointing(iter_num, epoch)
            self.print_with_rank(f"DEBUG: End of iteration {iter_num}")

    def train(self, seed):
        """Start training with the given random seed.

        Args:
            seed: Random seed for reproducible training
        """
        set_seed(seed)
        logger.info(
            f"Training for {self.max_epochs:.4f} epochs and {self.max_iters} iterations"
        )
        self.run_training_loop()
