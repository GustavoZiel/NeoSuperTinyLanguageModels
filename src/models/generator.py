"""Generator Base Wrapper"""

import torch

from core.logger import get_logger
from training.utils import set_seed

logger = get_logger(__name__)


class StandardGenerator(torch.nn.Module):
    """Standard Generator Wrapper for GPT models.

    Handles text generation, perplexity evaluation, and rank evaluation.
    Supports both custom models with `embedding_model` and Hugging Face models with `tokenizer`.
    """

    def __init__(self, model, generate_cfg, tokenizer=None):
        """Initialize the generator.

        Args:
            model (torch.nn.Module): The language model.
            generate_cfg (dict): Generation configuration.
            tokenizer (PreTrainedTokenizer, optional): Hugging Face tokenizer.
        """
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)

        self.tokenizer = tokenizer
        self.generate_config = generate_cfg

        set_seed(self.generate_config["seed"])

    def default_generate(self, input_text):
        """Generate text using the default configuration.

        Args:
            input_text (str): The prompt text.

        Returns:
            tuple: (decoded_text, messages)
        """
        return self.generate(
            input_text,
            self.generate_config["max_new_tokens"],
            self.generate_config["temperature"],
            self.generate_config["top_k"],
        )

    def _format_messages(self, messages, steps_to_log):
        """Formats the generation log messages.

        Args:
            messages (list): List of log messages from generation steps.
            steps_to_log (int): Number of steps to include in the formatted output.

        Returns:
            str: Formatted string containing the log messages.
        """
        return "".join(message + "\n" for message in messages[:steps_to_log])

    @torch.no_grad()
    def evaluate_rank(self, input_text, correct_answer, temperature=1.0, top_k=None):
        """Evaluate the rank of the correct answer tokens given the prompt.

        Args:
            input_text (str): The prompt.
            correct_answer (str): The expected continuation.
            temperature (float): Sampling temperature.
            top_k (int): Top-k filtering.

        Returns:
            tuple: (list of ranks, average rank)
        """
        original_mode = self.model.training
        self.model.eval()

        try:
            if self.tokenizer is None:
                idx = self.model.embedding_model.tokenize_input(
                    input_string=input_text, add_eot=False, truncate=True
                )
                idx = torch.tensor(idx).unsqueeze(0).to(self.model.device)

                idx_correct = self.model.embedding_model.tokenize_input(
                    input_string=correct_answer, add_eot=False, truncate=True
                )
                idx_correct = (
                    torch.tensor(idx_correct).unsqueeze(0).to(self.model.device)
                )
            else:
                idx = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                ).input_ids.to(self.model.device)

                idx_correct = self.tokenizer(
                    correct_answer,
                    return_tensors="pt",
                    padding=True,
                ).input_ids.to(self.model.device)

            ranks_correct = []
            for i in range(idx_correct.shape[1]):
                if self.tokenizer is None:
                    logits, model_input = self.model.inference(idx)
                else:
                    logits = self.model(idx).logits[:, -1, :]

                # Comment or uncomment to enable/disable temperature scaling and top-k filtering during rank evaluation
                # logits = logits / temperature
                # if top_k is not None:
                #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                #     if len(v.size()) == 3:
                #         logits[logits < v[:, :, [-1]]] = -float("Inf")
                #     else:
                #         logits[logits < v[:, [-1]]] = -float("Inf")

                sorted_indices = torch.argsort(logits[0], descending=True)

                # Get the position (rank) of the correct index
                rank = (sorted_indices == idx_correct[0, i]).nonzero(as_tuple=True)[
                    0
                ].item() + 1  # +1 for 1-based rank
                ranks_correct.append(rank)

                idx = torch.cat((idx, idx_correct[:, i].unsqueeze(0)), dim=1)

            return ranks_correct, sum(ranks_correct) / len(ranks_correct)

        finally:
            self.model.train(original_mode)

    @torch.no_grad()
    def evaluate_perplexity(
        self, input_text, correct_answer, temperature=1.0, top_k=None
    ):
        """Evaluate the perplexity of the correct answer given the prompt.

        Args:
            input_text (str): The prompt.
            correct_answer (str): The expected continuation.
            temperature (float): Sampling temperature.
            top_k (int): Top-k filtering.

        Returns:
            tuple: (list of probabilities, perplexity score)
        """
        original_mode = self.model.training
        self.model.eval()

        try:
            if self.tokenizer is None:
                idx = self.model.embedding_model.tokenize_input(
                    input_string=input_text, add_eot=False, truncate=True
                )
                idx = torch.tensor(idx).unsqueeze(0).to(self.model.device)

                idx_correct = self.model.embedding_model.tokenize_input(
                    input_string=correct_answer, add_eot=False, truncate=True
                )
                idx_correct = (
                    torch.tensor(idx_correct).unsqueeze(0).to(self.model.device)
                )
            else:
                idx = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                ).input_ids.to(self.model.device)

                idx_correct = self.tokenizer(
                    correct_answer,
                    return_tensors="pt",
                    padding=True,
                ).input_ids.to(self.model.device)

            probs_correct = []
            for i in range(idx_correct.shape[1]):
                if self.tokenizer is None:
                    logits, model_input = self.model.inference(idx)
                else:
                    logits = self.model(idx).logits[:, -1, :]

                # Comment or uncomment to enable/disable temperature scaling and top-k filtering during perplexity evaluation
                # logits = logits / temperature
                # if top_k is not None:
                #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                #     if len(v.size()) == 3:
                #         logits[logits < v[:, :, [-1]]] = -float("Inf")
                #     else:
                #         logits[logits < v[:, [-1]]] = -float("Inf")

                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs_correct.append(probs[0, idx_correct[0, i]].item())
                idx = torch.cat((idx, idx_correct[:, i].unsqueeze(0)), dim=1)

            probs = torch.tensor(probs_correct)
            probs = torch.clamp(probs, min=1e-12)  # avoid log(0)

            # Negative log-likelihood (mean per token)
            nll_mean = -torch.mean(torch.log(probs))

            # Perplexity = exp(mean NLL)
            ppl = torch.exp(nll_mean)

            return [round(float(x), 4) for x in probs.tolist()], ppl.item()

        finally:
            self.model.train(original_mode)

    @torch.no_grad()
    def generate(self, input_text, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text from the model.

        Args:
            input_text (str): The prompt.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int): Top-k filtering.

        Returns:
            tuple: (decoded_message, messages)
        """
        original_mode = self.model.training
        self.model.eval()

        try:
            if self.tokenizer is None:
                idx = self.model.embedding_model.tokenize_input(
                    input_string=input_text, add_eot=False, truncate=True
                )
                # push to device
                idx = torch.tensor(idx).unsqueeze(0).to(self.model.device)
            else:
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                idx = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                ).input_ids.to(self.model.device)

            messages = []
            for i_token in range(max_new_tokens):
                message = f"Step {i_token + 1}:\n"

                # forward the model to get the logits for the index in the sequence
                if self.tokenizer is None:
                    logits, model_input = self.model.inference(idx)
                else:
                    logits = self.model(idx).logits[:, -1, :]

                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature

                # logits have shape (b,t,v)
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # check for dim
                    if len(v.size()) == 3:
                        logits[logits < v[:, :, [-1]]] = -float("Inf")
                    else:
                        logits[logits < v[:, [-1]]] = -float("Inf")

                # apply softmax to convert logits to (normalized) probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)

                # sample from the distribution
                # check if byte-level and if so, flatten
                if len(probs.size()) == 4:
                    B, S, S_c, H = probs.size()
                    probs = probs.view(B * S * S_c, H)
                    flattened = True
                else:
                    flattened = False

                # For every i_token, collect top_k token info and format messages
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                for i_idx, i_prob in zip(
                    top_k_indices.flatten(), top_k_probs.flatten()
                ):
                    if self.tokenizer is not None:
                        decoded_token = self.tokenizer.batch_decode(i_idx.view(1, -1))
                    else:
                        decoded_token = self.model.embedding_model.decode(
                            i_idx.view(1, -1)
                        )
                    message += f"Token: {decoded_token}, Probability: {i_prob.item():.4f}, Index: {i_idx.item()}\n"

                idx_next = torch.multinomial(probs, num_samples=1)

                # check if byte-level and if so, unflatten
                if flattened:
                    idx_next = idx_next.view(B, S)
                else:
                    if self.tokenizer is not None:
                        if idx_next == self.tokenizer.eos_token_id:
                            break
                    else:
                        if idx_next == self.model.embedding_model.eot_token:
                            break

                if flattened:
                    idx_next = idx_next.unsqueeze(0)
                idx = torch.cat((idx, idx_next), dim=1)

                messages.append(message)

            decoded_message = ""
            if self.tokenizer is not None:
                decoded_message = self.tokenizer.batch_decode(idx.tolist())
            else:
                decoded_message = self.model.embedding_model.decode(idx.tolist())

            return decoded_message, messages

        finally:
            # Always restore the original training state
            self.model.train(original_mode)

    def forward(self, x):
        """Call the underlying model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output.
        """
        return self.model(x)

    def embed(self, x):
        """Embed the input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Embedded input.
        """
        return self.model.embed(x)
