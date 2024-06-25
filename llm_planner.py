import abc

from openai import OpenAI
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)

from vllm import LLM, RequestOutput, SamplingParams


class PlanningProblem:
    def __init__(
        self,
        natural_language: str,
        domain: str,
        problem: str,
    ):
        """Initializes a new PlanningProblem.

        Args:
            natural_language (str): The natural language description of the
                problem to be solved.
            domain (str): A string representation of the domain of the problem.
            problem (str): A string representation of the ground truth PDDL.
        """
        self.natural_language = natural_language
        self.domain = domain
        self.problem = problem

    def apply_template(
        self,
        domain_prompt: str = "",
        problem_prompt: str = "",
        include_answer: bool = True,
    ) -> list[dict[str, str]]:
        """Apply problem template to the problem.

        Args:
            domain_prompt (str, optional): How to prompt the domain. Defaults to "".
            problem_prompt (str, optional): How to prompt the problem. Defaults to "".
            include_answer (bool, optional): Whether to include the answer. Defaults to True.

        Returns:
            list[dict[str, str]]: Problem prompt.
        """
        return [
            {
                "role": "user",
                "content": f"{problem_prompt} {self.natural_language} "
                + f"{domain_prompt}\n{self.domain}\n",
            },
        ] + (
            [
                {
                    "role": "assistant",
                    "content": " " + self.problem,
                },
            ]
            if include_answer
            else []
        )


class Planner(abc.ABC):
    @abc.abstractmethod
    def plan_chat(
        self,
        messages: list[list[dict[str, str]]],
        device=None,
        max_new_tokens: int = 8_000,
        **kwargs,
    ) -> list[str]:
        """Passes messages to a model for completion.

        Args:
            messages (list[list[dict[str, str]]]): A list of messages to be
                passed to the model.
            device (optional): The device to run the model on. Defaults to None.
            max_new_tokens (int): The maximum number of tokens to generate.
                Defaults to 8_000.

        Returns:
            list[str]: The message completion.
        """
        pass


class HFPlanner:
    """A class for planning using Huggingface transformers."""

    def __init__(
        self,
        model_name: str | None = None,
        tokenizer_name: str | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        model: PreTrainedModel | None = None,
        **kwargs,
    ):
        """Initializes a new HFPlanner.

        Args:
            model_name (str): The name of the model to be used.
            tokenizer_name (str, optional): The name of the tokenizer to be used.
                Defaults to None, in which case the model_name is used.
            kwargs: Additional keyword arguments to be passed to the model.
        """
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    def plan(self, prompt: str, device=None, **kwargs) -> str:
        """Passes the prompt to the model for completion, without applying a
            chat template.

        Args:
            prompt (str): The prompt to be passed to the model.
            device (optional): The device to run the model on. Defaults to None.

        Returns:
            str: The message completion.
        """
        encoded = self.tokenizer.encode(prompt, return_tensors="pt")

        if device is not None:
            encoded = encoded.to(device)

        generate_config = {
            "max_new_tokens": 4000,
            "temperature": 0.0,
            "do_sample": False,
        }
        generate_config.update(kwargs)
        generated_ids = self.model.generate(encoded, **generate_config)

        decoded = self.tokenizer.decode(generated_ids)

        return decoded[0]

    def plan_chat(
        self,
        messages: list[dict[str, str]],
        device=None,
        **kwargs,
    ) -> list[str]:
        """Passes messages to the model for completion, applying a chat template.

        Args:
            messages (list[dict[str, str]]): A list of messages to be passed to
                the model.
            device (optional): The device to run the model on. Defaults to None.
            kwargs: Additional keyword arguments to be passed to the model.

        Returns:
            str: The message completion.
        """
        encoded = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )

        if device is not None:
            encoded = encoded.to(device)

        generate_config = {  # default generate config
            "max_new_tokens": 4000,
            "do_sample": False,
        }
        generate_config.update(kwargs)
        with torch.no_grad():
            generated_ids = self.model.generate(encoded, **generate_config)

        decoded = []
        for g, e in zip(generated_ids, encoded):
            decoded.append(
                self.tokenizer.decode(
                    g[len(e) :],
                    skip_special_tokens=True,
                )
            )

        return decoded


class VLLMPlanner(Planner):
    """A class for planning using VLLM models."""

    def __init__(self, model_name: str, **kwargs):
        """Initializes a new VLLMPlanner.

        Args:
            model_name (str): The name of the model to be used.
            kwargs: Additional keyword arguments to be passed to the model.
        """
        self.model = LLM(model_name, **kwargs)
        self.tokenizer = self.model.get_tokenizer()

    def plan_chat(
        self,
        messages: list[list[dict[str, str]]],
        device=None,
        max_new_tokens: int = 8_000,
        **kwargs,
    ) -> list[str]:
        """Passes messages to the model for completion.

        Args:
            messages (list[dict[str, str]]): A list of messages to be passed to
                the model.

        Returns:
            list[str]: The message completions.
        """
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            min_p=kwargs.get("min_p", 0.0),
        )

        outputs: list[RequestOutput] = self.model.generate(
            encoded,
            params,
            use_tqdm=False,
        )

        return [output.outputs[0].text for output in outputs]


class OpenAIPlanner:
    """A class for planning using OpenAI models."""

    def __init__(self, model_name: str, **kwargs):
        """Initializes a new OpenAIPlanner.

        Args:
            model_name (str): The name of the model to be used.
            kwargs: Additional keyword arguments to be passed to the OpenAI
                client.
        """
        self.client = OpenAI(**kwargs)
        self.model_name = model_name

    def _plan_chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int | None = None,
        device=None,
        **kwargs,
    ) -> str:
        """Passes messages to the model for completion.

        Args:
            messages (list[dict[str, str]]): A list of messages to be passed to
                the model.
            device (optional): The device to run the model on (ignored for OpenAI).

        Returns:
            str: The message completion.
        """

        return (
            self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                frequency_penalty=kwargs.get("frequency_penalty", None),
                max_tokens=max_new_tokens,
                n=1,
                presence_penalty=kwargs.get("presence_penalty", None),
                temperature=kwargs.get("temperature", 0.0),
                top_p=kwargs.get("top_p", None),
            )
            .choices[0]
            .message.content
        )

    def plan_chat(
        self,
        messages: list[list[dict[str, str]]],
        max_new_tokens: int | None = None,
        device=None,
        **kwargs,
    ) -> list[str]:
        """Passes messages to the model for completion.

        Args:
            messages (list[list[dict[str, str]]]): A list of messages to be
                passed to the model.
            device (optional): The device to run the model on (ignored for OpenAI).

        Returns:
            list[str]: The message completions.
        """
        return [
            self._plan_chat(
                message,
                max_new_tokens=max_new_tokens,
                device=device,
                **kwargs,
            )
            for message in messages
        ]