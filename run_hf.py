import torch, copy
import torch.nn.functional as F
from typing import Optional, List
from transformers import UnbatchedClassifierFreeGuidanceLogitsProcessor
from transformers.generation.utils import *
from transformers.generation.logits_process import *

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationMixin
from transformers.generation.logits_process import UnbatchedClassifierFreeGuidanceLogitsProcessor
from transformers import GenerationConfig


def get_entropy(logits):
    B, S, E = logits.shape
    with torch.no_grad(): 
        # logits = logits.half() 
        probs = F.softmax(logits, dim=-1)
        entropy_list = None
        for i in range(B):
            probs_chunk = probs[i: i + 1]
            entropy__ = probs_chunk * torch.log(probs_chunk)
            entropy_ = torch.where(
                ~torch.isnan(entropy__),
                entropy__,
                torch.zeros_like(entropy__)
            )
            entropy = -torch.sum(entropy_, dim=-1)# / math.log(E)

            entropy_list = entropy if entropy_list is None else torch.cat([entropy_list, entropy], dim=0)
    return entropy_list

class MTI(UnbatchedClassifierFreeGuidanceLogitsProcessor):

    def __init__(
        self,
        guidance_scale: float,
        model,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        entropy_threshold=None,
        tokenizer__=None,
        lightweight_negative_prompt=None,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.entropy_threshold = entropy_threshold
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }
        self.tokenizer__ = tokenizer__
        self.lightweight_negative_prompt = lightweight_negative_prompt

    def get_unconditional_logits(self, input_ids):
        mask_ = self.model.cfg_mask__[:, -1]
        uc_ = self.tokenizer__.batch_encode_plus([self.lightweight_negative_prompt] * input_ids.shape[0])
        uc_ = {k: torch.tensor(v).to(self.model.device) for k, v in uc_.items()}
        uc_input_ids_, uc_attention_mask_ = uc_['input_ids'], uc_['attention_mask']

        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = torch.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        if mask_.sum() == 0:
            return None

        indices_ = mask_.nonzero(as_tuple=False).squeeze(1)
        kv_cache_ = copy.deepcopy(self.kv_cache_)
        kv_cache_.batch_select_indices(indices_)

        attention_mask_c_uc_ = torch.cat([self.unconditional_context["attention_mask"], uc_attention_mask_], dim=1)[mask_]
        position_ids_ = (attention_mask_c_uc_.long().cumsum(-1) - 1).masked_fill_(attention_mask_c_uc_ == 0, 1)
        position_ids_ = position_ids_[:, -uc_input_ids_[mask_].shape[-1]:]
        out = self.model(
            uc_input_ids_[mask_],
            attention_mask=attention_mask_c_uc_,
            use_cache=False,
            past_key_values=kv_cache_,
            position_ids=position_ids_,
        )
        del kv_cache_
        return out.logits

    def __call__(self, input_ids, scores, kv_cache_=None):
        self.kv_cache_ = kv_cache_
        if self.guidance_scale == 1:
            scores = torch.nn.functional.log_softmax(scores, dim=-1)
            return scores
        c_entropy = get_entropy(scores.unsqueeze(1))
        mask_ = c_entropy > self.entropy_threshold
        self.model.cfg_mask__ = mask_ if self.model.cfg_mask__ is None else torch.cat([self.model.cfg_mask__, mask_], dim=-1)
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        logits = self.get_unconditional_logits(input_ids)
        if logits is None:
            return scores
        unconditional_logits = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
        unconditional_logits_ = torch.zeros_like(scores, dtype=unconditional_logits.dtype)
        unconditional_logits_[mask_.squeeze()] = unconditional_logits
        guidance_scale = torch.where(mask_, self.guidance_scale * torch.ones_like(mask_), torch.ones_like(mask_))
        scores_processed = guidance_scale * (scores - unconditional_logits_) + unconditional_logits_
        del unconditional_logits_
        return scores_processed

    
def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:

    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    model_forward = self.__call__
    compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
    if compile_forward:
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    # TODO add ####################################################################
    self.cfg_mask__ = None
    # TODO add ####################################################################

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distribution
        # TODO change ####################################################################
        self.position_ids_ = model_inputs['position_ids']
        kwargs = {'kv_cache_': outputs.past_key_values}
        next_token_scores = logits_processor(input_ids, next_token_logits, **kwargs)
        # TODO change ####################################################################

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    
    
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids



def _get_logits_processor(
    self,
    generation_config: GenerationConfig,
    input_ids_seq_length: Optional[int] = None,
    encoder_input_ids: torch.LongTensor = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    device: Optional[str] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
) -> LogitsProcessorList:
    """
    This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
    instances used to modify the scores of the language model head.
    """
    # instantiate processors list
    processors = LogitsProcessorList()
    if logits_processor is None:
        logits_processor = []

    # TODO change ####################################################################
    if generation_config.guidance_scale is not None and generation_config.guidance_scale != 1:
        if generation_config.entropy_threshold is None:
            processors.append(
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    generation_config.guidance_scale,
                    self,
                    unconditional_ids=negative_prompt_ids,
                    unconditional_attention_mask=negative_prompt_attention_mask,
                    use_cache=generation_config.use_cache,
                )
            )
        else:
            processors.append(
                MTI(
                    generation_config.guidance_scale,
                    self,
                    entropy_threshold=generation_config.entropy_threshold,
                    tokenizer__=generation_config.tokenizer__,
                    lightweight_negative_prompt=generation_config.lightweight_negative_prompt,
                )
            )
    # TODO change ####################################################################
    if generation_config.sequence_bias is not None:
        processors.append(SequenceBiasLogitsProcessor(sequence_bias=generation_config.sequence_bias))

    if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=generation_config.diversity_penalty,
                num_beams=generation_config.num_beams,
                num_beam_groups=generation_config.num_beam_groups,
            )
        )
    if (
        generation_config.encoder_repetition_penalty is not None
        and generation_config.encoder_repetition_penalty != 1.0
    ):
        if len(encoder_input_ids.shape) == 2:
            processors.append(
                EncoderRepetitionPenaltyLogitsProcessor(
                    penalty=generation_config.encoder_repetition_penalty,
                    encoder_input_ids=encoder_input_ids,
                )
            )
        else:
            warnings.warn(
                "Passing `encoder_repetition_penalty` requires some form of `input_ids` to be passed to "
                "`generate`, ignoring the argument.",
                UserWarning,
            )
    if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
    if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
    if (
        generation_config.encoder_no_repeat_ngram_size is not None
        and generation_config.encoder_no_repeat_ngram_size > 0
    ):
        if len(encoder_input_ids.shape) == 2:
            processors.append(
                EncoderNoRepeatNGramLogitsProcessor(
                    generation_config.encoder_no_repeat_ngram_size,
                    encoder_input_ids,
                )
            )
        else:
            warnings.warn(
                "Passing `encoder_no_repeat_ngram_size` requires some form of `input_ids` to be passed to "
                "`generate`, ignoring the argument.",
                UserWarning,
            )
    if generation_config.bad_words_ids is not None:
        processors.append(
            NoBadWordsLogitsProcessor(
                generation_config.bad_words_ids,
                generation_config._eos_token_tensor,
            )
        )
    if (
        generation_config.min_length is not None
        and getattr(generation_config, "_eos_token_tensor", None) is not None
        and generation_config.min_length > 0
    ):
        processors.append(
            MinLengthLogitsProcessor(
                generation_config.min_length,
                generation_config._eos_token_tensor,
                device=device,
            )
        )
    if (
        generation_config.min_new_tokens is not None
        and getattr(generation_config, "_eos_token_tensor", None) is not None
        and generation_config.min_new_tokens > 0
    ):
        processors.append(
            MinNewTokensLengthLogitsProcessor(
                input_ids_seq_length,
                generation_config.min_new_tokens,
                generation_config._eos_token_tensor,
                device=device,
            )
        )
    if prefix_allowed_tokens_fn is not None:
        processors.append(
            PrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn,
                generation_config.num_beams // generation_config.num_beam_groups,
            )
        )
    if generation_config.forced_bos_token_id is not None:
        processors.append(
            ForcedBOSTokenLogitsProcessor(
                generation_config.forced_bos_token_id,
            )
        )
    if generation_config.forced_eos_token_id is not None:
        processors.append(
            ForcedEOSTokenLogitsProcessor(
                generation_config.max_length,
                generation_config.forced_eos_token_id,
                device=device,
            )
        )
    if generation_config.remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    if generation_config.exponential_decay_length_penalty is not None:
        processors.append(
            ExponentialDecayLengthPenalty(
                generation_config.exponential_decay_length_penalty,
                generation_config._eos_token_tensor,
                input_ids_seq_length,
            )
        )
    if generation_config.suppress_tokens is not None:
        processors.append(
            SuppressTokensLogitsProcessor(
                generation_config.suppress_tokens,
                device=device,
            )
        )
    if generation_config.begin_suppress_tokens is not None:
        begin_index = input_ids_seq_length
        begin_index = (
            begin_index
            if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
            else begin_index + 1
        )
        processors.append(
            SuppressTokensAtBeginLogitsProcessor(
                generation_config.begin_suppress_tokens,
                begin_index,
                device=device,
            )
        )

    # TODO (joao): find a strategy to specify the order of the processors
    processors = self._merge_criteria_processor_list(processors, logits_processor)

    # Processors previously known as `LogitsWarpers`, only applied with sampling strategies
    if generation_config.do_sample:
        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(list(generation_config._eos_token_tensor)) + 1)
        if generation_config.num_beams > 1:
            if isinstance(generation_config._eos_token_tensor, list):
                min_tokens_to_keep = len(generation_config._eos_token_tensor) + 1
            elif isinstance(generation_config._eos_token_tensor, torch.Tensor):
                min_tokens_to_keep = generation_config._eos_token_tensor.shape[0] + 1
            else:
                min_tokens_to_keep = 2
        else:
            min_tokens_to_keep = 1

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            processors.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            processors.append(
                TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            processors.append(
                TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.min_p is not None:
            # Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)
            processors.append(
                MinPLogitsWarper(min_p=generation_config.min_p, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
            processors.append(
                TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
            processors.append(
                EpsilonLogitsWarper(
                    epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep
                )
            )
        if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
            processors.append(
                EtaLogitsWarper(
                    epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep, device=device
                )
            )

    # Watermarking should be after all logits processing is finished (see #34630)
    if generation_config.watermarking_config is not None:
        processors.append(
            generation_config.watermarking_config.construct_processor(
                self.config.get_text_config().vocab_size, device
            )
        )

    # `LogitNormalization` should always be the last logit processor, when present
    if generation_config.renormalize_logits is True:
        processors.append(LogitNormalization())
    return processors

GenerationMixin._sample = _sample
GenerationMixin._get_logits_processor = _get_logits_processor

def main():

    model_id = '/home/m2v_intern/yangzhen/checkpoint/Qwen3-8B'
    
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto"
    )

    gen_config = GenerationConfig(
        entropy_threshold=-1,
        lightweight_negative_prompt="OUTPUT ERROR",
        guidance_scale=100, 
        tokenizer__=tokenizer,
    )

    # prepare the model input
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=100,
        generation_config=gen_config,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)


if __name__ == "__main__":
    main()

