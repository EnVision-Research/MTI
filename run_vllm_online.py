import torch, copy
from transformers import AutoTokenizer
from vllm.v1.worker.gpu_model_runner import *
from vllm.v1.attention.backends.flash_attn import *
from vllm.entrypoints.openai.api_server import *

def get_entropy_from_logits(logits):
    p = torch.softmax(logits, dim=-1)
    return -(torch.where(p > 0, p * p.log(), torch.zeros_like(p))).sum(dim=-1)

def get_position(x: torch.Tensor, length: int):
    z = (x == 0)
    if not torch.any(z):
        res = []
        for i in range(1, length + 1):
            res.append(x + i)
        return torch.stack(res, dim=-1).flatten()
    g = z.cumsum(0)
    if z[0]: g -= 1
    m = torch.full((int(g[-1]) + 1,), x.min() - 1, device=x.device, dtype=x.dtype)
    m = m.scatter_reduce(0, g, x, reduce='amax', include_self=True)
    res = []
    for i in range(1, length + 1):
        res.append(m + i)
    return torch.stack(res, dim=-1).flatten()

def forward(
    self,
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata: FlashAttentionMetadata,
    output: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    output_block_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward pass with FlashAttention.

    Args:
        query: shape = [num_tokens, num_heads, head_size]
        key: shape = [num_tokens, num_kv_heads, head_size]
        value: shape = [num_tokens, num_kv_heads, head_size]
        kv_cache: shape =
            [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: Metadata for attention.
    Returns:
        shape = [num_tokens, num_heads * head_size]
    NOTE: FP8 quantization, flash-attn expect the size of
            {q,k,v}_descale to be (num_sequences, num_kv_heads).
            We use torch's .expand() to avoid duplicating values
    """
    assert output is not None, "Output tensor must be provided."

    if output_scale is not None or output_block_scale is not None:
        raise NotImplementedError(
            "fused output quantization is not yet supported"
            " for FlashAttentionImpl")

    if attn_metadata is None:
        # Profiling run.
        return output

    attn_type = self.attn_type

    # IMPORTANT!
    # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
    # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
    # in this method. For example, `view` and `slice` (or `[:n]`) operations
    # are surprisingly slow even in the case they do not invoke any GPU ops.
    # Minimize the PyTorch ops in this method as much as possible.
    # Whenever making a change in this method, please benchmark the
    # performance to make sure it does not introduce any overhead.

    num_actual_tokens = attn_metadata.num_actual_tokens

    # Handle encoder attention differently - no KV cache needed
    if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
        # For encoder attention,
        # we use direct Q, K, V tensors without caching
        return self._forward_encoder_attention(query[:num_actual_tokens],
                                                key[:num_actual_tokens],
                                                value[:num_actual_tokens],
                                                output[:num_actual_tokens],
                                                attn_metadata, layer)

    # For decoder and cross-attention, use KV cache as before
    key_cache, value_cache = kv_cache.unbind(0)

    # key and value may be None in the case of cross attention. They are
    # calculated once based on the output from the encoder and then cached
    # in KV cache.
    # TODO add ####################################################################
    if getattr(attn_metadata, "dont_save_kv_cache", False):
        pass
    else:
        if (self.kv_sharing_target_layer_name is None and key is not None
                and value is not None):
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
    # TODO add ####################################################################

    if self.kv_cache_dtype.startswith("fp8"):
        dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
            self.kv_cache_dtype)
        key_cache = key_cache.view(dtype)
        value_cache = value_cache.view(dtype)
        num_tokens, num_heads, head_size = query.shape
        query, _ = ops.scaled_fp8_quant(
            query.reshape(
                (num_tokens, num_heads * head_size)).contiguous(),
            layer._q_scale)
        query = query.reshape((num_tokens, num_heads, head_size))

    if not attn_metadata.use_cascade:
        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table
        scheduler_metadata = attn_metadata.scheduler_metadata

        descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            scheduler_metadata=scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=attn_metadata.max_num_splits,
            s_aux=self.sinks,
        )
        return output

    # Cascade attention (rare case).
    cascade_attention(
        output[:num_actual_tokens],
        query[:num_actual_tokens],
        key_cache,
        value_cache,
        cu_query_lens=attn_metadata.query_start_loc,
        max_query_len=attn_metadata.max_query_len,
        cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
        prefix_kv_lens=attn_metadata.prefix_kv_lens,
        suffix_kv_lens=attn_metadata.suffix_kv_lens,
        max_kv_len=attn_metadata.max_seq_len,
        softmax_scale=self.scale,
        alibi_slopes=self.alibi_slopes,
        sliding_window=self.sliding_window,
        logits_soft_cap=self.logits_soft_cap,
        block_table=attn_metadata.block_table,
        common_prefix_len=attn_metadata.common_prefix_len,
        fa_version=self.vllm_flash_attn_version,
        prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
        suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
        q_descale=layer._q_scale,
        k_descale=layer._k_scale,
        v_descale=layer._v_scale,
    )
    return output

@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:

    with record_function_or_nullcontext("Preprocess"):
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output,
                                                self.vllm_config)
        if self.cache_config.kv_sharing_fast_prefill:
            assert not self.input_batch.num_prompt_logprobs, (
                "--kv-sharing-fast-prefill produces incorrect logprobs for "
                "prompt tokens, tokens, please disable it when the requests"
                " need prompt logprobs")

        if self.prepare_inputs_event is not None:
            # Ensure prior step has finished with reused CPU tensors.
            self.prepare_inputs_event.synchronize()
        try:
            # Prepare the decoder inputs.
            (attn_metadata, logits_indices, spec_decode_metadata,
                num_scheduled_tokens_np, spec_decode_common_attn_metadata,
                max_query_len) = self._prepare_inputs(scheduler_output)

        finally:
            if self.prepare_inputs_event is not None:
                self.prepare_inputs_event.record()

        (
            num_scheduled_tokens,
            num_input_tokens,
            num_tokens_across_dp,
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
        ) = self._preprocess(scheduler_output, intermediate_tensors)

        uniform_decode = (max_query_len
                            == self.uniform_decode_query_len) and (
                                num_scheduled_tokens
                                == self.input_batch.num_reqs * max_query_len)
        batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                            uniform_decode=uniform_decode)
        cudagraph_runtime_mode, batch_descriptor = \
            self.cudagraph_dispatcher.dispatch(batch_descriptor)
    # Run the model.
    # Use persistent buffers for CUDA graphs.
    with (set_forward_context(
            attn_metadata, 
            self.vllm_config, 
            num_tokens=num_input_tokens, 
            num_tokens_across_dp=num_tokens_across_dp, 
            cudagraph_runtime_mode=cudagraph_runtime_mode, 
            batch_descriptor=batch_descriptor, 
        ), 
        record_function_or_nullcontext("Forward"), 
        self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output):
        model_output = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    # print('hello')
    with record_function_or_nullcontext("Postprocess"):
        if self.use_aux_hidden_state_outputs:
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = None

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        broadcast_pp_output = \
            self.parallel_config.distributed_executor_backend \
            == "external_launcher" and len(get_pp_group().ranks) > 0
        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            assert isinstance(hidden_states, IntermediateTensors)
            if not broadcast_pp_output:
                hidden_states.kv_connector_output = kv_connector_output
                return hidden_states
            get_pp_group().send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group())
            logits = None
        else:
            if self.is_pooling_model:
                return self._pool(hidden_states, num_scheduled_tokens,
                                    num_scheduled_tokens_np,
                                    kv_connector_output)

            sample_hidden_states = hidden_states[logits_indices]
            logits = self.model.compute_logits(sample_hidden_states, None)

            # TODO add ####################################################################
            entropy_threshold_ = torch.tensor([
                (req.sampling_params.extra_args or {}).get("entropy_threshold") or -1
                for req in self.requests.values()
            ]).to(logits.device)
            c_entropy_ = get_entropy_from_logits(logits)
            mask_ = c_entropy_ > entropy_threshold_
            for i, req in enumerate(self.requests.values()): 
                mask_[i] = (req.sampling_params.extra_args.get("guidance_scale", 1) != 1) if mask_[i] else mask_[i]
            logits = torch.nn.functional.log_softmax(logits, dim=-1)

            if mask_.sum() != 0:
                true_idx_ = torch.nonzero(mask_, as_tuple=False).view(-1).tolist()
                kv_reuse_np_ = [
                    (req.sampling_params.extra_args or {}).get("kv_reuse_np") or "OUTPUT ERROR"
                    for i, req in enumerate(self.requests.values()) if i in true_idx_
                ]
                guidance_scale_ = torch.tensor([
                    (req.sampling_params.extra_args or {}).get("guidance_scale") or 1
                    for i, req in enumerate(self.requests.values()) if i in true_idx_
                ]).unsqueeze(-1).to(logits.device).to(logits.dtype)

                if not hasattr(self, "tokenizer_"):
                    self.tokenizer_ = AutoTokenizer.from_pretrained(self.vllm_config.model_config.tokenizer)
                kv_reuse_np_list = self.tokenizer_.batch_encode_plus(kv_reuse_np_, add_special_tokens=False)['input_ids']
                kv_reuse_np_ = torch.tensor([x for pair in kv_reuse_np_list for x in pair]).to(logits.device)
                length_np_ = len(kv_reuse_np_list[0])

                attn_metadata_uc_ = copy.deepcopy(attn_metadata)
                new_temporal_seq_lens = torch.tensor([len(x) for x in kv_reuse_np_list], dtype=torch.int32).to(logits.device)
                for key_ in attn_metadata_uc_.keys():
                    attn_metadata_uc_[key_].num_actual_tokens = kv_reuse_np_.shape[0]
                    attn_metadata_uc_[key_].max_query_len = max((len(x) for x in kv_reuse_np_list), default=0)
                    attn_metadata_uc_[key_].block_table = attn_metadata[key_].block_table[mask_]
                    attn_metadata_uc_[key_].query_start_loc = torch.tensor([i * length_np_ for i in range(len(kv_reuse_np_list) + 1)]).to(logits.device).to(attn_metadata[key_].query_start_loc.dtype)
                    attn_metadata_uc_[key_].seq_lens = attn_metadata[key_].seq_lens[mask_] + new_temporal_seq_lens 
                    attn_metadata_uc_[key_].dont_save_kv_cache = True
                    attn_metadata_uc_[key_].use_cascade = False

                new_positions = get_position(positions, length_np_)[:length_np_ * len(kv_reuse_np_list)].to(logits.device)
                logits_indices_uc_ = torch.tensor([i * length_np_ - 1 for i in range(1, len(kv_reuse_np_list) + 1)])

                new_num_tokens_ = min((s for s in self.vllm_config.compilation_config.cudagraph_capture_sizes if s >= kv_reuse_np_.shape[0]), default=kv_reuse_np_.shape[0])

                batch_descriptor_uc_ = BatchDescriptor(num_tokens=new_num_tokens_, uniform_decode=None)
                cudagraph_runtime_mode_uc_, batch_descriptor_uc_ = self.cudagraph_dispatcher.dispatch(batch_descriptor_uc_)

                if kv_reuse_np_.shape[0] != new_num_tokens_:
                    kv_reuse_np_ = torch.cat([kv_reuse_np_, torch.zeros(new_num_tokens_ - kv_reuse_np_.shape[0], device=kv_reuse_np_.device)]) if kv_reuse_np_.shape[0] < new_num_tokens_ else kv_reuse_np_
                    new_positions = torch.cat([new_positions, torch.zeros(new_num_tokens_ - new_positions.shape[0], device=new_positions.device)]) if new_positions.shape[0] < new_num_tokens_ else new_positions

                with (set_forward_context(
                        attn_metadata_uc_, 
                        self.vllm_config, 
                        num_tokens=new_num_tokens_, 
                        num_tokens_across_dp=num_tokens_across_dp, 
                        cudagraph_runtime_mode=cudagraph_runtime_mode_uc_, 
                        batch_descriptor=batch_descriptor_uc_, 
                    ), 
                    record_function_or_nullcontext("Forward"), 
                    self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output):
                    model_output_uc = self.model(
                        input_ids=kv_reuse_np_,
                        positions=new_positions,
                        **model_kwargs,
                    )
                logits_uc = self.model.compute_logits(model_output_uc[logits_indices_uc_], None)
                uc_ =  torch.nn.functional.log_softmax(logits_uc, dim=-1)
                logits[mask_] = guidance_scale_ * (logits[mask_] - uc_) + uc_

            # TODO add ####################################################################
        if broadcast_pp_output:
            model_output_broadcast_data = {
                "logits": logits.contiguous(),
            } if logits is not None else {}
            model_output_broadcast_data = get_pp_group(
            ).broadcast_tensor_dict(model_output_broadcast_data,
                                    src=len(get_pp_group().ranks) - 1)
            assert model_output_broadcast_data is not None
            logits = model_output_broadcast_data["logits"]

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            self.apply_grammar_bitmask(scheduler_output, logits)

    with record_function_or_nullcontext("Sample"):
        sampler_output = self._sample(logits, spec_decode_metadata)

    with record_function_or_nullcontext("Bookkeep"):
        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(scheduler_output, sampler_output,
                                    logits, hidden_states,
                                    num_scheduled_tokens)

    if self.speculative_config:
        assert spec_decode_common_attn_metadata is not None
        with record_function_or_nullcontext("Draft"):
            self._draft_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                valid_sampled_token_ids,
                self.input_batch.sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,
            )

    with record_function_or_nullcontext("EPLB"):
        self.eplb_step()

    output = ModelRunnerOutput(
        req_ids=req_ids_output_copy,
        req_id_to_index=req_id_to_index_output_copy,
        sampled_token_ids=valid_sampled_token_ids,
        logprobs=logprobs_lists,
        prompt_logprobs_dict=prompt_logprobs_dict,
        pooler_output=[],
        kv_connector_output=kv_connector_output,
        num_nans_in_logits=num_nans_in_logits,
    )

    if not self.use_async_scheduling:
        return output

    return AsyncGPUModelRunnerOutput(
        model_runner_output=output,
        sampled_token_ids=sampler_output.sampled_token_ids,
        invalid_req_indices=invalid_req_indices,
        async_output_copy_stream=self.async_output_copy_stream,
    )

GPUModelRunner.execute_model = execute_model
FlashAttentionImpl.forward = forward

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()

