import math
import torch
import copy
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers import OPTConfig
from .long_mem_cross_attn_vanilla import CrossAttentionMemory
from accelerate.logging import get_logger
from torch.profiler import profile, record_function, ProfilerActivity
import random
import evaluate
from huggingface_hub import PyTorchModelHubMixin

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

IGNORE_INDEX = -100

class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, num_prepend):
        super().__init__()
        self.model = base_model
        self.n_prepend = num_prepend
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        if num_mem_tokens > 0:
            if isinstance(self.model.config, OPTConfig):
                mem_emb_dim = getattr(self.model.config, 'n_embd', self.model.config.word_embed_proj_dim)
            else:
                mem_emb_dim = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
            # sum_embeds serve as prompt tokens for summarizing current segment's topic
            sum_emb_weights = torch.randn((num_mem_tokens, mem_emb_dim)) * embeddings.weight.data.std()
            self.register_parameter('sum_embeds', torch.nn.Parameter(sum_emb_weights, requires_grad=True))

    def set_sum_embeds(self, input_shape):
        sum_embeds = self.sum_embeds.repeat(input_shape[0], 1, 1)
        return sum_embeds

    def forward(self, input_ids, memory_state=None, prepend_state=None, **kwargs):
        input_ids = input_ids.cuda()
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                kwargs[k] = v.cuda()
        if memory_state is None and self.num_mem_tokens > 0:
            memory_state = self.set_sum_embeds(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, prepend_state=prepend_state, **kwargs)

        out = self.model(**seg_kwargs)
        n_prepend = self.n_prepend
        out, new_memory_state = self.process_output(out, 0 if prepend_state is None else n_prepend, **kwargs)
        input_ids = input_ids.cpu()
        for k, v in kwargs.items():
                if torch.is_tensor(v):
                    kwargs[k] = v.cpu()
        return out, new_memory_state
    
    def generate(self, input_ids, memory_state, prepend_state, attention_mask, **generate_kwargs):


        if memory_state is None and self.num_mem_tokens > 0:
            memory_state = self.set_sum_embeds(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, prepend_state=prepend_state, generate=True, attention_mask=attention_mask)        
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, prepend_state=None, generate=False, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if prepend_state is not None:
            prepend_embeds = self.model.get_input_embeddings()(prepend_state)
            inputs_embeds = torch.cat([prepend_embeds, inputs_embeds], dim=1)
        if memory_state is not None:
            if generate:
                inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)
            else:
                inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape, 0 if prepend_state is None else self.n_prepend, generate)
        seg_kwargs['output_hidden_states'] = True
        return seg_kwargs
    
    def pad_attention_mask(self, attention_mask, shape, n_prepend, generate=False):
        if self.num_mem_tokens in {0, None}:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, (n_prepend):] = attention_mask
            return mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            if generate:
                mask[:, (self.num_mem_tokens+n_prepend):] = attention_mask
            else:
                mask[:, (self.num_mem_tokens+n_prepend):-self.num_mem_tokens] = attention_mask
            return mask
    
    def process_output(self, model_outputs, n_prepend, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens:]
            out['logits'] = model_outputs.logits[:, (self.num_mem_tokens+n_prepend):-self.num_mem_tokens]
            out['logits'] = out['logits'].cpu()

            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, (self.num_mem_tokens+n_prepend):-self.num_mem_tokens] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = None
            out['logits'] = model_outputs.logits[:, (n_prepend):]
            
            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, (n_prepend):] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
            
        return out, memory_state 

class SegmentIterator:
    def __init__(self, **kwargs):
        self.iter_content = kwargs
        self.pointer = 0
        self.empty = False
    
    def next(self, segment_length):
        segment = {}
        for k, tensor in self.iter_content.items():
            if tensor is not None:
                if self.pointer >= tensor.shape[1]:
                    self.empty = True
                    return None
                segment[k] = tensor[:, self.pointer:self.pointer+segment_length]
        
        self.pointer += segment_length
        return segment
    
    def is_empty(self):
        for k, tensor in self.iter_content.items():
            if tensor is not None:
                if self.pointer >= tensor.shape[1]:
                    self.empty = True
                    return True
                else:
                    return False

class RecurrentWrapper(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, memory_cell, mem_emb_dim=None, hidden_dim=4096, ltm_context=64, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs
        self.ltm_context = ltm_context
        self.logger = get_logger('')
        if mem_emb_dim is not None:
            self.cross_attn = CrossAttentionMemory(mem_emb_dim, hidden_dim)
        else:
            self.cross_attn = None
        
        self.rouge = evaluate.load('rouge')
        self.f1 = evaluate.load("f1")

    def forward(self, 
            input_ids, 
            labels=None, 
            labels_mask=None, 
            inputs_embeds=None, 
            attention_mask=None, 
            mask_size=None,  # Size of the attention mask used to compute the loss, it should be the length of the labels. If it's None, then self.mask_size is used. 
            output_attentions=None, 
            output_hidden_states=None, 
            sum_fraction=0.5,
            segment_size=1022, 
            mode='train', 
            prof=False,
            pos_mask=None,
            **kwargs
        ):

        mask_size = self.rmt_config.get('mask_size') if mask_size is None else mask_size

        memory_state = None
        prepend_state = None
        segment = None
        seg_iter = SegmentIterator(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        seg_num = 0

        cell_outputs = []
        n_cell_out = self.rmt_config.get('n_cell_out')
        memory_seq = None

        total_hist = []

        while True:

            prepend_state = segment['input_ids'][:,-self.memory_cell.n_prepend:].cuda() if segment is not None and self.memory_cell.n_prepend > 0 else None
            segment = seg_iter.next(segment_size)
            if segment is None:
                break

            if self.cross_attn is not None:
                seg = copy.deepcopy(segment)
                seg['input_ids'] = seg['input_ids'][:,:int(round(segment_size * sum_fraction))]
                seg['attention_mask'] = seg['attention_mask'][:,:int(round(segment_size * sum_fraction))]
                _, q_mem = self.memory_cell(**seg, memory_state=None)
                memory_state, hist = self.cross_attn(
                    memory_seq,
                    q_mem,
                    mode,
                    seg_num if seg_num < self.ltm_context else self.ltm_context,
                )
                if hist is not None:
                    total_hist.extend(hist)

            if prof:
                with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=True) as prof_m:
                    with record_function("model_inference"):
                        cell_out, memory_state = self.memory_cell(
                            **segment,
                            memory_state=memory_state,
                            prepend_state=prepend_state,
                            output_hidden_states=True,
                        )
                
                with open('model_profile_dump.txt', 'w') as file:
                    file.write(prof_m.key_averages().table(sort_by="cuda_time_total"))
                
                prof_m.export_chrome_trace("model_trace.json")
                exit(0)
            else:
                cell_out, memory_state = self.memory_cell(
                    **segment,
                    memory_state=memory_state,
                    prepend_state=prepend_state,
                    output_hidden_states=True,
                )

            # if prof:
            #     torch.cuda.synchronize()
            #     self.logger.info('segment ' + str(seg_num) + ' elapsed time: ' + str(start.elapsed_time(end)) + ' ms')

            cell_outputs.append(cell_out)
            if len(cell_outputs) > n_cell_out:
                cell_outputs.pop(0)
            
            if self.cross_attn is not None:
                if memory_seq is None:
                    memory_seq = memory_state.cpu()
                else:
                    memory_seq = torch.cat([memory_seq, memory_state.cpu()], dim=1)
                    if memory_seq.shape[1] > self.ltm_context:
                        memory_seq = memory_seq[:,-self.ltm_context:,:]

            if memory_state is not None:
                self.manage_gradients(memory_state, seg_num)

            seg_num+=1
        
        out, metrics = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states,
                                   mask_size=mask_size)
        return out, total_hist, metrics
    
    def generate(self, input_ids, attention_mask, segment_size, mem_seq=None, **generate_kwargs):
        seg_iter = SegmentIterator(input_ids=input_ids, attention_mask=attention_mask)
        memory_state = None
        prepend_state = None
        seg_num = 0

        while True:
            prepend_state = segment['input_ids'][:, -self.memory_cell.n_prepend:].cuda() if segment is not None and self.memory_cell.n_prepend > 0 else None
            segment = seg_iter.next(segment_size)
            if segment is None:
                break

            if self.cross_attn is not None and mem_seq is not None:
                seg = copy.deepcopy(segment)
                seg_len = max(1, int(round(segment_size * 0.5)))
                seg['input_ids'] = seg['input_ids'][:, :seg_len]
                seg['attention_mask'] = seg['attention_mask'][:, :seg_len]
                _, q_mem = self.memory_cell(**seg, memory_state=None)
                memory_state, _ = self.cross_attn(
                    mem_seq,
                    q_mem,
                    "inference",
                    seg_num if seg_num < self.ltm_context else self.ltm_context,
                )

            if seg_iter.is_empty():
                for k, v in segment.items():
                    if torch.is_tensor(v):
                        segment[k] = v.cuda()
                return self.memory_cell.generate(
                    **segment,
                    memory_state=memory_state,
                    prepend_state=prepend_state,
                    **generate_kwargs,
                )

            _, memory_state = self.memory_cell(
                **segment,
                memory_state=memory_state,
                prepend_state=prepend_state,
                output_hidden_states=False,
            )

            if memory_state is not None:
                new_mem = memory_state.cpu()
                if mem_seq is None:
                    mem_seq = new_mem
                else:
                    mem_seq = torch.cat([mem_seq, new_mem], dim=1)
                    if mem_seq.shape[1] > self.ltm_context:
                        mem_seq = mem_seq[:, -self.ltm_context:, :]

            seg_num += 1

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])
        
        mask_size = kwargs.get('mask_size')
        metrics = {
            'loss': None,
            'ppl': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'accuracy': None
        }

        labels = kwargs.get('labels')
        if labels.shape[1] <= mask_size:
            mask_size = labels.shape[1]-1
        
        if labels is not None:
            shift_labels = labels[..., -mask_size:].contiguous()
            shift_logits = full_logits[..., -(mask_size+1):-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            gen_loss = loss_fct(flat_logits.cuda(), flat_labels.cuda())
            out['loss'] = gen_loss
            metrics['loss'] = out['loss'].detach().item()
            metrics['ppl'] = torch.exp(gen_loss.detach()).item()

            # filter ignore_index before computing metrics
            flat_labels_cpu = flat_labels.detach().cpu()
            mask = flat_labels_cpu != IGNORE_INDEX
            labels_valid = flat_labels_cpu[mask]

            predictions = flat_logits.argmax(dim=-1).detach().cpu()
            preds_valid = predictions[mask]

            # add zero_division=0 to avoild warning
            if labels_valid.numel() > 0:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels_valid,
                    preds_valid,
                    average='weighted',
                    zero_division=0
                )
                accuracy = accuracy_score(labels_valid, preds_valid)
                metrics['precision'] = float(precision)
                metrics['recall'] = float(recall)
                metrics['f1'] = float(f1)
                metrics['accuracy'] = float(accuracy)

        else:
            zero = torch.tensor(0.0, device=full_logits.device)
            out['loss'] = zero
            metrics['loss'] = 0.0
            metrics['ppl'] = None
            metrics['f1'] = None
            metrics['accuracy'] = None

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

        return out, metrics
        
    def manage_gradients(self, memory_state, seg_num):
        if seg_num == 0:
            return True
        memory_state = memory_state.detach()
        return False
