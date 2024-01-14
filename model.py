import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch.optim import AdamW
import torch.distributed as dist
from transformers import PreTrainedModel, RobertaModel, AutoModelForCausalLM, EncoderDecoderConfig, EncoderDecoderModel, AutoConfig, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, BartPretrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutput, 
    Seq2SeqModelOutput, 
    Seq2SeqLMOutput, 
    SequenceClassifierOutput, 
    BaseModelOutputWithCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.optimization import get_linear_schedule_with_warmup
from torchmetrics.text.rouge import ROUGEScore
import senteval


# modified from https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py
class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooler_type = config.pooler_type

    def forward(self, last_hidden_states, attention_mask):
        if self.pooler_type == "cls":
            return last_hidden_states[:, 0:1]
        elif self.pooler_type == "avg":
            return ((last_hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)).unsqueeze(1)

class Add(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size*2), 
            nn.ReLU(),
            nn.Linear(config.hidden_size*2, config.hidden_size),
        )

    def forward(self, x, y):
        z = torch.cat((x, y), dim=-1)
        z = self.mlp(z)
        return z


class Diff(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size*2, config.hidden_size*2), 
            nn.ReLU(),
            nn.Linear(config.hidden_size*2, config.hidden_size),
        )

    def forward(self, x, y):
        z = torch.cat((x, y), dim=-1)
        z = self.mlp(z)  
        return z

class ExtComp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.bn_size), 
            nn.ReLU(),
            nn.Linear(config.bn_size, config.hidden_size),
        )
        
    def forward(self, x):
        x = self.mlp(x)
        return x

class AbsComp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.bn_size), 
            nn.ReLU(),
            nn.Linear(config.bn_size, config.hidden_size),
        )
        
    def forward(self, x):
        x = self.mlp(x)
        return x

class Proj(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size*2), 
            nn.ReLU(),
            nn.Linear(config.hidden_size*2, config.hidden_size),
        )
        
    def forward(self, x):
        x = self.mlp(x)
        return x

class Sim(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.dense(x)  
        x = self.activation(x)
        return x

@dataclass
class SentEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    sent_emb_sim: Optional[torch.FloatTensor] = None
    sent_emb_pos: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# modified from https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/roberta/modeling_roberta.py#L1165
class RobertaSentEncoder(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)

        # sentence embedding pooler
        self.pooler = Pooler(config)

        # operation modules
        self.add = Add(config)
        self.diff = Diff(config)
        self.extcomp = ExtComp(config)
        self.abscomp = AbsComp(config)

        # projection after LM
        self.proj = Proj(config)

        self.sim = Sim(config)

        self.LayerNorm = nn.LayerNorm(config.hidden_size)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        task = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if task != "sent_emb": 
            # reshape to (sent_num * batch_size, seq_len)
            input_seq_len = input_ids.size(2)
            input_ids = input_ids.view(-1, input_seq_len)
            attention_mask = attention_mask.view(-1, input_seq_len)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sent_embs = self.pooler(outputs.last_hidden_state, attention_mask)

        if not self.config.skip_mlp:
            sent_embs = self.proj(sent_embs)
            sent_embs = self.LayerNorm(sent_embs)

        if task == "sent_emb":
            return SentEncoderOutput(
                last_hidden_state=sent_embs, sent_emb_sim=None, sent_emb_pos=None, hidden_states=None, attentions=None
            )
        
        sent_emb_a, sent_emb_b, sent_emb_target = torch.chunk(sent_embs, 3, dim=0)

        if task == "para":
            sent_emb = sent_emb_a
        elif task == "add":
            sent_emb = self.add(sent_emb_a, sent_emb_b)
        elif task == "diff":
            sent_emb = self.diff(sent_emb_a, sent_emb_b)
        elif task == "extcomp":
            sent_emb = self.extcomp(sent_emb_a)
        elif task == "abscomp":
            sent_emb = self.abscomp(sent_emb_a)

        else:
            raise NotImplementedError("%s not supported", task)

        if not self.config.skip_mlp and task in ["add", "diff", "extcomp", "abscomp"]:
            sent_emb = self.LayerNorm(sent_emb)

        sent_emb_sim = sent_emb
        sent_emb_pos = sent_emb_target

        return SentEncoderOutput(
            last_hidden_state=sent_emb, sent_emb_sim=sent_emb_sim, sent_emb_pos=sent_emb_pos, hidden_states=None, attentions=None
        )


# modified from https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L165
class MyEncoderDecoderModel(PreTrainedModel):
    r"""
    [`EncoderDecoderModel`] is a generic model class that will be instantiated as a transformer architecture with one
    of the base model classes of the library as encoder and another one as decoder when created with the
    :meth*~transformers.AutoModel.from_pretrained* class method for the encoder and
    :meth*~transformers.AutoModelForCausalLM.from_pretrained* class method for the decoder.
    """
    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.encoder.hidden_size:
                raise ValueError(
                    "If `cross_attention_hidden_size` is specified in the decoder's configuration, "
                    "it has to be equal to the encoder's `hidden_size`. "
                    f"Got {config.decoder.cross_attention_hidden_size} for `config.decoder.cross_attention_hidden_size` "
                    f"and {config.encoder.hidden_size} for `config.encoder.hidden_size`."
                )

        # initialize with config
        super().__init__(config)

        if encoder is None:
            from ..auto.modeling_auto import AutoModel

            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            from ..auto.modeling_auto import AutoModelForCausalLM

            decoder = AutoModelForCausalLM.from_config(config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config: {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config: {self.config.decoder}"
            )

        # make sure that the individual model's config refers to the shared config
        # so that the updates to the config will be synced
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # encoder outputs might need to be projected to different dimension for decoder
        if (
            self.encoder.config.hidden_size != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            self.enc_to_dec_proj = nn.Linear(self.encoder.config.hidden_size, self.decoder.config.hidden_size)

        if self.encoder.get_output_embeddings() is not None:
            raise ValueError(
                f"The encoder {self.encoder} should not have a LM Head. Please use a model without LM Head"
            )

        # tie encoder, decoder weights if config set accordingly
        self.tie_weights()


    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for EncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                task=task
            )

        sent_emb = encoder_outputs.last_hidden_state
        sent_emb_sim = encoder_outputs.sent_emb_sim
        sent_emb_pos = encoder_outputs.sent_emb_pos

        if task == "sent_emb":
            return sent_emb

        if labels is not None:
            label_seq_len = labels.size(2)
            labels = labels.view(-1, label_seq_len)
            labels = torch.chunk(labels, 3, dim=0)[0]

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=sent_emb,
            encoder_attention_mask=None,    # no need to apply attention mask on bottleneck
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        gen_loss = None
        cl_loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = nn.CrossEntropyLoss()
            gen_loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

            if self.config.cl_loss_weight == 0:
                cl_loss = 0.0
            else:
                # Gather all embeddings if using distributed training
                if dist.is_initialized() and self.training:
                    # Dummy vectors for allgather
                    sent_emb_sim_list = [torch.zeros_like(sent_emb_sim) for _ in range(dist.get_world_size())]
                    sent_emb_pos_list = [torch.zeros_like(sent_emb_pos) for _ in range(dist.get_world_size())]
                    # Allgather
                    dist.all_gather(tensor_list=sent_emb_sim_list, tensor=sent_emb_sim.contiguous())
                    dist.all_gather(tensor_list=sent_emb_pos_list, tensor=sent_emb_pos.contiguous())

                    # Since allgather results do not have gradients, we replace the
                    # current process's corresponding embeddings with original tensors
                    sent_emb_sim_list[dist.get_rank()] = sent_emb_sim
                    sent_emb_pos_list[dist.get_rank()] = sent_emb_pos
                    # Get full batch embeddings: (bs x N, hidden)
                    sent_emb_sim = torch.cat(sent_emb_sim_list, 0)
                    sent_emb_pos = torch.cat(sent_emb_pos_list, 0)


                # contrastive loss with in-batch negatives
                sim_fct = nn.CosineSimilarity(dim=-1)
                cos_sim = sim_fct(sent_emb_sim, sent_emb_pos.squeeze(1).unsqueeze(0)) / self.config.temp
                cl_labels = torch.arange(cos_sim.size(0)).long().to(self.device)
                loss_fct = nn.CrossEntropyLoss()
                cl_loss = loss_fct(cos_sim, cl_labels)

            loss = self.config.cl_loss_weight * cl_loss + self.config.gen_loss_weight * gen_loss 

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return InterSentOutput(
            loss=loss,
            gen_loss=gen_loss,
            cl_loss=cl_loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.decoder.config.pad_token_id, self.decoder.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, task=None, **kwargs
    ):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
            "task": task,
        }
        return input_dict

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)

class InterSentOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    gen_loss: Optional[torch.FloatTensor] = None
    cl_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class InterSent(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args 

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.encoder, cache_dir=args.cache_dir)

        # load encoder
        config = AutoConfig.from_pretrained(args.encoder, cache_dir=args.cache_dir) 
        config.pooler_type = args.pooler_type  
        config.bn_size = args.bn_size
        config.skip_mlp = args.skip_mlp
        #if "roberta" in args.encoder:
        encoder = RobertaSentEncoder.from_pretrained(args.encoder, config=config, cache_dir=args.cache_dir)
        #elif "bart" in args.encoder:
        #    encoder = BartSentEncoder.from_pretrained(args.encoder, config=config, cache_dir=args.cache_dir)

        # load decoder
        decoder = AutoModelForCausalLM.from_pretrained(args.decoder, cache_dir=args.cache_dir, is_decoder=True, add_cross_attention=True)

        # build encoder-decoder model
        self.model = MyEncoderDecoderModel(encoder=encoder, decoder=decoder)
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.cl_loss_weight = args.cl_loss_weight
        self.model.config.gen_loss_weight = args.gen_loss_weight
        self.model.config.temp = args.temp

        # freeze roberta encoder if needed
        if args.freeze_encoder:
            for n, p in self.model.named_parameters():
                if "roberta" in n:
                    p.requires_grad = False

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        gen_loss, cl_loss, loss = outputs.gen_loss, outputs.cl_loss, outputs.loss
        self.log("train_gen_loss", gen_loss, sync_dist=True)
        self.log("train_cl_loss", cl_loss, sync_dist=True)
        self.log("train_loss", loss, sync_dist=True) 
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        gen_loss, cl_loss, loss = outputs.gen_loss, outputs.cl_loss, outputs.loss

        outputs = self.model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], task=batch["task"], max_length=128, do_sample=True)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        input_seq_len = batch["input_ids"].size(2)
        input_ids_a, input_ids_b, targets = torch.chunk(batch["input_ids"].view(-1, input_seq_len), 3, dim=0)
        decoded_targets = self.tokenizer.batch_decode(targets, skip_special_tokens=True)

        return gen_loss, cl_loss, loss, decoded_targets, decoded_outputs
    
    def validation_epoch_end(self, outputs):
        gen_loss, cl_loss, loss, decoded_targets, decoded_outputs = [], [], [], [], []

        for batch in outputs:
            gen_loss.append(batch[0])
            cl_loss.append(batch[1])
            loss.append(batch[2])
            decoded_targets += batch[3]
            decoded_outputs += batch[4]

        self.log("val_gen_loss", sum(gen_loss)/len(gen_loss), sync_dist=True)
        self.log("val_cl_loss", sum(cl_loss)/len(cl_loss), sync_dist=True)
        self.log("val_loss", sum(loss)/len(loss), sync_dist=True)

        rouge = ROUGEScore()
        rouge_score = rouge(decoded_outputs, decoded_targets)
        self.log("val_rouge1", rouge_score["rouge1_fmeasure"], sync_dist=True)
        self.log("val_rouge2", rouge_score["rouge2_fmeasure"], sync_dist=True)
        self.log("val_rougeL", rouge_score["rougeL_fmeasure"], sync_dist=True)
                   
        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            for k in batch:
                batch[k] = batch[k].to(self.device)
            with torch.no_grad():
                sent_emb = self.model(**batch, task="sent_emb").cpu().squeeze()

            return sent_emb

        # Set params for SentEval (fastmode)
        params = {'task_path': os.path.join('senteval/data'), 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ['STSBenchmark']
        self.model.eval()
        results = se.eval(tasks)
        
        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]

        self.log("val_stsb", stsb_spearman)

    def test_step(self, batch, batch_idx):
        outputs = self.model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], task=batch["task"], max_length=128, num_beams=5)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        input_seq_len = batch["input_ids"].size(2)
        input_ids_a, input_ids_b, targets = torch.chunk(batch["input_ids"].view(-1, input_seq_len), 3, dim=0)
        decoded_inputs_a = self.tokenizer.batch_decode(input_ids_a, skip_special_tokens=True)
        decoded_inputs_b = self.tokenizer.batch_decode(input_ids_b, skip_special_tokens=True)
        decoded_targets = self.tokenizer.batch_decode(targets, skip_special_tokens=True)

        return decoded_inputs_a, decoded_inputs_b, decoded_targets, decoded_outputs, batch["task"]

    def test_epoch_end(self, outputs):
        decoded_inputs_a, decoded_inputs_b, decoded_targets, decoded_outputs = [], [], [], []
        for batch in outputs:
            decoded_inputs_a += batch[0]
            decoded_inputs_b += batch[1]
            decoded_targets += batch[2]
            decoded_outputs += batch[3]
            task = batch[4]

        rouge = ROUGEScore()
        rouge_score = rouge(decoded_outputs, decoded_targets)
        self.log("%s_test_rouge1"%task, rouge_score["rouge1_fmeasure"], sync_dist=True)           
        self.log("%s_test_rouge2"%task, rouge_score["rouge2_fmeasure"], sync_dist=True)           
        self.log("%s_test_rougeL"%task, rouge_score["rougeL_fmeasure"], sync_dist=True)           
        
        # save generation results (from device 0)
        if self.trainer.is_global_zero:
            with open(os.path.join(self.trainer.log_dir, "%s_test_results.txt" % task), "w") as f:
                for sents in zip(decoded_inputs_a, decoded_inputs_b, decoded_targets, decoded_outputs):
                    f.write("\n".join(sents))
                    f.write("\n\n")


    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        fast_train = ["encoder.add", "encoder.diff", "encoder.extcomp", "encoder.abscomp", "decoder"]
        #fast_train = ["decoder"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay)) and (not any(ft in n for ft in fast_train))], 
                "weight_decay": self.args.weight_decay, 
                "lr": self.args.learning_rate,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and (not any(ft in n for ft in fast_train))], 
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if (not any(nd in n for nd in no_decay)) and (any(ft in n for ft in fast_train))], 
                "weight_decay": self.args.weight_decay, 
                "lr": self.args.fast_learning_rate,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and (any(ft in n for ft in fast_train))], 
                "weight_decay": 0.0,
                "lr": self.args.fast_learning_rate,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.train_steps)

        return {
                    "optimizer": optimizer, 
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1
                    }
        }