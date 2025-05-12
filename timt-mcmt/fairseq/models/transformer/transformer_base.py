# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from base64 import encode
import imp
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor, embedding
from torch.autograd import Function, Variable
import torch.nn.functional as F
import numpy as np
import logging

from argparse import Namespace
from .vq_function import vq, vq_st
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import (
    TransformerConfig,
    TransformerDecoderBase,
    TransformerEncoderBase,
    MultimodalTransformerEncoderBase,
    MultimodalTransformerDecoderBase,
    MultimodalTransformerEncoderNewBase
)
from fairseq.modules import MultiheadAttention

logger = logging.getLogger(__name__)




class TransformerModelBase(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)



class MultimodalTransformerModelBase(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True
        self.discr = Discriminator(cfg, max_step=cfg.discriminator_max_step)
        if cfg.transalation_checkpoint is not None:
            self.load_state_dict(torch.load(cfg.transalation_checkpoint)['model'],strict=False)
            

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return MultimodalTransformerEncoderNewBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return MultimodalTransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        recon_prev_output_tokens = None,
        imgs_list = None,
        correct_src_tokens = None,
        correct_src_lengths = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        if correct_src_tokens is not None and imgs_list is not None:
            correct_encoder_out = self.encoder(
                correct_src_tokens, src_lengths=correct_src_lengths, return_all_hiddens=return_all_hiddens
            )
            encoder_out = self.encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, 
                imgs_list=imgs_list, correct_encoder_out=correct_encoder_out['encoder_out'][0]
            )
        else:
            encoder_out = self.encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens, imgs_list=imgs_list,
            )

        img_text_decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        
        #correcttext_text
        cortext_text_decoder_out = None
        if correct_src_tokens is not None:

            if imgs_list is not None:
                z_q_x_st, cor_commit_loss, cor_text_indices = self.encoder.codebook(correct_encoder_out["encoder_out"][0],ret_indices = True, freeze = False)
                has_pads = src_tokens.device.type == "xla"
                if has_pads:
                    z_q_x_st = z_q_x_st * (1 - correct_encoder_out['encoder_padding_mask'][0].transpose(0, 1).unsqueeze(-1).type_as(z_q_x_st))

                image_discr_out = self.discr(encoder_out["encoder_out"][0], encoder_out["encoder_padding_mask"][0])
                
                correct_encoder_cat_out = torch.cat((correct_encoder_out["encoder_out"][0], z_q_x_st),axis=0)
                correct_encoder_cat_padding_mask = torch.cat((correct_encoder_out["encoder_padding_mask"][0], correct_encoder_out["encoder_padding_mask"][0]),axis=1)
                correct_discr_out = self.discr(correct_encoder_cat_out, correct_encoder_cat_padding_mask, grad_backprop=False)
                
                # code-conditioned mask translation task
                mask_encoder_out = mask_tensor(correct_encoder_out['encoder_out'][0].clone(), self.cfg.mask_probability, correct_encoder_out['encoder_padding_mask'][0].cpu())
                correct_encoder_out['encoder_out'][0] = torch.cat((mask_encoder_out, encoder_out['z_q_x_st_i']),axis=0)
                correct_encoder_out['encoder_padding_mask'][0] = torch.cat((correct_encoder_out['encoder_padding_mask'][0], encoder_out['origin_encoder_padding_mask']),axis=1)


                mask_decoder_out = self.decoder(
                    prev_output_tokens,
                    encoder_out=correct_encoder_out,
                    features_only=features_only,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )

                return (img_text_decoder_out, cortext_text_decoder_out, mask_decoder_out,
                        encoder_out['img_commit_loss'],cor_commit_loss,
                        encoder_out['z_q_x_st_i'],z_q_x_st,
                        encoder_out['img_indices'].permute(1, 0).contiguous(),cor_text_indices.permute(1, 0).contiguous(),
                        encoder_out['img_attn_weight'],img_text_decoder_out[1]['attn'][0][:,:,:],
                        image_discr_out, correct_discr_out)

        

        return img_text_decoder_out, None, None, None, None, None, None, None, None, None, None, None, None

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg=None,
        args: Optional[Namespace] = None,
    ):

        return super().load_state_dict(state_dict,False,model_cfg,args)

# GradReverse layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None
    
class Discriminator(nn.Module):
    def __init__(self, cfg, max_step=100000, filter_sizes=[3,4,5], output_dim=2, dropout=0.1):
        super(Discriminator, self).__init__()
        self.max_step = max_step
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=cfg.encoder.embed_dim, out_channels=cfg.discriminator_hidden_dim, kernel_size=fs, padding=fs-2) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * cfg.discriminator_hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.count = 0
    def forward(self, hidden, encoder_padding_mask, grad_backprop=True):
        if grad_backprop is False:
            hidden = hidden.detach()
        else:
            p  = min(self.count / self.max_step, 1)
            self.count += 1
            # hyp_lambad from 0 to 1
            hyp_lambda = 2 / (1+np.exp(-10 * p)) - 1
            hidden = GradReverse.apply(hidden, hyp_lambda)  # shape: (max_seq_len, batch_size, input_dim)

        hidden = hidden.permute(1, 2, 0)  # shape: (batch_size, input_dim, max_seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_output = conv(hidden)  # [batch_size, num_filters, seq_len - filter_size + 1]
            conv_output = nn.functional.relu(conv_output)
            conv_output = torch.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)  # [batch_size, num_filters]
            conv_outputs.append(conv_output)
        

        cat = self.dropout(torch.cat(conv_outputs, dim=1))  # shape: (batch_size, len(filter_sizes) * num_filters)
        
        logits = self.fc(cat)  # shape: (batch_size, output_dim)
        return logits
    

class DvaeMultimodalTransformerModelBase(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True
        if cfg.transalation_checkpoint is not None:
            self.load_state_dict(torch.load(cfg.transalation_checkpoint)['model'],strict=False)
        self.discr = Discriminator(cfg, max_step=cfg.discriminator_max_step)



    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return MultimodalTransformerEncoderNewBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return MultimodalTransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        recon_prev_output_tokens = None,
        imgs_list = None,
        correct_src_tokens = None,
        correct_src_lengths = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        if correct_src_tokens is not None:
            #stage 3
            if imgs_list is not None:
                
                
                correct_encoder_out = self.encoder(
                    correct_src_tokens, src_lengths=correct_src_lengths, return_all_hiddens=return_all_hiddens,imgs_list=imgs_list,
                )
                z_q_x_st, cor_commit_loss, cor_text_indices = self.encoder.codebook(correct_encoder_out["origin_encoder_out"].detach(),ret_indices = True)
                has_pads = src_tokens.device.type == "xla"
                if has_pads:
                    z_q_x_st = z_q_x_st * (1 - correct_encoder_out['encoder_padding_mask'][0].transpose(0, 1).unsqueeze(-1).type_as(z_q_x_st))
                z_q_x_st_i = correct_encoder_out["z_q_x_st_i"]
                
                text_part_mask = torch.ones_like(correct_encoder_out['origin_encoder_padding_mask'])
                text_part_mask = torch.cat((correct_encoder_out['origin_encoder_padding_mask'],text_part_mask),axis=1)
                
                mask_encoder_out = torch.cat((correct_encoder_out['origin_encoder_out'].detach(), correct_encoder_out['z_q_x_st_i']),axis=0)
                mask_encoder_out = mask_tensor(mask_encoder_out, self.cfg.mask_probability, text_part_mask.cpu())

                mask_discr_out = self.discr(mask_encoder_out, correct_encoder_out['encoder_padding_mask'][0])
                
                correct_encoder_cat_out = torch.cat((correct_encoder_out["origin_encoder_out"],z_q_x_st),axis=0)
                correct_encoder_cat_padding_mask = torch.cat((correct_encoder_out["origin_encoder_padding_mask"], correct_encoder_out["origin_encoder_padding_mask"]),axis=1)
                discr_out = self.discr(correct_encoder_cat_out.detach(), correct_encoder_cat_padding_mask, grad_backprop=False)
                z_q_x_st = z_q_x_st.permute(1, 0, 2).contiguous()
                z_q_x_st_i = z_q_x_st_i.permute(1, 0, 2).contiguous()
                return (cor_commit_loss,correct_encoder_out["img_commit_loss"], z_q_x_st, z_q_x_st_i,
                        cor_text_indices.permute(1, 0).contiguous(),correct_encoder_out["img_indices"].permute(1, 0).contiguous(),
                        mask_discr_out, discr_out
                        )
                
            # stage 2
            else:
                    
                correct_encoder_out = self.encoder(
                    correct_src_tokens, src_lengths=correct_src_lengths, return_all_hiddens=return_all_hiddens
                )
                decoder_out = self.decoder(
                    prev_output_tokens,
                    encoder_out=correct_encoder_out,
                    features_only=features_only,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )

                z_q_x_st, cor_commit_loss, cor_text_indices = self.encoder.codebook(correct_encoder_out["encoder_out"][0].detach(),ret_indices = True)
                has_pads = src_tokens.device.type == "xla"
                if has_pads:
                    z_q_x_st = z_q_x_st * (1 - correct_encoder_out['encoder_padding_mask'][0].transpose(0, 1).unsqueeze(-1).type_as(z_q_x_st))
                

                mask_encoder_out = mask_tensor(correct_encoder_out['encoder_out'][0].clone(), self.cfg.mask_probability, correct_encoder_out['encoder_padding_mask'][0].cpu())
                correct_encoder_out['encoder_out'][0] = torch.cat((mask_encoder_out,z_q_x_st),axis=0)
                correct_encoder_out['encoder_padding_mask'][0] = torch.cat((correct_encoder_out['encoder_padding_mask'][0], correct_encoder_out['encoder_padding_mask'][0]),axis=1)
                

                mask_decoder_out = self.decoder(
                    prev_output_tokens,
                    encoder_out=correct_encoder_out,
                    features_only=features_only,
                    alignment_layer=alignment_layer,
                    alignment_heads=alignment_heads,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )

                return (decoder_out, mask_decoder_out, cor_commit_loss,  z_q_x_st, cor_text_indices.permute(1, 0).contiguous())

        

        return None, None, None, None

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)
    
    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg=None,
        args: Optional[Namespace] = None,
    ):

        return super().load_state_dict(state_dict,False,model_cfg,args)
        
def get_neg_pair(z_q_x_st, z_q_x_st_i):
    neg_pair_list = []
    for i in range(z_q_x_st.shape[0]):
        neg_pair_list.append(randint_exclude(0,z_q_x_st.shape[0], i))
    
    return (z_q_x_st,z_q_x_st_i[neg_pair_list])

def get_pos_pair(z_q_x_st, z_q_x_st_i):
    return (z_q_x_st,z_q_x_st_i)
    
def randint_exclude(low, high, exclude):

    int_list = list(range(low, high))
    if exclude in int_list: 
        int_list.remove(exclude)

    np.random.shuffle(int_list)

    return int_list.pop()

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def mask_tensor(input_tensor, mask_probability, pad_indices):
    probability_matrix = torch.full(pad_indices.shape, mask_probability)
    probability_matrix.masked_fill_(pad_indices, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    masked_indices = masked_indices.transpose(0, 1)
    input_tensor[masked_indices] = 0.0

    return input_tensor