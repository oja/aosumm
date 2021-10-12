import copy
import bisect
import numpy as np

import torch
from torch._C import ClassType, device, dtype
import torch.nn as nn
from pytorch_transformers import BertModel, BertConfig
from torch.nn.init import xavier_uniform_

from models.decoder import TransformerDecoder
from models.encoder import Classifier, ExtTransformerEncoder
from models.optimizers import Optimizer
from others.logging import logger, init_logger


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))


    return optim

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_bert)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps_dec)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]
    optim.set_parameters(params)


    return optim


def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if(large):
            self.model = BertModel.from_pretrained('bert-large-cased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-cased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec, _ = self.model(x, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)
        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size,
                                     num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads, intermediate_size=args.ext_ff_size)
            self.bert.model = BertModel(bert_config)
            self.ext_layer = Classifier(self.bert.model.config.hidden_size)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def _chunk_to_limit(self, src, segs, clss, mask_src, mask_cls, limit):
        src_list = []
        segs_list = []
        clss_list = []
        mask_src_list = []
        mask_cls_list = []
        
        original_cls_len = len(clss)
        clss = torch.cat((clss, torch.tensor([len(src) - 1]).to("cuda"))).to("cuda")
        prefix_tensor = src[:clss[0]]
        suffix_tensor = torch.tensor([src[-1]], dtype=torch.int64).to("cuda")
        extra_length = len(prefix_tensor) + len(suffix_tensor)

        start = 0
        end = 1

        src = src[clss[0]:-1]
        segs = segs[clss[0]:-1]
        mask_src = mask_src[clss[0]:-1]

        clss = torch.sub(clss, clss[0])

        while end < len(clss):
            while end + 1 < len(clss) and clss[end + 1] - clss[start] < limit - extra_length:
                end += 1
            
            if len(src[clss[start]:clss[end]]) == 0:
                print(f"{src[clss[start]:clss[end]]=}")
                break
            
            

            src_list.append(
                torch.cat((prefix_tensor, src[clss[start]:clss[end]], suffix_tensor))
            )
            

            next_segs = segs[clss[start]:clss[end]]
            prefix_tensor_segs = (torch.zeros(prefix_tensor.shape, dtype=torch.int64) if next_segs[0] == 1 else torch.ones(prefix_tensor.shape, dtype=torch.int64)).to("cuda")
            suffix_tensor_segs = (torch.zeros(suffix_tensor.shape, dtype=torch.int64) if next_segs[-1] == 0 else torch.ones(suffix_tensor.shape, dtype=torch.int64)).to("cuda")
            segs_list.append(
                torch.cat((prefix_tensor_segs, segs[clss[start]:clss[end]], suffix_tensor_segs))
            )
            mask_src_list.append(
                torch.cat(
                    (torch.full(prefix_tensor.shape, True, dtype=torch.bool).to("cuda"), 
                    mask_src[clss[start]:clss[end]], 
                    torch.full(suffix_tensor.shape, True, dtype=torch.bool).to("cuda"))
                )
            )
            
            clss_list.append(torch.add(torch.sub(clss[start:end], clss[start]), len(prefix_tensor)))
            mask_cls_list.append(mask_cls[start:end])

            # logger.info(f"full src: {src}")
            # logger.info(f"full clss: {clss}")
            # logger.info(f"spliced src: {src_list[-1]}, length: {len(src_list[-1])}")
            # logger.info(f"spliced clss: {clss_list[-1]}, length: {len(clss_list[-1])}")

            # logger.info(f"Committing chunk ({clss[start]}, {clss[end]}), since it is the largest < limit ({limit})")
            
            start = end
            end += 1
        
        final_cls_len = sum([len(clss) for clss in clss_list])
        # logger.info(f"{final_cls_len=} {original_cls_len=}")
                
        return {'src': src_list, 
                'segs': segs_list, 
                'clss': clss_list, 
                'mask_src': mask_src_list, 
                'mask_cls': mask_cls_list}
    
    def flip_tensor(self, x):
        y = torch.zeros(x.shape, dtype=x.dtype).to("cuda")
        y[x == 0] = 1
        return y

    def forward(self, src, segs, clss, mask_src, mask_cls, limit=None):
        original_mask_cls = copy.deepcopy(mask_cls)
        chunks = self._chunk_to_limit(src[0], segs[0], clss[0], mask_src[0], mask_cls[0], limit)

        sents_vecs = []
        for src, segs, clss, mask_src, mask_cls in zip(chunks['src'], chunks['segs'], chunks['clss'], chunks['mask_src'], chunks['mask_cls']):

            if segs[0] == 1:
                segs = self.flip_tensor(segs)
                        
            top_vec = self.bert(torch.unsqueeze(src, 0), torch.unsqueeze(segs, 0), torch.unsqueeze(mask_src, 0))
            sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), torch.unsqueeze(clss, 0)]
            sents_vec = sents_vec * torch.unsqueeze(mask_cls, 0)[:, :, None].float()
            sents_vecs.append(sents_vec)

        # Concatenate the BERT representations together
        concat_sents_vecs = torch.cat(sents_vecs, 1)

        # logger.info(f"{concat_sents_vecs=}")

        # Feed them into the classification layer
        sent_scores = self.ext_layer(concat_sents_vecs, original_mask_cls).squeeze(-1)

        # logger.info(f"new sent scores: {sent_scores} {sent_scores.shape}")

        return sent_scores, original_mask_cls


class AbsSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint=None, bert_from_extractive=None):
        super(AbsSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)

        if bert_from_extractive is not None:
            self.bert.model.load_state_dict(
                dict([(n[11:], p) for n, p in bert_from_extractive.items() if n.startswith('bert.model')]), strict=True)

        if (args.encoder == 'baseline'):
            bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.enc_hidden_size,
                                     num_hidden_layers=args.enc_layers, num_attention_heads=8,
                                     intermediate_size=args.enc_ff_size,
                                     hidden_dropout_prob=args.enc_dropout,
                                     attention_probs_dropout_prob=args.enc_dropout)
            self.bert.model = BertModel(bert_config)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.model.config.vocab_size
        tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
        if (self.args.share_emb):
            tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.dec_heads,
            d_ff=self.args.dec_ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)

        self.generator = get_generator(self.vocab_size, self.args.dec_hidden_size, device)
        self.generator[0].weight = self.decoder.embeddings.weight


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if(args.use_bert_emb):
                tgt_embeddings = nn.Embedding(self.vocab_size, self.bert.model.config.hidden_size, padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(self.bert.model.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
                self.generator[0].weight = self.decoder.embeddings.weight

        self.to(device)

    def forward(self, src, tgt, segs, clss, mask_src, mask_tgt, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        dec_state = self.decoder.init_decoder_state(src, top_vec)
        decoder_outputs, state = self.decoder(tgt[:, :-1], top_vec, dec_state)
        return decoder_outputs, None
