import logging

import einops
import torch
import torch.nn as nn
from video_llama.models.blip2 import Blip2Base, disabled_train
from video_llama.models.ImageBind.models import imagebind_model
from video_llama.models.ImageBind.models.imagebind_model import ModalityType
from video_llama.models.Qformer import BertConfig, BertLMHeadModel


class VideoLLAMABackbone(Blip2Base):

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/" +\
            "LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        output_embed_size=5120,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        frozen_llama_proj=True,
        frozen_video_Qformer=True,
        frozen_audio_Qformer=True,
        llama_proj_model="",
        max_frame_pos=32,
        num_video_query_token=32,
        num_audio_query_token=8,
        imagebind_ckpt_path="/code/Video-LLaMA/ckpt",
        equip_audio_branch=True
    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer()

        logging.info("Loading VIT")
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for _, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for _, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        logging.info("Loading VIT Done")

        logging.info("Loading Q-Former")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for _, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info("Loading Q-Former Done")

        logging.info("Loading LLAMA proj")
        self.llama_proj = nn.Linear(self.Qformer.config.hidden_size, output_embed_size)
        if llama_proj_model:
            logging.info("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            self.load_state_dict(llama_proj_weight["model"], strict=False)

        if frozen_llama_proj:
            for _, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info("LLAMA proj is frozen")
        else:
            for _, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info("LLAMA proj is not frozen")
        logging.info("Loading llama_proj Done")

        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)

        self.num_video_query_token = num_video_query_token
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(
            num_query_token=num_video_query_token,
            vision_width=self.Qformer.config.hidden_size,
            num_hidden_layers=2
        )

        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if frozen_video_Qformer:
            for _, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for _, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
            
            logging.info("video_Qformer is frozen")
        else:
            for _, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for _, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            logging.info("video_Qformer is not frozen")

        if frozen_video_Qformer and (not frozen_audio_Qformer):
            self.train_flag = 1 # audio_Qformer
        elif not(frozen_video_Qformer) and frozen_audio_Qformer:
            self.train_flag = 0 # video_Qformer
        elif not(frozen_video_Qformer) and not(frozen_audio_Qformer):
            self.train_flag = 2 # video_Qformer and AL trained
        else:
            self.train_flag = 3

        if equip_audio_branch:
            logging.info(f"Initializing audio encoder from {imagebind_ckpt_path} ...")
            self.audio_encoder,self.audio_hidden_size = \
                imagebind_model.imagebind_huge()
            self.audio_encoder.load_state_dict(
                torch.load("{}/imagebind_huge.pth".format(imagebind_ckpt_path))
            )
            # free vision encoder
            for _, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            logging.info("audio encoder initialized.")
            
            self.num_audio_query_token = num_audio_query_token
            self.audio_Qformer,self.audio_query_tokens = self.init_video_Qformer(
                num_query_token = self.num_audio_query_token,
                vision_width=self.audio_hidden_size,
                num_hidden_layers=2
            )
            self.audio_Qformer.cls = None
            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.audio_llama_proj = nn.Linear(self.audio_Qformer.config.hidden_size, output_embed_size)
            self.audio_position_embedding = nn.Embedding(8, self.audio_hidden_size)

            if frozen_audio_Qformer:
                for _, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = False
                self.audio_query_tokens.requires_grad = False
                for _, param in self.audio_llama_proj.named_parameters():
                    param.requires_grad = False
                for _, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = False
                logging.info("audio_Qformer and audio-LLAMA proj is frozen")
            else:
                for _, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = True
                self.audio_query_tokens.requires_grad = True
                for _, param in self.audio_llama_proj.named_parameters():
                    param.requires_grad = True
                for _, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = True
                logging.info("audio_Qformer is not frozen")

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_videoQformer(self, image):
        device = image.device
         
        # input shape b, c, t, h, w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, "b c t h w -> (b t) c h w")
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, "(b t) q h -> b t q h",b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # frame attention
            frame_hidden_state = einops.rearrange(frame_hidden_state, "b t q h -> b (t q) h",b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens,
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
            )
            video_hidden = video_query_output.last_hidden_state
            llama_proj_embeds = self.llama_proj(video_hidden)
            
        return llama_proj_embeds
    
    # input audio shape [b t c h w]
    def encode_audioQformer(self, audio, modality_type=ModalityType.AUDIO):
        device = audio.device
        with self.maybe_autocast():
            _, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(
                audio, modality_type=modality_type
            )
            batch_size,time_length = audio.size()[:2]

            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            audio_position_embeddings = self.audio_position_embedding(position_ids)
            audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

            audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
            frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

            audio_query_output = self.audio_Qformer.bert(
                query_embeds=audio_query_tokens,  # [32, 768]
                encoder_hidden_states=audio_imagebind_finalout,
                encoder_attention_mask=frame_atts,
                return_dict=True,
            )
            audio_hidden = audio_query_output.last_hidden_state
            llama_proj_embeds = self.audio_llama_proj(audio_hidden)
    
        return llama_proj_embeds

    def forward(self, video, audio=None):
        if self.train_flag == 0: # only video branch
            embeds = self.encode_videoQformer(video) # [b c t h w]
        elif self.train_flag == 1: # only audio branch
            # video = einops.rearrange(video, 'b c t h w -> b t c h w')
            embeds = self.encode_audioQformer(audio)
        else: # training on audio + video branchs
            video_embeds = self.encode_videoQformer(video) # video: [b c t h w] -> [b, 32, 4096]
            audio_embeds = self.encode_audioQformer(audio) # audio: [b t c h w] -> [b, 8,  4096]
            embeds = torch.cat((video_embeds, audio_embeds), dim=1)
        
        return embeds

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get(
            "q_former_model",
            "https://storage.googleapis.com/sfr-vision-language-research" + \
            "/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
        )
        img_size = cfg.get("image_size", 224)
        num_query_token = cfg.get("num_query_token", 32)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        
        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)
        frozen_audio_Qformer = cfg.get("frozen_audio_Qformer", True)

        llama_proj_model = cfg.get("llama_proj_model", "")
        max_frame_pos = cfg.get("max_frame_pos", 32)
        num_video_query_token =  cfg.get("num_video_query_token", 32)

        equip_audio_branch= cfg.get("equip_audio_branch", True)
        num_audio_query_token =  cfg.get("num_audio_query_token", 8)
        imagebind_ckpt_path = cfg.get("imagebind_ckpt_path", "/code/Video-LLaMA/ckpt")
        
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            max_frame_pos=max_frame_pos,
            frozen_llama_proj=frozen_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            num_video_query_token=num_video_query_token,
            num_audio_query_token=num_audio_query_token,
            imagebind_ckpt_path=imagebind_ckpt_path,
            equip_audio_branch=equip_audio_branch,
            llama_proj_model=llama_proj_model,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)
        ckpt_path_2 = cfg.get("ckpt_2", "")  
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)
        
        return model
