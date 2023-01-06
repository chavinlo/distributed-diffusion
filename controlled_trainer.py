# Install bitsandbytes:
# `nvcc --version` to get CUDA version.
# `pip install -i https://test.pypi.org/simple/ bitsandbytes-cudaXXX` to install for current CUDA.
# Example Usage:
# Single GPU: torchrun --nproc_per_node=1 trainer/diffusers_trainer.py --model="CompVis/stable-diffusion-v1-4" --run_name="liminal" --dataset="liminal-dataset" --hf_token="hf_blablabla" --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=1 --fp16=True --image_log_steps=250 --epochs=20 --resolution=768 --use_ema=True
# Multiple GPUs: torchrun --nproc_per_node=N trainer/diffusers_trainer.py --model="CompVis/stable-diffusion-v1-4" --run_name="liminal" --dataset="liminal-dataset" --hf_token="hf_blablabla" --bucket_side_min=64 --use_8bit_adam=True --gradient_checkpointing=True --batch_size=10 --fp16=True --image_log_steps=250 --epochs=20 --resolution=768 --use_ema=True

import gc
import ipaddress
import itertools
import json
import os
import pickle
import random
import resource
import shutil
import socket
import sqlite3
import threading
import time
import traceback
from datetime import datetime

import diffusers
import hivemind
import numpy as np
import omegaconf
import psutil
import pynvml
import requests
import torch
import tqdm
import transformers
import wandb

try:
    pynvml.nvmlInit()
except pynvml.nvml.NVMLError_LibraryNotFound:
    pynvml = None

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, PNDMScheduler, DDIMScheduler, \
    StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.optimization import get_scheduler
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from hivemind import Float16Compression

from utils.data import ImageStore, SimpleBucket, AspectDataset, EMAModel

torch.backends.cuda.matmul.allow_tf32 = True

# Connect to the database
conn = sqlite3.connect('danbooru.db')
cursor = conn.cursor()

# Load the posts table into memory
cursor.execute('SELECT * FROM posts')
posts = cursor.fetchall()


def select_random_post():
    # Select a random record from the posts table
    random_post = random.choice(posts)
    post_id = random_post[0]
    image_ext = random_post[1]
    rating = random_post[2]

    # Return the post_id, image_ext, and rating
    return post_id, image_ext, rating


# Get the total number of rows in the posts table
cursor.execute('SELECT COUNT(*) FROM posts')
num_rows = cursor.fetchone()[0]


def get_num_rows():
    return num_rows


# Close the connection to the database
conn.close()


class StopTrainingException(Exception):
    pass


def get_gpu_ram() -> str:
    """
    Returns memory usage statistics for the CPU, GPU, and Torch.

    :return:
    """
    gpu_str = ""
    torch_str = ""
    try:
        cudadev = torch.cuda.current_device()
        nvml_device = pynvml.nvmlDeviceGetHandleByIndex(cudadev)
        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_device)
        gpu_total = int(gpu_info.total / 1E6)
        gpu_free = int(gpu_info.free / 1E6)
        gpu_used = int(gpu_info.used / 1E6)
        gpu_str = f"GPU: (U: {gpu_used:,}mb F: {gpu_free:,}mb " \
                  f"T: {gpu_total:,}mb) "
        torch_reserved_gpu = int(torch.cuda.memory.memory_reserved() / 1E6)
        torch_reserved_max = int(torch.cuda.memory.max_memory_reserved() / 1E6)
        torch_used_gpu = int(torch.cuda.memory_allocated() / 1E6)
        torch_max_used_gpu = int(torch.cuda.max_memory_allocated() / 1E6)
        torch_str = f"TORCH: (R: {torch_reserved_gpu:,}mb/" \
                    f"{torch_reserved_max:,}mb, " \
                    f"A: {torch_used_gpu:,}mb/{torch_max_used_gpu:,}mb)"
    except AssertionError:
        pass
    cpu_maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1E3 +
                     resource.getrusage(
                         resource.RUSAGE_CHILDREN).ru_maxrss / 1E3)
    cpu_vmem = psutil.virtual_memory()
    cpu_free = int(cpu_vmem.free / 1E6)
    return f"CPU: (maxrss: {cpu_maxrss:,}mb F: {cpu_free:,}mb) " \
           f"{gpu_str}" \
           f"{torch_str}"


def setuphivemind(conf, log_queue):
    log_queue.put("Setting up hivemind")
    if os.path.exists(conf.intern.workingdir):
        shutil.rmtree(conf.intern.workingdir)
    os.makedirs(conf.intern.workingdir)

def pingGeo(server):
    requests.get('https://' + server + '/api/v1/get/announcelocation')

def sendLoss(server, secretpass, currloss):
    import socket
    requests.post(
        'https://' + server + '/api/v1/private/postloss',
         auth=(socket.gethostname(), secretpass),
         json={"time": time.time(), 'loss': currloss}
         )


def getchunk(amount, conf, log_queue):
    log_queue.put("Requesting Chunks")
    if os.path.isdir(conf.intern.tmpdataset):
        shutil.rmtree(conf.intern.tmpdataset)
    os.mkdir(conf.intern.tmpdataset)

    # Select 500 random records from the posts table
    random_posts = [select_random_post() for _ in range(int(amount))]

    threads = []
    for post_id, image_ext, rating in random_posts:
        t = threading.Thread(target=download_image, args=(post_id, image_ext, conf,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    log_queue.put("Chunks ready")


def download_image(post_id, image_ext, conf):

    try:
        # Download the image
        image_url = f"https://crowdcloud.us-southeast-1.linodeobjects.com/crowdcloud/opendataset/v1/danbooru/{post_id}.{image_ext}"
        image_response = requests.get(image_url)
        if image_response.status_code != 200:
            raise ConnectionError("Image Response not 200")
        open(f"{conf.intern.tmpdataset}/{post_id}.{image_ext}", 'wb').write(image_response.content)

        # Download the tags
        tags_url = f"https://crowdcloud.us-southeast-1.linodeobjects.com/crowdcloud/opendataset/v1/danbooru/{post_id}.json"
        tags_response = requests.get(tags_url)
        if tags_response.status_code != 200:
            raise ConnectionError("Tags Response not 200")
        tags = tags_response.json()['tags']
        if not tags:
            raise ConnectionError("Tags is empty")
        open(f"{conf.intern.tmpdataset}/{post_id}.txt", 'w').write(', '.join(tags))
    except ConnectionError as e:
        print("Failed to retrieve entry", post_id)
        print("Reason:", e)

        if os.path.exists(f"{conf.intern.tmpdataset}/{post_id}.{image_ext}"):
            os.remove(f"{conf.intern.tmpdataset}/{post_id}.{image_ext}")

        if os.path.exists(f"{conf.intern.tmpdataset}/{post_id}.txt"):
            os.remove(f"{conf.intern.tmpdataset}/{post_id}.txt")

def dataloader(tokenizer, text_encoder, device, world_size, rank, conf, log_queue):
    # load dataset
    log_queue.put("Setting up ImageStore")
    store = ImageStore(conf.intern.tmpdataset, conf)
    log_queue.put("Setting up AspectDataset")
    dataset = AspectDataset(store, tokenizer, text_encoder, device, conf, ucg=float(conf.everyone.ucg))
    log_queue.put("Setting up SimpleBucket")
    sampler = SimpleBucket(
        store=store,
        batch_size=int(conf.batchSize),
        shuffle=conf.everyone.buckets_shuffle,
        resize=conf.image_store_resize,
        image_side_min=int(conf.everyone.buckets_side_min),
        image_side_max=int(conf.everyone.buckets_side_max),
        image_side_divisor=64,
        max_image_area=int(conf.everyone.resolution) ** 2,
        num_replicas=world_size,
        rank=rank
    )
    out_length = "Store Length: " + str(len(store))
    log_queue.put(str(out_length))
    log_queue.put("Setting up Dataloader")
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

    return train_dataloader


# This will be initialized on a separate thread and killed when everything finishes
def informationExchangeServer(conf, maddrs):
    internal_port = int(conf.internal_ie)

    from threading import Thread
    from flask import Flask, jsonify

    app = Flask(__name__)

    @app.route('/peer')
    def peer():
        return maddrs

    @app.route('/globalconf')
    def globalconf():
        dict_with_configuration = {
            "model": conf.everyone.model,
            "extended_chunks": conf.everyone.extended_chunks,
            "clip_penultimate": conf.everyone.clip_penultimate,
            "fp16": conf.everyone.fp16,
            "resolution": conf.everyone.resolution,
            "seed": conf.everyone.seed,
            "train_text_encoder": conf.everyone.train_text_encoder,
            "lr": conf.everyone.lr,
            "ucg": conf.everyone.ucg,
            "use_ema": conf.everyone.use_ema,
            # Advanced, do not touch
            "opt_betas_one": conf.everyone.opt_betas_one,
            "opt_betas_two": conf.everyone.opt_betas_two,
            "opt_epsilon": conf.everyone.opt_epsilon,
            "opt_weight_decay": conf.everyone.opt_weight_decay,
            "buckets_shuffle": conf.everyone.buckets_shuffle,
            "buckets_side_min": conf.everyone.buckets_side_min,
            "buckets_side_max": conf.everyone.buckets_side_max,
            # Recheck this in the future if we get grad offloading with HM
        }
        return jsonify(dict_with_configuration)

    def run():
        app.run(host="0.0.0.0", port=internal_port)

    thread = Thread(target=run, )
    return thread


class DistributedTrainer:
    """The core trainer for SD, distributed over the Hivemind"""

    def __init__(self, _command_queue, _log_queue, _conf):
        self.command_queue = _command_queue
        self.log_queue = _log_queue
        self.conf = _conf

        self.imgs_per_epoch = get_num_rows()
        self.total_epochs = 10
        self.config = omegaconf.OmegaConf.create(self.conf)
        self.rank = 0
        self.device = torch.device('cuda')
        self.global_step = 0  # TODO: Do we need to read the global step from the swarm?

        torch.cuda.set_device(self.rank)
        os.makedirs(self.config.intern.workingdir, exist_ok=True)

        mode = 'disabled'
        if self.config.enable_wandb:
            mode = 'online'
        self.run = wandb.init(project="Hivemind Project", name="Hivemind", config=self.config,
                              dir=self.config.intern.workingdir + '/wandb', mode=mode)

        self._log_debugging()

        # setup fp16 stuff
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.conf.everyone.fp16)

        # Set seed
        torch.manual_seed(self.conf.everyone.seed)
        random.seed(self.conf.everyone.seed)
        np.random.seed(self.conf.everyone.seed)

        # I think the hf token is set to an empty string, and not None, so we should be ok. thx js
        self.tokenizer = CLIPTokenizer.from_pretrained(self.conf.everyone.model, subfolder='tokenizer',
                                                       use_auth_token=self.conf.hftoken)
        self.text_encoder = CLIPTextModel.from_pretrained(self.conf.everyone.model, subfolder='text_encoder',
                                                          use_auth_token=self.conf.hftoken)
        self.vae = AutoencoderKL.from_pretrained(self.conf.everyone.model, subfolder='vae',
                                                 use_auth_token=self.conf.hftoken)
        self.unet = UNet2DConditionModel.from_pretrained(self.conf.everyone.model, subfolder='unet',
                                                         use_auth_token=self.conf.hftoken)

        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        if not self.conf.everyone.train_text_encoder:
            self.text_encoder.requires_grad_(False)

        if self.conf.gradckpt:
            self.unet.enable_gradient_checkpointing()
            if self.conf.everyone.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        if self.conf.xformers:
            self.unet.set_use_memory_efficient_attention_xformers(True)

        # "The “safer” approach would be to move the model to the device first and create the optimizer afterwards."
        self.weight_dtype = torch.float16 if self.conf.everyone.fp16 else torch.float32

        # move models to device
        self.vae = self.vae.to(self.device, dtype=self.weight_dtype)
        self.unet = self.unet.to(self.device, dtype=torch.float32)
        self.text_encoder = self.text_encoder.to(self.device,
                                                 dtype=self.weight_dtype if not self.conf.everyone.train_text_encoder else torch.float32)

        if self.conf.eightbitadam:  # Bits and bytes is only supported on certain CUDA setups, so default to regular adam if it fails.
            try:
                import bitsandbytes as bnb
                self.optimizer_cls = bnb.optim.AdamW8bit
            except Exception:
                self.log_queue.put('bitsandbytes not supported, using regular Adam optimizer')
                self.optimizer_cls = torch.optim.AdamW
        else:
            self.optimizer_cls = torch.optim.AdamW

        """
        optimizer = optimizer_cls(
            unet.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
        """

        self.optimizer_parameters = self.unet.parameters() if not self.conf.everyone.train_text_encoder else itertools.chain(
            self.unet.parameters(), self.text_encoder.parameters())

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.conf.everyone.model,
            subfolder='scheduler',
            use_auth_token=self.conf.hftoken,
        )

        # Hivemind Setup
        # get network peers (if mother peer then ignore)
        if self.conf.mother is False:
            rmaddrs_rq = requests.get('http://' + self.conf.server + "/peer")
            if rmaddrs_rq.status_code == 200:
                _peer_list = json.loads(rmaddrs_rq.content)
            else:
                self.log_queue.put("Unable to obtain peers from server")
                raise ConnectionError("Unable to obtain peers from server")
        else:
            _peer_list = None

        self.log_queue.put("Trainer set to " + self.conf.trainermode + " mode")
        if self.conf.trainermode == "Client":
            _client_mode = True
            _host_maddrs_full = None
            _public_maddrs_full = None
        elif self.conf.trainermode == "Relay":
            _client_mode = False

            # set local maddrs ports
            # Warning: It is highly recommended to not use UDP due to it being very unstable.
            # Further information: https://discord.com/channels/865254854262652969/865254854967427124/1057781288318271518
            host_maddrs_tcp = "/ip4/0.0.0.0/tcp/" + str(self.conf.internal_tcp)
            _host_maddrs_full = [host_maddrs_tcp]

            # set public to-be-announced maddrs
            # get public ip
            if self.conf.publicip == "":
                self.conf.publicip = None

            if self.conf.publicip == "auto" or self.conf.publicip is None:
                _ip = self._get_ip_online()
                self.log_queue.put("IP Detected!")
                self.ipsrc = "online"
            else:
                self.log_queue.put("Loading public IP from configuration")
                _ip = self.conf.publicip
                self.ipsrc = "config"

            # check if valid ip
            try:
                self.ip = str(ipaddress.ip_address(_ip))
            except Exception:
                self.log_queue.put("Invalid IP, please check the configuration file. IP Source: " + self.ipsrc)
                raise ValueError("Invalid IP, please check the configuration file. IP Source: " + self.ipsrc)
            
            public_maddrs_tcp = "/ip4/" + self.ip + "/tcp/" + str(self.conf.external_tcp)
            _public_maddrs_full = [public_maddrs_tcp]
        else:
            raise ValueError(f"Trainer mode {self.conf.trainermode} currently not supported")

        # init dht
        self.dht = hivemind.DHT(
            host_maddrs=_host_maddrs_full,
            initial_peers=_peer_list,
            start=True,
            announce_maddrs=_public_maddrs_full,
            client_mode=_client_mode,
        )

        # set compression and optimizer
        _compression = Float16Compression()

        # Create distributed optimizer
        # from torch.distributed.optim import ZeroRedundancyOptimizer
        # we changed to cls for single gpu training
        self.log_queue.put("Stating standard optimizer")
        tmp_optimizer = self.optimizer_cls(
            self.optimizer_parameters,
            # optimizer_class=optimizer_cls,
            # parameters_as_bucket_view=True,
            lr=(float(self.conf.everyone.lr) * int(self.conf.batchSize)),
            betas=(float(self.conf.everyone.opt_betas_one), float(self.conf.everyone.opt_betas_two)),
            eps=float(self.conf.everyone.opt_epsilon),
            weight_decay=float(self.conf.everyone.opt_weight_decay),
        )
        self.log_queue.put("Finished standard optimizer")

        self.log_queue.put("Starting hivemind optimizer")

        self.optimizer = hivemind.Optimizer(
            dht=self.dht,
            run_id="testrun",
            batch_size_per_step=(1 * int(self.conf.batchSize)),
            target_batch_size=75000,
            optimizer=tmp_optimizer,
            use_local_updates=False,
            matchmaking_time=260.0,
            averaging_timeout=1200.0,
            allreduce_timeout=1200.0,
            load_state_timeout=1200.0,
            grad_compression=_compression,
            state_averaging_compression=_compression,
            verbose=True,
        )

        self.log_queue.put("Finished hivemind optimizer")

        self.log_queue.put("Hivemind Optimizer and DHT started successfully!")

        if self.conf.trainermode == "Relay":
            self.log_queue.put(
                "Direct Hivemind mADDR (DHT only):")
            self.log_queue.put('\n'.join(str(addr) for addr in self.dht.get_visible_maddrs()))
            if hivemind.utils.networking.choose_ip_address(self.dht.get_visible_maddrs()) == "":
                computer_ip = self.ip
            else:
                computer_ip = hivemind.utils.networking.choose_ip_address(self.dht.get_visible_maddrs())

            self.log_queue.put("Global IP: " + str(computer_ip))

        # create ema
        if self.conf.everyone.use_ema:
            self.ema_unet = EMAModel(self.unet.parameters())
            self.use_ema = True
        else:
            self.use_ema = False

        if self.conf.trainermode == "Relay":
            maddrs_list = []
            for addr in self.dht.get_visible_maddrs():
                maddrs_list.append(str(addr))
            self.iethread = informationExchangeServer(self.conf, maddrs_list)
            self.iethread.start()
            self.log_queue.put("IE Server started at: " + str(computer_ip) + ":" + str(self.conf.external_ie))

    def save_checkpoint(self):
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        if self.rank == 0:
            if self.use_ema:
                self.ema_unet.store(self.unet.parameters())
                self.ema_unet.copy_to(self.unet.parameters())
            pipeline = StableDiffusionPipeline(
                text_encoder=self.text_encoder,
                # if type(text_encoder) is not torch.nn.parallel.DistributedDataParallel else text_encoder.module,
                vae=self.vae,
                unet=self.unet,
                tokenizer=self.tokenizer,
                scheduler=PNDMScheduler.from_pretrained(self.conf.everyone.model, subfolder="scheduler",
                                                        use_auth_token=self.conf.hftoken),
                safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )
            self.log_queue.put(f'Saving checkpoint to: {self.conf.intern.workingdir}/{"hivemind"}_{time_str}')
            pipeline.save_pretrained(f'{self.conf.intern.workingdir}/{"hivemind"}_{time_str}')
            self.log_queue.put("Checkpoint Saved")

            if self.use_ema:
                self.ema_unet.restore(self.unet.parameters())

    def train(self):
        # train!
        try:
            if self.optimizer.tracker.global_progress.epoch >= 1:
                self.log_queue.put("Syncronizing")
                self.optimizer.load_state_from_peers()
            while True:
                getchunk(self.conf.imageCount, self.conf, self.log_queue)

                # Note: we removed worldsize here
                train_dataloader = dataloader(self.tokenizer,
                                              self.text_encoder,
                                              self.device,
                                              1,
                                              self.rank,
                                              self.conf,
                                              self.log_queue)
                num_steps_per_epoch = len(train_dataloader)
                progress_bar = tqdm.tqdm(range(num_steps_per_epoch), desc="Total Steps", leave=False)

                self.unet.train()
                if self.conf.everyone.train_text_encoder:
                    self.text_encoder.train()

                for _, batch in enumerate(train_dataloader):
                    if self.command_queue.qsize() > 0:
                        command = self.command_queue.get()
                        if command == 'stop':
                            # Start training
                            self.log_queue.put("Stopping training...")
                            raise StopTrainingException("Recieved Stop Training Command.")
                        elif command == 'save':
                            # Save the model
                            self.log_queue.put("Saving Checkpoint...")
                            self.save_checkpoint()

                    b_start = time.perf_counter()
                    latents = self.vae.encode(
                        batch['pixel_values'].to(self.device, dtype=self.weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215  ## TODO: Document magic number

                    # Sample noise
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the embedding for conditioning
                    encoder_hidden_states = batch['input_ids']

                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type: {self.noise_scheduler.config.prediction_type}")

                    # Predict the noise residual and compute loss
                    with torch.autocast('cuda', enabled=self.conf.everyone.fp16):
                        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                    # backprop and update
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
                    if self.conf.everyone.train_text_encoder:
                        torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    # self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # Update EMA
                    if self.use_ema:
                        self.ema_unet.step(self.unet.parameters())

                    # perf
                    b_end = time.perf_counter()
                    seconds_per_step = b_end - b_start
                    steps_per_second = 1 / seconds_per_step
                    rank_images_per_second = int(self.conf.batchSize) * steps_per_second
                    # world_images_per_second = rank_images_per_second # * world_size
                    samples_seen = self.global_step * int(self.conf.batchSize)  # * world_size

                    # get global loss for logging
                    # torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
                    loss = loss  # / world_size

                    if self.rank == 0:
                        if self.conf.enablestats:
                            threading.Thread(target=pingGeo, args=(self.conf.stats_ip,)).start()
                        progress_bar.update(1)
                        self.global_step += 1
                        currloss = loss.detach().item()
                        if self.conf.sendloss:
                            threading.Thread(target=sendLoss, args=(self.conf.stats_ip, self.conf.secretpass, currloss,)).start()
                        logs = {
                            "train/loss": currloss,
                            "train/epoch": 1,
                            "train/step": self.global_step,
                            "train/lr": self.optimizer.state_dict()['param_groups'][0]['lr'],
                            "train/samples_seen": samples_seen,
                            "perf/rank_samples_per_second": rank_images_per_second,
                            # "perf/global_samples_per_second": world_images_per_second,
                        }
                        progress_bar.set_postfix(logs)
                        self.run.log(logs, step=self.global_step)
                        if self.global_step % 20 == 0 and self.global_step > 0:
                            first_str_to_log = "steps/s: " + str(steps_per_second) + " imgs/s: " + str(
                                rank_images_per_second) + " imgs seen: " + str(samples_seen)
                            second_str_to_log = "loss: " + str(loss.detach().item()) + " lr: " + str(
                                self.optimizer.state_dict()['param_groups'][0]['lr']) + " step: " + str(self.global_step)
                            first_opt_log = str(self.optimizer.tracker.global_progress)
                            self.log_queue.put(first_str_to_log)
                            self.log_queue.put(second_str_to_log)
                            self.log_queue.put(first_opt_log)

        except StopTrainingException as e:
            self.log_queue.put("TRAINING STOPPED")
            if self.conf.trainermode == "Relay":
                self.iethread.stop()
            pass
        except Exception as e:
            self.log_queue.put(
                f'Exception caught on rank {self.rank} at step {self.global_step}, saving checkpoint...\n{e}\n{traceback.format_exc()}')
            self._log_debugging()
            pass

        self.save_checkpoint()
        if self.conf.trainermode == "Relay":
            self.iethread.stop()
        self.log_queue.put("TRAINING FINISHED")
        exit()

    def _get_ip_online(self) -> str:
        """Since Hivemind is still implementing NAT hole punching, we get IP from online sources for now"""
        self.log_queue.put("Auto-detecting public IP")
        try:
            ip = requests.get("https://api.ipify.org/", timeout=5).text
            return ip
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            self.log_queue.put("Ipfy.org took too long, trying another domain.")
        try:
            ip = requests.get("https://ipv4.icanhazip.com/", timeout=5).text
            return ip
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            self.log_queue.put("Icanhazip.com took too long, trying another domain.")
        try:
            tmpjson = json.loads(requests.get("https://jsonip.com/", timeout=5).content)
            ip = tmpjson["ip"]
            return ip
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as err:
            self.log_queue.put("Jsonip.com took too long, ran out of alternatives.")
        raise ValueError("Could not get IP online.")

    def _log_debugging(self) -> None:
        """Useful logging function"""
        # Inform the user of host, and various versions -- useful for debugging issues.
        self.log_queue.put("HOST: " + str(socket.gethostname()))
        self.log_queue.put(("CUDA: " + str(torch.version.cuda)))
        self.log_queue.put(("TORCH: " + str(torch.__version__)))
        self.log_queue.put(("TRANSFORMERS: " + str(transformers.__version__)))
        self.log_queue.put(("DIFFUSERS: " + str(diffusers.__version__)))
        self.log_queue.put(("MODEL:" + str(self.conf.everyone.model)))
        self.log_queue.put(("FP16: " + str(self.conf.everyone.fp16)))
        self.log_queue.put(("RESOLUTION: " + str(self.conf.everyone.resolution)))
        self.log_queue.put(("DEVICE: " + str(self.device)))


def PyTorchTrainer(command_queue, log_queue):
    while True:
        command = command_queue.get()
        if command == 'start':
            # info: we had some issues while passing the conf to the
            # thread, so instead we pickle it here. Please note, that
            # this pickle could include sensitive data such as your
            # HuggingFace Token.
            with open("DO_NOT_DELETE_config.pickle", 'rb') as f:
                conf = pickle.load(f)
            # stop is gonna be done inside the function, must change this later
            trainer = DistributedTrainer(command_queue, log_queue, conf)
            trainer.train()
        elif command == 'stop':
            # kill before it even starts????
            return
