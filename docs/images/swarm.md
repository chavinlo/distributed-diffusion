# Swarm how-to

## 1. Dataset
Currently code expect dataset to be in form of image jpg,jpeg,webp,png and accompanying json file that
contains tags i.e danbooru dataset.
Currently it's hardcoded and will be used as example one for that guide.
```
image_url = f"https://crowdcloud.us-southeast-1.linodeobjects.com/crowdcloud/opendataset/v1/danbooru/{post_id}.{image_ext}"
```

Sqlite db is stored in repo as index of files, and files served from linode bucket in pairs,
there is check in code to discard non paired images.
During training, both image and json are fetched, and from json extracted only description tags and saved as txt file
alongside image in workplace directory.

## 2. Mother node
Currently it expects to have GPU and will be part of the swarm

to start run ```python server.py -m configs/sd_1-5.yaml ```
it will start server in mother mode, aka it will be able to start training without needing external server,
(just put in configuration amount of images and batch size and that will do if you need to check your local setup)
For distributed training you also need configuration in next step.

sd_1-5.yaml:
```
model: "runwayml/stable-diffusion-v1-5"
extended_chunks: 2
clip_penultimate: True
fp16: True
resolution: 512
seed: 42
train_text_encoder: True
lr: 1e-4
ucg: 0.1
use_ema: False
opt_betas_one: 0.9
opt_betas_two: 0.999
opt_epsilon: 1e-08
opt_weight_decay: 1e-2
buckets_shuffle: True
buckets_side_min: 256
buckets_side_max: 512
```


## 3. Relay mode
Next step is to set up relay configuration in UI,
that will allow your node to serve config endpoint to peers - from wich they will be able to grab config,
for that purprose I.E port are used.
So if you behind NAT it makes sense to portforward port 80 to your internal I.E port,
TCP ports are used for DHT connections so as well needed to be portforwarded,
and your public ip will be used as DHT server, so that ip (or domain name ) you can spread 
among peers so they would be able to join swarm.

If you are NOT behind NAT, put matching pairs of ports here, so advertised and listened ports are matched 
and that will do the trick. 

## 4. Training
After configuration and start of training, there will be main events after 1200 seconds i.e 20 minutes
```
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
```
Optimizer and Averager events should start and peers will exchange trained weights - 
that should lead to decrease in loss.

Due to geography and network speeds, and size of weights that part may be tricky so keep an eye on that.

## 5. Wait
