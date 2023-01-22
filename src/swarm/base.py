import requests
import hivemind

def get_config():
    r = requests.get("http://127.0.0.1:5000/internal/config")
    r = r.json()
    final_config = r

    max_batch_size = int((float(r['trainer_config']['local']['max-vram']) - float(r['trainer_config']['global']['minimun_vram'])) / float(r['trainer_config']['global']['vram_per_batch']))

    final_config['trainer_config']['local']['final_batch_size'] = max_batch_size

    return final_config
    
def make_dataloader(config):
    #TODO: make the dataloader
    """
    So basically we are going to do something similar to what we did previously.
    Using Lopho's preprocessor, we will convert the images to latents, and load them
    randomly
    """
    return ""

def wrap_opt(optimizer, config):

    #TODO: support other dht access types
    if config['swarm_config']['dht']['type'] == "direct":
        init_peers = config['swarm_config']['dht']['entrypoint']

    dht = hivemind.DHT(
        start=True,
        daemon=True,
        initial_peers = init_peers
    )

    # TODO: offload optimizer gradients and such to achieve less vram usage
    hm_opt = hivemind.Optimizer(
        dht=dht,
        run_id="change_this_later",
        target_batch_size=10000,  
        optimizer=optimizer,            
        use_local_updates=True,   
        matchmaking_time=3.0,     
        averaging_timeout=10.0,   
        verbose=True              
    )

    return hm_opt
