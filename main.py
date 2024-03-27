import os
import torch
from models.resnets import _all_models 
from utils.online_eval import delayed_eval_online
from utils.dataloader import get_dataloader, get_cp
from tta_methods import _all_methods
from utils.config import parse_option

def main(config):
    
    if config.run.use_wandb:
        import wandb
        wandb.init(project="tta", config=config)
    print(config)
    
    # Initializing the model
    model = _all_models[config.model.arch](pretrained=True, progress=True).to(config.run.device)

    # Putting the model into a wrapper
    tta_method = _all_methods[config.model.method](model, config)

    # Getting the corrupted data that we need to evaluate the model at
    all_corruptions = get_cp(config)
    print(all_corruptions)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)
    
    for corruption in all_corruptions: 
         
        config.evaluation.corruption = corruption
        print("loading "+ corruption+" corruption ...")
        if config.evaluation.single_model:
            print('Performing single model evaluation')
        corrupted_dataloader = get_dataloader(config)

        # Evaluating the model
        adjusted_acc, tta_method = delayed_eval_online(tta_method, 
                                                corrupted_dataloader, 
                                                eta=config.evaluation.eta, 
                                                device=config.run.device, 
                                                dataset_name=config.dataset.name, 
                                                single_model=config.evaluation.single_model)

        # logger.info(args.corruption)
        print(f"Under shift type {config.evaluation.corruption} After {config.model.method} Top-1 Adjusted Accuracy: {adjusted_acc*100:.5f}")
        print(f"Under shift type {config.evaluation.corruption} After {config.model.method} Top-1 Error Rate: {100-adjusted_acc*100:.5f}")
        print(f'Finished {config.model.method} on {config.evaluation.corruption} with level {config.evaluation.level}, Adjusted Error Rate: {100-adjusted_acc*100:.5f}, eta: {config.evaluation.eta}')
        
        with open(os.path.join(config.output_dir, '{}.txt'.format(config.evaluation.corruption)), 'w') as f:
            f.write('eta {}'.format(config.evaluation.eta))
            f.write('\n')
            f.write('error_rate {}'.format(100-100*adjusted_acc))
    
    if config.run.use_wandb:
        wandb.log({'adjusted_acc': adjusted_acc, 'error_rate': 100-100*adjusted_acc, 'eta': config.evaluation.eta})
    return

if __name__ == '__main__':
    # args = get_args()
    args, opts, config = parse_option()
    torch.manual_seed(config.run.seed)
    main(config)