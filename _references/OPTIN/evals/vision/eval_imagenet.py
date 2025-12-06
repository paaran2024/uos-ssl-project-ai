# from timm.utils import accuracy, AverageMeter
from tqdm.auto import tqdm
import torch, random
import torch.nn as nn
import evaluate
from utils._hooks import ModelHooking
from data.scripts.glue import target_dev_metric
import timm
import tome

@torch.no_grad()
def eval_imagenet_acc(args, model, val_dataloader, task_name, pruningParams=None):
    
    
    modPatch = [0]
    for i in range(1, len(pruningParams["patch_mask"])):
        modPatch.append(pruningParams["patch_mask"][i-1].shape[0] - pruningParams["patch_mask"][i].shape[0])
    
    
    
    name = (args.model_name).replace("-", "_")
    
    model = timm.create_model(name, pretrained=True).cuda()
    
    #* Implement ToME patch to support token merging @ ToME implementation: Bolya et al. 2023
    #* Note: As stated in the main paper, any other token pruning/merging implementation can be used as well, simply apply our searched reduction strategy.
    
    if sum(modPatch) > 0:
        print(modPatch, sum(modPatch))
        tome.patch.timm(model)
        model.r = modPatch

    metric = evaluate.load("accuracy")

    print([i.sum() for i in pruningParams["patch_mask"]])
    
    if pruningParams != None:
        prunedModelObject = ModelHooking(args=args, model=model, maskProps=None, evalState=pruningParams)
        model = prunedModelObject.return_model()
        print("returned model")
        mask = pruningParams["head_mask"].cuda()
    else:
        mask = torch.ones((args.model_config["num_hidden_layers"], args.model_config["num_attention_heads"])).cuda()
    
    model.eval()
    model.cuda()
    print(mask.sum(-1))
    
    for idx, (batch_x, batch_y) in tqdm(enumerate(val_dataloader)):
        mappingBatch = {}
        mappingBatch["pixel_values"] = batch_x.to("cuda", non_blocking=True)
        mappingBatch["labels"] = batch_y.to("cuda", non_blocking=True)
        
        outputs = model(mappingBatch["pixel_values"])
        predictions = outputs.argmax(dim=-1)
        metric.add_batch(
            predictions=predictions,
            references=mappingBatch["labels"],
        )

    eval_results = metric.compute()
    print(eval_results)
    target_metric = target_dev_metric(task_name)
    accuracy = eval_results[target_metric]
    return accuracy