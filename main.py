import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# from module import LitClassifier
import neptune.new as neptune
from module.utils.data_module import fMRIDataModule
from module.utils.parser import str2bool
from module.pl_classifier import LitClassifier


def cli_main():

    # ------------ args -------------
    parser = ArgumentParser(add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", default=2024, type=int, help="random seeds. recommend aligning this argument with data split number to control randomness")
    parser.add_argument("--dataset_name", type=str, choices=["S1200", "ABCD", "UKB", "Dummy", "olfactory", "taowu", "ABIDE"], default="ABIDE")
    parser.add_argument("--subject_csv_path", type=str, default="D:\\Project\\Parkinson\\SwiFT-main\\project\\data\\ABIDE\\Phenotypic_V1_0b_preprocessed1.csv")
    parser.add_argument("--downstream_task", type=str, default="sex", help="downstream task")
    parser.add_argument("--downstream_task_type", type=str, default="classification", help="select either classification or regression according to your downstream task")
    parser.add_argument("--classifier_module", default="ABIDE", type=str, help="A name of lightning classifier module (outdated argument)")
    parser.add_argument("--loggername", default="tensorboard", type=str, help="A name of logger")
    parser.add_argument("--project_name", default="ABIDE_ROI", type=str, help="A name of project (Neptune)")
    parser.add_argument("--resume_ckpt_path", type=str, help="A path to previous checkpoint. Use when you want to continue the training from the previous checkpoints")
    # parser.add_argument("--load_model_path", type=str, default="./output/taowu_baseline_corrected_none_agg/checkpt-epoch=11-valid_loss=2.73.ckpt", help="A path to the pre-trained model weight file (.pth)")
    parser.add_argument("--load_model_path", type=str, default="./output/ABIDE_0/checkpt-epoch=15-valid_loss=2.22.ckpt", help="A path to the pre-trained model weight file (.pth)")
    # parser.add_argument("--load_model_path", type=str, help="A path to the pre-trained model weight file (.pth)")
    parser.add_argument("--test_only", default=False, help="specify when you want to test the checkpoints (model weights)")
    parser.add_argument("--test_ckpt_path", type=str, help="A path to the previous checkpoint that intends to evaluate (--test_only should be True)")
    parser.add_argument("--freeze_feature_extractor", default=False, help="Whether to freeze the feature extractor (for evaluating the pre-trained weight)")
    temp_args, _ = parser.parse_known_args()

    # Set classifier
    Classifier = LitClassifier
    
    # Set dataset
    Dataset = fMRIDataModule

    
    # add two additional arguments
    parser = Classifier.add_model_specific_args(parser)
    parser = Dataset.add_data_specific_args(parser)

    _, _ = parser.parse_known_args()  # This command blocks the help message of Trainer class.
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    #override parameters
    max_epochs = args.max_epochs
    num_nodes = args.num_nodes
    devices = args.devices
    project_name = args.project_name
    image_path = args.image_path

    if temp_args.resume_ckpt_path is not None:
        # resume previous experiment
        from module.utils.neptune_utils import get_prev_args
        args = get_prev_args(args.resume_ckpt_path, args)
        exp_id = args.id
        # override max_epochs if you hope to prolong the training
        args.project_name = project_name
        args.max_epochs = max_epochs
        args.num_nodes = num_nodes
        args.devices = devices
        args.image_path = image_path       
    else:
        exp_id = None
    
    setattr(args, "default_root_dir", f"output/{args.project_name}")

    # ------------ data -------------
    data_module = Dataset(**vars(args))
    pl.seed_everything(args.seed)

    # ------------ logger -------------
    if args.loggername == "tensorboard":
        # logger = True  # tensor board is a default logger of Trainer class
        dirpath = args.default_root_dir
        logger = TensorBoardLogger(dirpath)
    elif args.loggername == "neptune":
        API_KEY = os.environ.get("NEPTUNE_API_TOKEN")
        # project_name should be "WORKSPACE_NAME/PROJECT_NAME"
        run = neptune.init(api_token=API_KEY, project=args.project_name, capture_stdout=False, capture_stderr=False, capture_hardware_metrics=False, run=exp_id)
        
        if exp_id == None:
            setattr(args, "id", run.fetch()['sys']['id'])

        logger = NeptuneLogger(run=run, log_model_checkpoints=False)
        dirpath = os.path.join(args.default_root_dir, logger.version)
    else:
        raise Exception("Wrong logger name.")

    # ------------ callbacks -------------
    # callback for pretraining task
    if args.pretraining:
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_loss",
            filename="checkpt-{epoch:02d}-{valid_loss:.2f}",
            save_last=True,
            mode="min",
        )
    # callback for classification task
    elif args.downstream_task == "sex" or args.downstream_task == "Dummy" or args.downstream_task_type == "classification":
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_acc",
            filename="checkpt-{epoch:02d}-{valid_acc:.2f}",
            save_last=True,
            mode="max",
        )
        test_checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="test_acc",  # 监控测试集准确率
            filename="test-checkpt-{epoch:02d}-{test_acc:.2f}",
            save_last=False,  # 可根据需要设置是否保存最后一个 epoch 的模型
            mode="max",  # 以最高测试集准确率保存
        )
    # callback for regression task
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="valid_mse",
            filename="checkpt-{epoch:02d}-{valid_mse:.2f}",
            save_last=True,
            mode="min",
        )
        test_checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            monitor="test_acc",  # 监控测试集准确率
            filename="test-checkpt-{epoch:02d}-{test_acc:.2f}",
            save_last=False,  # 可根据需要设置是否保存最后一个 epoch 的模型
            mode="max",  # 以最高测试集准确率保存
        )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]

    # ------------ trainer -------------
    if args.grad_clip:
        print('using gradient clipping')
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm",
            track_grad_norm=-1,
            #################################
            accelerator="gpu",  # 设置 GPU 加速
            devices=1,
            num_sanity_val_steps=0
        )
    else:
        print('not using gradient clipping')
        print(args)
        trainer = pl.Trainer.from_argparse_args(
            args,
            logger=logger,
            check_val_every_n_epoch=1,
            #val_check_interval=100 if not args.scalability_check else None,
            callbacks=callbacks,
            #################################
            accelerator="gpu",  # 设置 GPU 加速
            devices=1,
            num_sanity_val_steps=0
        )

    # ------------ model -------------
    model = Classifier(data_module = data_module, **vars(args)) 

    if args.load_model_path is not None:
        print(f'loading model from {args.load_model_path}')
        path = args.load_model_path
        ckpt = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            if 'model.' in k: #transformer-related layers
                new_state_dict[k.removeprefix("model.")] = v
        model.model.load_state_dict(new_state_dict)

    if args.freeze_feature_extractor:
        # layers are frozen by using eval()
        model.model.eval()
        # freeze params
        for name, param in model.model.named_parameters():
            if 'output_head' not in name: # unfreeze only output head
                param.requires_grad = False
                print(f'freezing layer {name}')

    # ------------ run -------------
    if args.test_only:
        trainer.test(model, datamodule=data_module, ckpt_path=args.test_ckpt_path) # dataloaders=data_module
    else:
        if args.resume_ckpt_path is None:
            # New run
            trainer.fit(model, datamodule=data_module)
        else:
            # Resume existing run
            trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt_path)

        trainer.test(model, dataloaders=data_module)


if __name__ == "__main__":
    cli_main()
