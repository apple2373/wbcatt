import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from tqdm.auto import tqdm
from io import BytesIO
import numpy as np
import random

from att_dataset import AttDataset
from attribute_predictor import AttributePredictor

att_names = ['cell_size', 'cell_shape', 'nucleus_shape', 'nuclear_cytoplasmic_ratio', 'chromatin_density',
             'cytoplasm_vacuole', 'cytoplasm_texture', 'cytoplasm_colour', 'granule_type', 'granule_colour', 'granularity']


def save_args(savedir, args, name="args.json"):
    # save args as "args.json" in the savedir
    path = os.path.join(savedir, name)
    with open(path, 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    print("args saved as %s" % path)


def save_json(dict, path):
    with open(path, 'w') as f:
        json.dump(dict, f, sort_keys=True, indent=4)
        print("log saved at %s" % path)


def save_checkpoint(path, model, key="model"):
    # save model state dict
    checkpoint = {}
    checkpoint[key] = model.state_dict()
    torch.save(checkpoint, path)
    print("checkpoint saved at", path)


def make_deterministic(seed):
    # https://github.com/pytorch/pytorch/issues/7068#issuecomment-487907668
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_metrics(true_labels, predicted_probs):
    predicted_probs = np.array(predicted_probs)
    true_labels = np.array(true_labels)
    predicted_labels = np.argmax(predicted_probs, axis=1)
    metrics = {
        "acc": accuracy_score(true_labels, predicted_labels),
        "f1_macro": f1_score(true_labels, predicted_labels, average='macro'),
        "pre_macro": precision_score(true_labels, predicted_labels, average='macro'),
        "rec_macro": recall_score(true_labels, predicted_labels, average='macro'),
    }
    return metrics


def get_transforms(split, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def get_image_encoder(name, pretrained=True):
    weights = "DEFAULT" if pretrained else None
    model = getattr(torchvision.models, name)(weights=weights)
    if name.startswith('vgg'):
        model.classifier[6] = nn.Identity()
    elif name.startswith('resnet'):
        model.fc = nn.Identity()
    elif name.startswith('vit'):
        model.heads = nn.Identity()
    elif name.startswith('convnext'):
        model.classifier[-1] = nn.Identity()
    else:
        raise ValueError(f"Unsupported image encoder: {name}")
    # Infer the output size of the image encoder
    with torch.inference_mode():
        out = model(torch.randn(5, 3, 224, 224))
    assert out.dim() == 2
    assert out.size(0) == 5
    image_encoder_output_dim = out.size(1)
    return model, image_encoder_output_dim


# Save predictions as CSV files
def save_predictions_to_csv(predictions, log_dir, filename, dataloader_val):
    decoded_predictions = []
    for j, preds in enumerate(predictions):
        attribute = dataloader_val.dataset.attribute_columns[j]
        encoder = dataloader_val.dataset.attribute_encoders[attribute]
        # Create a reverse mapping
        decoder = {v: k for k, v in encoder.items()}
        decoded_preds = [decoder[np.argmax(p)] for p in preds]
        decoded_predictions.append(decoded_preds)
    data = {"image_path": dataloader_val.dataset.df['path']}
    data.update({attribute: preds for attribute, preds in zip(
        dataloader_val.dataset.attribute_columns, decoded_predictions)})
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(log_dir, filename), index=False)


def evaluate(model, dataloader):
    model.eval()
    num_attributes = len(dataloader.dataset.attribute_columns)
    all_predictions = [[] for _ in range(num_attributes)]
    all_probabilities = [[] for _ in range(num_attributes)]
    all_true_labels = [[] for _ in range(num_attributes)]
    all_image_paths = []
    with torch.inference_mode():
        for i, data in enumerate(dataloader):
            images, attributes = data['image'], data['attributes']
            images, attributes = images.cuda(), attributes.cuda()
            model_outputs = model(images)
            for j, model_output in enumerate(model_outputs):
                model_output = torch.softmax(model_output, dim=1)
                all_probabilities[j].extend(model_output.cpu().tolist())
                all_predictions[j].extend(torch.argmax(
                    model_output, dim=1).cpu().tolist())
                all_true_labels[j].extend(attributes[:, j].cpu().tolist())
            all_image_paths.extend(data['img_path'])
    # Call calculate_metrics once with a fake data to get the list of available metrics
    _, initial_metrics = next(iter(dataloader.dataset.attribute_columns)), calculate_metrics(
        [0, 1, 1], [[0.98, 0.02], [0.02, 0.98], [0.60, 0.40]])
    overall_metrics = {metric: 0.0 for metric in initial_metrics.keys()}
    per_attribute_metrics = {column: {}
                             for column in dataloader.dataset.attribute_columns}
    for j, attribute in enumerate(dataloader.dataset.attribute_columns):
        metrics = calculate_metrics(all_true_labels[j], all_probabilities[j])
        for metric in overall_metrics.keys():
            overall_metrics[metric] += metrics[metric]
        per_attribute_metrics[attribute] = metrics
    for metric in overall_metrics.keys():
        overall_metrics[metric] /= num_attributes
    return {
        "overall_metrics": overall_metrics,
        "per_attribute_metrics": per_attribute_metrics,
        "all_image_paths": all_image_paths,
        "all_probabilities": all_probabilities,
        "all_predictions": all_predictions,
    }


def main(args):
    print(args)

    # set seed
    make_deterministic(args.seed)

    # setup the directory to save the experiment log and trained models
    log = {}
    log_dir = args.logdir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_path = os.path.join(log_dir, "log.json")
    save_json(log, log_path)
    save_args(log_dir, args)

    # setup image encoder
    image_encoder, image_encoder_output_dim = get_image_encoder(args.backbone)

    # setup dataset and dataloader
    dataset_train = AttDataset(args.train, att_names, image_dir=args.image_dir,
                               transform=get_transforms('train'), multiply=args.epoch_multiply)
    dataset_val = AttDataset(args.val, att_names, image_dir=args.image_dir, transform=get_transforms(
        'test'), attribute_encoders=dataset_train.attribute_encoders)
    dataset_test = AttDataset(args.test, att_names, image_dir=args.image_dir,
                              transform=get_transforms('test'), attribute_encoders=dataset_train.attribute_encoders)
    attribute_sizes = [len(encoder)
                       for encoder in dataset_train.attribute_encoders.values()]
    # testing attributes should be the same set or at least subset of training
    # if testing set contains attribute value  that has not appeared in  training set, things will be broken.
    for column in dataset_val.attribute_columns + dataset_test.attribute_columns:
        for value in sorted(dataset_val.df[column].unique()):
            assert value in dataset_train.attribute_encoders[
                column], f"Attribute value '{value}' in column '{column}' not found in training dataset"
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, persistent_workers=(args.workers > 0), pin_memory=True, drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, persistent_workers=(args.workers > 0), pin_memory=True, drop_last=False)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, persistent_workers=(args.workers > 0), pin_memory=True, drop_last=False)

    model = AttributePredictor(
        attribute_sizes, image_encoder_output_dim, image_encoder)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.decay)

    model.cuda()
    model.train()
    best_val_metric = 0

    training_logs = []
    best_epoch = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        num_processed_samples = 0
        with tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch") as t:
            for i, data in enumerate(t):
                images, attribute_targets = data['image'], data['attributes']
                images, attribute_targets = images.cuda(), attribute_targets.cuda()
                optimizer.zero_grad()
                attribute_outputs = model(images)
                loss = 0
                for idx, (output, target) in enumerate(zip(attribute_outputs, attribute_targets.t())):
                    loss += criterion(output, target)
                # average loss over all attributes. just rescaling.
                loss = loss / len(attribute_outputs)
                loss.backward()
                optimizer.step()
                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                num_processed_samples += batch_size
                t.set_postfix(train_loss=(
                    running_loss / num_processed_samples))
        training_loss = running_loss / num_processed_samples
        evaluation_results = evaluate(model, dataloader_val)
        overall_metrics, per_attribute_metrics, _, _, _ = evaluation_results.values()
        print(f"Epoch {epoch + 1}, Overall Metrics on Val: " +
              ', '.join([f"{k}: {(100*v):.2f}" for k, v in overall_metrics.items()]))
        val_metric = overall_metrics[args.eval_metric]
        # Save the best model based on f1_macro
        if val_metric > best_val_metric:
            best_epoch = epoch
            best_val_metric = val_metric
            # https://discuss.pytorch.org/t/how-to-make-a-copy-of-a-gpu-model-on-the-cpu/90955/4
            # Save the model to memory from GPU
            model_data_in_memory = BytesIO()
            torch.save(model.state_dict(),
                       model_data_in_memory, pickle_protocol=-1)
            model_data_in_memory.seek(0)

        # Log training progress
        training_log = {
            'epoch': epoch + 1,
            'training_loss': training_loss,
            'evaluation': evaluation_results,
        }
        training_logs.append(training_log)
        log["trainining"] = training_logs
        save_json(log, log_path)

    print("Best Epoch", best_epoch)
    # Evaluate the bestval model on the test set
    # Load the model from memory to CPU and then send to GPU
    model_in_cpu = torch.load(model_data_in_memory, map_location="cpu")
    model_data_in_memory.close()
    model.load_state_dict(model_in_cpu)
    model.cuda()
    evaluation_results = evaluate(model, dataloader_test)
    overall_metrics_best = evaluation_results['overall_metrics']
    best_decoded_predictions = evaluation_results['all_probabilities']
    print(f"Epoch {epoch + 1}, BestValModel Overall Metrics on Test: " +
          ', '.join([f"{k}: {(100*v):.2f}" for k, v in overall_metrics_best.items()]))
    save_predictions_to_csv(best_decoded_predictions, log_dir,
                            'bestval_epoch_predictions.csv', dataloader_test)
    log["bestval"] = evaluation_results
    save_json(log, log_path)
    # Save the best model
    best_model_path = os.path.join(log_dir, 'best_model.pth')
    save_checkpoint(best_model_path, model, key="model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train an attribute prediction model.')
    parser.add_argument('--train', default='./pbc_attr_v1_train.csv',
                        help='Path to the training CSV file')
    parser.add_argument('--val', default='./pbc_attr_v1_val.csv',
                        help='Path to the validation CSV file')
    parser.add_argument(
        '--test', default='./pbc_attr_v1_test.csv', help='Path to the test CSV file')
    parser.add_argument('--image_dir', default='./data/PBC/',
                        help='Root directory containing image files')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--epoch_multiply', type=int, default=1,
                        help='Number of times to repeat the dataset in each epoch')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--decay', type=float,
                        default=0.01, help="weight decay")
    parser.add_argument('--batch_size', type=int,
                        default=128, help='Batch size')
    parser.add_argument('--eval_metric', default='f1_macro',
                        help='Evaluation metric for model selection')
    parser.add_argument('--backbone', default='resnet50',
                        help='Choice of image encoder', choices=['vgg16', 'resnet50', 'convnext_tiny', 'vit_b_16'])
    parser.add_argument('--logdir', default='./log',
                        help='directory to save experiment logs')
    parser.add_argument('--savemodel', action='store_true',
                        help='Save the model if specified')
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed.')
    parser.add_argument('--workers', type=int, default=8,
                        help="workers for torch.utils.data.DataLoader")
    args = parser.parse_args()
    main(args)
