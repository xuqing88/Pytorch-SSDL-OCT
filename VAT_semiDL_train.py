import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import copy
from VAT.vat import VAT
import argparse
import statistics

def eval_model(model, criterion, dataloader_test, dataset_size_test, use_gpu):
    print("Start Model Evaluation.............................")
    since = time.time()
    loss_test = 0
    acc_test = 0
    test_batches = len(dataloader_test)

    for i, data in enumerate(dataloader_test):
        model.train(False)
        model.eval()
        inputs, labels = data

        with torch.no_grad():
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

        # print(loss.data.item())
            loss_test += loss.data.item()  # loss.data[0]
            acc_test += torch.sum(preds == labels.data).item()

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    avg_loss = float(loss_test) / dataset_size_test
    avg_acc = float(acc_test) / dataset_size_test

    elapsed_time = time.time() - since
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("test loss={:.4f} | test acc={:.4f}".format(avg_loss,avg_acc))
    print()
    torch.cuda.empty_cache()
    return avg_acc


def train_model_vat(model, reg_fn, criterion, optimizer, num_epochs, dataloader_train, dataloader_val,dataloader_unlabel, use_gpu):
    print("Start  VGG_VAT training............................")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    avg_loss = 0
    avg_loss_VAT = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0

    train_batches = len(dataloader_train)
    val_batches = len(dataloader_val)

    for epoch in range(num_epochs):
        # print("Epoch {}/{}".format(epoch, num_epochs))
        # print('-' * 10)

        loss_train = 0
        loss_train_VAT = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0

        model.train(True)

        for i, data in enumerate(dataloader_train):
            # if i % 2 == 0:
            #     print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)

            inputs, labels = data
            unlabel_inputs, _ = dataloader_unlabel.__iter__().next()

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                unlabel_inputs = Variable(unlabel_inputs.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
                unlabel_inputs = Variable(unlabel_inputs)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)

            with torch.no_grad():
                # vgg.eval()
                unlabel_logit = model(unlabel_inputs)
                # torch.cuda.empty_cache()
                # vgg.train_org()

            vat = reg_fn(unlabel_inputs, unlabel_logit)

            loss = criterion(outputs, labels) + 2.0 * vat

            loss.backward()
            optimizer.step()

            loss_train_VAT += vat.data.item()
            loss_train += loss.data.item()  # loss.data[0]
            acc_train += torch.sum(preds == labels.data).item()

            del inputs, labels, outputs, preds, unlabel_inputs,unlabel_logit
            torch.cuda.empty_cache()

        avg_loss = float(loss_train) / (train_batches *8)
        avg_acc = float(acc_train) / (train_batches *8)
        avg_loss_VAT = float(loss_train_VAT) / (train_batches *8)

        model.train(False)
        model.eval()

        for i, data in enumerate(dataloader_val):
            # if i % 2 == 0:
            #     print("\rValidation batch {}/{}".format(i, val_batches), end='', flush=True)

            inputs, labels = data

            with torch.no_grad():
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

            # optimizer.zero_grad()

                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                loss_val += loss.data.item()  # loss.data[0]
                acc_val += torch.sum(preds == labels.data).item()

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

        avg_loss_val = float(loss_val) / (val_batches * 8)
        avg_acc_val = float(acc_val) / (val_batches * 8)

        print("Epoch-{} | train loss={:.4f} | train acc={:.4f} | val loss={:.4f} | val acc={:.4f} | vat_loss={:.4f} "
              .format(epoch,avg_loss,avg_acc,avg_loss_val,avg_acc_val,avg_loss_VAT))

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())

    elapsed_time = time.time() - since
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best val acc={:.4f}".format(best_acc))
    print()
    model.load_state_dict(best_model_wts)
    return model


def main(args):

    use_gpu = torch.cuda.is_available()
    if use_gpu: print("Using CUDA")

    epochs = args.epochs
    dataset = args.dataset

    if dataset == 'BOE':
        print(" Loading BOE data set")
        data_dir = './dataset/Semi_BOEdata'
        unlabeled_labeled_ratio = 4
        TRAIN= 'train'
    elif dataset == 'OCT':
        print(" Loading OCT data set")
        data_dir = './dataset/OCT2017'
        unlabeled_labeled_ratio = 2
        TRAIN = 'train_semi'

    VAL, TEST, UNLABEL = 'val','test','unlabel'
    batch_size = 8
    iterations = 5
    lr= 0.001

    #====================== Data Loading ==================================================
    # VGG-16 Takes 224x224 images as input, so we resize all of them
    data_transforms = {
        TRAIN: transforms.Compose([
            # Data augmentation is a good practice for the train_org set
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally.
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
        VAL: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
        TEST: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
        UNLABEL: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),transform=data_transforms[x])
        for x in [TRAIN, VAL, TEST, UNLABEL]
    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
            shuffle=True, num_workers=0,
        )
        for x in [TRAIN, VAL, TEST]
    }

    dataloaders[UNLABEL] = torch.utils.data.DataLoader(image_datasets[UNLABEL],
                                                       batch_size=batch_size*unlabeled_labeled_ratio,
                                                       shuffle=True, num_workers=0)

    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST, UNLABEL]}

    for x in [TRAIN, VAL, TEST, UNLABEL]: print("Loaded {} images under {}".format(dataset_sizes[x], x))

    class_names = image_datasets[TRAIN].classes
    print("Classes: ",image_datasets[TRAIN].classes)

    # Get a batch of training data and display
    print('dataloaders[TRAIN_u].batch_size = ', dataloaders[UNLABEL].batch_size)

    #====================== Model Training =======================================================
    test_acc = []
    saved_model_name = './result/' + dataset+'_vgg_vat_best.pt'
    best_acc = 0.0
    for iter in range(1,iterations+1):

        # create model and load the pretrained weights
        vgg16 = models.vgg16_bn(pretrained=True)

        # Newly created modules have require_grad=True by default
        num_features = vgg16.classifier[6].in_features
        features = list(vgg16.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 4 outputs
        vgg16.classifier = nn.Sequential(*features) # Replace the model classifier
        # print(vgg16)
        if use_gpu: vgg16.cuda()  # .cuda() will move everything to the GPU side
        model = vgg16

        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=lr, momentum=0.9)

        reg_fn = VAT(model)
        model = train_model_vat(model, reg_fn, criterion, optimizer_ft, epochs,
                                dataloaders[TRAIN],dataloaders[VAL],dataloaders[UNLABEL], use_gpu)

        acc = eval_model(model, criterion, dataloaders[TEST], dataset_sizes[TEST], use_gpu)
        if acc > best_acc:
            print("Save model")
            print()
            best_acc = acc
            torch.save(model.state_dict(),saved_model_name)
        test_acc.append(acc)

    print('test_acc=',  test_acc)
    test_acc_avg = sum(test_acc) / len(test_acc)
    test_acc_var = statistics.stdev(test_acc)
    print("Average test acc: %.3f" % (test_acc_avg), '| Variance test: %.3f' % (test_acc_var))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset',
                        help='dataset, choose from [BOE,OCT]',
                        type=str,
                        choices=['BOE','OCT'],
                        default='OCT')

    parser.add_argument('-e', '--epochs',
                        help='training epochs',
                        type=int,
                        default = 30)

    args = parser.parse_args()
    main(args)