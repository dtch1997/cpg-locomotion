from functools import partial
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from ray import tune
import torchvision


DATA_DIR = os.path.join(os.getcwd(), "./blind_walking/examples/data/heightmap.npy")
single_data_shape = (16, 12)  # x-axis 12, y-axis 16


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class LinearAE(torch.nn.Module):
    def __init__(self, input_size=140, code_size=32):
        super().__init__()
        # encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, code_size),
        )
        # decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(code_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_size),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvAE(torch.nn.Module):
    def __init__(self, code_size=32):
        super().__init__()
        # encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, stride=2 ,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3, stride=1, padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=1, padding=0),
            torch.nn.ReLU(),

            torch.nn.Flatten(start_dim=1),

            torch.nn.Linear(2 * 4 * 32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, code_size),
        )
        # decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(code_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2 * 4 * 32),
            torch.nn.ReLU(),

            torch.nn.Unflatten(dim=1, unflattened_size=(32, 4, 2)),

            torch.nn.ConvTranspose2d(32, 16, 3, stride=1, output_padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 3, stride=1, output_padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_data():
    # load dataset
    dataset_np = np.load(DATA_DIR)
    single_data_size = len(dataset_np[0])
    assert np.prod(single_data_shape) == single_data_size

    # shuffle dataset
    np.random.seed(12)
    np.random.shuffle(dataset_np)

    # # offset dataset for more enriching features
    # dataset_np = -(dataset_np - dataset_np.max(axis=1)[:, None])

    # split into train, test, validation sets
    train_size = int(0.8 * len(dataset_np))
    val_size = int(0.1 * len(dataset_np))
    train_dataset_np = dataset_np[:train_size, :]
    val_dataset_np = dataset_np[train_size : train_size + val_size, :]
    test_dataset_np = dataset_np[train_size + val_size :, :]

    # add noise to dataset
    # train_dataset_noisy_np = train_dataset_np + np.random.normal(0, 0.01, train_dataset_np.shape)
    # val_dataset_noisy_np = val_dataset_np + np.random.normal(0, 0.01, val_dataset_np.shape)
    # test_dataset_noisy_np = test_dataset_np + np.random.normal(0, 0.01, test_dataset_np.shape)
    train_dataset_noisy_np = train_dataset_np
    val_dataset_noisy_np = val_dataset_np
    test_dataset_noisy_np = test_dataset_np

    # make tensor dataset
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_dataset_np), torch.Tensor(train_dataset_noisy_np))
    val_dataset = torch.utils.data.TensorDataset(torch.Tensor(val_dataset_np), torch.Tensor(val_dataset_noisy_np))
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_dataset_np), torch.Tensor(test_dataset_noisy_np))

    return (train_dataset, val_dataset, test_dataset), (single_data_size, single_data_shape)


def train_model(config, checkpoint_dir=None, tobetune=True, model_type="linear"):
    # load datasets
    dataset_and_info = load_data()
    train_dataset, val_dataset, _ = dataset_and_info[0]
    single_data_size, single_data_shape = dataset_and_info[1]
    # datasets loader used for training and validation
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )

    # model initialisation
    if model_type == "linear":
        model = LinearAE(input_size=single_data_size, code_size=config["code_size"])
    elif model_type == "conv":
        model = ConvAE(code_size=config["code_size"])
    # use gpu if available
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
    # loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    # load checkpoint if provided
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    min_val_loss = 100  # artbitrary high number
    val_grace = 4  # number of grace times for training to continue
    epochs = 100000
    for epoch in range(epochs):
        train_loss = 0
        for (batch_data_truth, batch_data) in train_loader:
            # load mini-batch data to the active device
            if model_type == "linear":
                batch_data = batch_data.view(-1, single_data_size).to(device)
                batch_data_truth = batch_data_truth.view(-1, single_data_size).to(device)
            elif model_type == "conv":
                batch_data = batch_data.view(-1, 1, *single_data_shape).to(device)
                batch_data_truth = batch_data_truth.view(-1, 1, *single_data_shape).to(device)
            # reset the gradients back to zero
            optimizer.zero_grad()
            # compute reconstructions
            outputs = model(batch_data)
            # compute loss
            loss = loss_function(outputs, batch_data_truth)
            # compute accumulated gradients
            loss.backward()
            # perform parameter update based on current gradients
            optimizer.step()
            # add the mini-batch training loss to epoch loss
            train_loss += loss.item()

        # compute and display the epoch training loss
        train_loss = train_loss / len(train_loader)
        if not tobetune and epoch % 100 == 0:
            print("epoch : {}/{}, loss = {:.6f}, min val loss = {:.6f}".format(epoch + 1, epochs, train_loss, min_val_loss))

        # validation
        val_loss = 0
        for (batch_data_truth, batch_data) in val_loader:
            with torch.no_grad():
                if model_type == "linear":
                    batch_data = batch_data.view(-1, single_data_size).to(device)
                    batch_data_truth = batch_data_truth.view(-1, single_data_size).to(device)
                elif model_type == "conv":
                    batch_data = batch_data.view(-1, 1, *single_data_shape).to(device)
                    batch_data_truth = batch_data_truth.view(-1, 1, *single_data_shape).to(device)
                outputs = model(batch_data)
                loss = loss_function(outputs, batch_data_truth)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        # early stopping
        if epoch % 100 == 0 and val_loss > min_val_loss * 1.1:
            val_grace -= 1
            if val_grace < 0:
                break
        min_val_loss = min(val_loss, min_val_loss)

        # report to ray tune
        if tobetune:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
            tune.report(loss=val_loss)

    if not tobetune:
        # save pytorch model
        torch.save(
            (model.state_dict(), optimizer.state_dict()),
            f"./autoenc_results/model_bs{config['batch_size']}_cs{config['code_size']}_lr{config['lr']}",
        )
    print("Finished Training")


def test_model(model, device="cpu", model_type="linear"):
    # load datasets
    dataset_and_info = load_data()
    _, _, test_dataset = dataset_and_info[0]
    single_data_size, single_data_shape = dataset_and_info[1]
    # datasets loader used for testing
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=False,
    )
    # loss function
    loss_function = torch.nn.MSELoss()
    # test
    test_loss = 0
    for (batch_data_truth, batch_data) in test_loader:
        with torch.no_grad():
            if model_type == "linear":
                batch_data = batch_data.view(-1, single_data_size).to(device)
                batch_data_truth = batch_data_truth.view(-1, single_data_size).to(device)
            elif model_type == "conv":
                batch_data = batch_data.view(-1, 1, *single_data_shape).to(device)
                batch_data_truth = batch_data_truth.view(-1, 1, *single_data_shape).to(device)
            outputs = model(batch_data)
            loss = loss_function(outputs, batch_data_truth)
            test_loss += loss.item()
    # render some test images
    n_test_render = 5
    test_images_truth, test_images = test_dataset.tensors[:]
    test_images_truth = test_images_truth[:n_test_render]
    test_images = test_images[:n_test_render]
    if model_type == "linear":
        recon_images = model(test_images.reshape(-1, single_data_size).to(device))
    elif model_type == "conv":
        recon_images = model(test_images.reshape(-1, 1, *single_data_shape).to(device))
    recon_images = recon_images.detach().cpu().numpy()
    fig, axes = plt.subplots(n_test_render, 3, figsize=(6, 6))
    for i, test_image in enumerate(test_images):
        axes[i, 0].imshow(test_images_truth[i].reshape(*single_data_shape), vmin=0.1, vmax=0.6)
        axes[i, 1].imshow(test_image.reshape(*single_data_shape), vmin=0.1, vmax=0.6)
        axes[i, 2].imshow(recon_images[i].reshape(*single_data_shape), vmin=0.1, vmax=0.6)
    plt.savefig("./autoenc_results/test_images.png")
    plt.close()
    # return loss
    return test_loss / len(test_loader)


def hyperparam_search():
    config = {
        "batch_size": tune.choice([32, 64, 128]),
        "code_size": tune.choice([16, 32, 64]),
        "lr": tune.loguniform(1e-4, 1e-1),
    }
    scheduler = tune.schedulers.ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10000,
        grace_period=10,
        reduction_factor=2,
    )
    reporter = tune.CLIReporter(
        parameter_columns=["batch_size", "code_size", "lr"],
        metric_columns=["loss", "training_iteration"],
    )
    result = tune.run(
        partial(train_model),
        config=config,
        num_samples=100,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=os.path.join(os.getcwd(), "./autoenc_results"),
    )

    # print best trial results
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create model with best performing hyperparameters
    best_trained_model = LinearAE(input_size=np.prod(single_data_shape), code_size=best_trial.config["code_size"]).to(device)
    # load model parameters from checkpoint
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    best_trained_model.eval()
    # test model
    test_loss = test_model(best_trained_model, device)
    print("test loss = {:.6f}".format(test_loss))


def single_train_run():
    config = {
        "batch_size": 32,
        "code_size": 32,
        "lr": 1e-3,
    }

    """ Train """
    train_model(config, tobetune=False, model_type="linear")

    """ Test """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearAE(code_size=config["code_size"], input_size=np.prod(single_data_shape)).to(device)
    # load trained model
    model_state, optimizer_state = torch.load(
        f"./autoenc_results/model_bs{config['batch_size']}_cs{config['code_size']}_lr{config['lr']}"
    )
    model.load_state_dict(model_state)
    model.eval()
    # test model
    test_loss = test_model(model, device, model_type="linear")
    print("test loss = {:.6f}".format(test_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyper", action="store_true", default=False, help="Hyperparameter search")
    args = parser.parse_args()

    if args.hyper:
        hyperparam_search()
    else:
        single_train_run()
