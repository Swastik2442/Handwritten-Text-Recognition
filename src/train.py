import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import NUM_CLASSES, NUM_EPOCHS
from .model import CRNN, ctc_greedy_decoder, transform
from .datasets.iam import IAMDataset, collate_fn

class TrainCRNN:
    def __init__(self, load_from: str | None = None, checkpoint_dir="weights/checkpoints", batch_size=16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_from is None:
            self.model = CRNN(NUM_CLASSES).to(self.device)
        else:
            torch.serialization.add_safe_globals([
                CRNN,
                nn.BatchNorm2d,
                nn.Conv2d,
                nn.Dropout,
                nn.Linear,
                nn.LogSoftmax,
                nn.LSTM,
                nn.ReLU,
                nn.MaxPool2d,
                nn.Sequential,
            ])
            self.model = torch.load(load_from, self.device)

        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.epoch = 0
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def _make_checkpoint(self):
        torch.save(
            self.model,
            os.path.join(self.checkpoint_dir, f"_crnn_checkpoint_{self.epoch}.pkl")
        )

    def _train_step_(self, batch):
        self.model.train()
        images, targets, input_lengths, target_lengths = batch
        images = images.to(self.device)
        targets = targets.to(self.device)
        input_lengths = input_lengths.to(self.device)
        target_lengths = target_lengths.to(self.device)

        outputs = self.model(images)
        outputs = outputs.permute(1, 0, 2)
        loss = self.criterion(outputs, targets, input_lengths, target_lengths)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_model(self):
        while self.epoch < NUM_EPOCHS:
            # Training
            total_loss = 0.0
            for batch_idx, batch in enumerate(self.loader):
                loss = self._train_step_(batch)
                total_loss += loss
                if batch_idx % 10 == 0:
                    print(f"Epoch [{self.epoch + 1}/{NUM_EPOCHS}], Step [{batch_idx}], Loss: {loss:.4f}")

            print(f"Epoch [{self.epoch + 1}] finished with avg loss: {total_loss / len(self.loader):.4f}")

            # Validation
            self.model.eval()
            with torch.no_grad():
                images, labels, _ = next(iter(self.loader))
                images = images.to(self.device)
                outputs = self.model(images)
                decoded_preds = ctc_greedy_decoder(outputs.cpu())

                print("\nSample predictions:")
                for i in range(min(3, len(decoded_preds))):
                    print(f"Prediction: {decoded_preds[i]}, Actual: {labels[i]}")
                print()

            self._make_checkpoint()
            self.epoch += 1

    def start(self, save_to="weights/crnn_model.pkl", dataset='iam', **kwargs):
        """
        Starts training on the specified ``dataset``

        Args:
            save_to (str): Path to save the Trained Model to. Defaults to ``weights/crnn_model.pkl``.
            dataset (``iam``): Dataset to train the Model on. Defaults to ``iam``.
            labels_path (str | None): Path to load the Labels from. Required when ``dataset`` is ``iam``.
            images_dir (str | None): Path to load the Images from. Required when ``dataset`` is ``iam``.
        """

        print("Setting up Dataset")
        if dataset == 'iam':
            self.dataset = IAMDataset(transform=transform, **kwargs)
            self.loader = DataLoader(self.dataset, self.batch_size, True, collate_fn=collate_fn)

        assert hasattr(self, "loader") and self.loader is not None, "Dataset not Loaded"
        print("Starting Training...")
        self.train_model()

        print("Training Completed. Saving Model...")
        torch.save(self.model, save_to)
        print("Model Saved.")
