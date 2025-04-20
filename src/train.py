import os
import time
from typing import Literal

import fastwer
import torch
from torch.utils.data import DataLoader

from .config import NUM_CLASSES, NUM_EPOCHS
from .model import CRNN, ctc_greedy_decoder, base_transform
from .datasets.iam import IAMWordsDataset, IAMLinesDataset
from .datasets.nist import NISTDataset
from .utils import to_time_string, decode_text, collate_fn

class TrainCRNN:
    """Class to simplify the Training of the CRNN"""

    def __init__(
        self,
        model: CRNN | str | None = None,
        checkpoint_dir="weights/checkpoints",
        device: torch.device | None = None,
        batch_size=16,
        validation=True
    ):
        """
        Parameters
        ----------
        model
            Model to train for. If str is provided, tries to load the model from that path. Else creates a new Model.
        checkpoint_dir
            Path to save the Model Training Checkpoints to. Defaults to ``weights/checkpoints``.
        device
            Device to load the Model on. Uses a CUDA-compatible Device if available, otherwise the CPU.
        batch_size
            Batch Size to use in the Training. Defaults to 16.
        validation
            Whether to use a Validation Set or not. If True, uses 1 Batch as the Validation Set. Defaults to True.
        """

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if model is None:
            self.model = CRNN(NUM_CLASSES).to(self.device)
        elif isinstance(model, str):
            self.model = torch.load(model, self.device, weights_only=False)
        else:
            self.model = model

        self.criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loader: DataLoader | None = None

        self.epoch = 1
        self.batch_size = batch_size
        self.validation = validation
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
        loss: torch.Tensor = self.criterion(outputs, targets, input_lengths, target_lengths)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _validation_step_(self, batch):
        self.model.eval()
        with torch.no_grad():
            images, targets, input_lengths, target_lengths = batch
            images = images.to(self.device)
            outputs = self.model(images)
            outputs = outputs.permute(1, 0, 2)
            loss: torch.Tensor = self.criterion(outputs, targets, input_lengths, target_lengths)

            predicted_labels = ctc_greedy_decoder(outputs.cpu())
            actual_labels = [decode_text(label) for label in targets]
            print(
                f"Validation Loss: {loss.item():.4f}",
                "WER:", fastwer.score(actual_labels, predicted_labels),
                "CER:", fastwer.score(actual_labels, predicted_labels, char_level=True)
            )
            print("\nSample predictions:")
            for i in range(min(3, len(predicted_labels))):
                print(f"Prediction: {predicted_labels[i]}, Actual: {actual_labels[i]}")
            print()

    def train_model(self):
        """
        Starts the Training.

        Make sure to set the ``loader`` variable before starting the Training.
        """

        assert self.loader is not None, "loader not set"
        num_batches = len(self.loader) - (self.batch_size if self.validation else 0)

        while self.epoch <= NUM_EPOCHS:
            # Training
            total_loss = 0.0
            epoch_start = time.time_ns()
            for batch_idx, batch in enumerate(self.loader):
                batch_start = time.time_ns()
                loss = self._train_step_(batch)
                total_loss += loss
                if batch_idx % 10 == 0:
                    time_spent = time.time_ns() - batch_start
                    print(
                        f"\rEpoch [{self.epoch}/{NUM_EPOCHS}],",
                        f"Step [{batch_idx}/{num_batches}],",
                        f"{to_time_string(time_spent)}/step,",
                        f"Loss: {loss:.4f}",
                        end=''
                    )

            time_spent = time.time_ns() - epoch_start
            print(
                f"\nEpoch [{self.epoch}] finished with avg loss",
                round(total_loss / len(self.loader), 4),
                "in", to_time_string(time_spent)
            )

            if self.validation:
                self._validation_step_(next(iter(self.loader)))
            self._make_checkpoint()
            self.epoch += 1

    def start(
        self,
        save_to: torch.serialization.FILE_LIKE = "weights/crnn_model.pkl",
        dataset: Literal["iamwords", "iamlines", "nist"] = 'iamwords',
        **kwargs
    ):
        """
        Starts the Training on the specified ``dataset``

        Parameters
        ----------
        save_to
            Path to save the Trained Model to. Defaults to ``weights/crnn_model.pkl``.
        dataset
            Dataset to train the Model on. Defaults to ``iamwords``.
        labels_path
            Path to load the Labels from.
        images_dir
            Path to load the Images from.
        """

        print("Setting up Dataset")
        if dataset == 'iamwords':
            self.dataset = IAMWordsDataset(base_transform=base_transform, **kwargs)
        elif dataset == 'iamlines':
            self.dataset = IAMLinesDataset(base_transform=base_transform, **kwargs)
        elif dataset == 'nist':
            self.dataset = NISTDataset(transform=base_transform, **kwargs)
        self.loader = DataLoader(self.dataset, self.batch_size, True, collate_fn=collate_fn)

        print("Starting Training...")
        self.train_model()

        print("Training Completed. Saving Model...")
        torch.save(self.model, save_to)
        print("Model Saved.")
