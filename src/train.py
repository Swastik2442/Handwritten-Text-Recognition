import os
import time
from typing import Literal

import fastwer
import torch
from torch.utils.data import DataLoader

from .config import NUM_CLASSES
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
        num_epochs=25,
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
        num_epoches
            Number of Epochs to run the Training for. Defaults to 25.
        batch_size
            Batch Size to use in the Training. Defaults to 16.
        validation
            Whether to use a Validation Set or not. If True, uses 1 Batch as the Validation Set. Defaults to True.
        """

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if isinstance(model, CRNN):
            self.model = model
        else:
            self.model = CRNN(NUM_CLASSES)
            if isinstance(model, str):
                self.model.load_state_dict(torch.load(model))
        self.model = self.model.to(self.device)

        self.criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loader: DataLoader | None = None

        self.train_history: list[tuple[int, int, float]] = []
        self.val_history: list[tuple[int, float, float, float]] = []

        self.epoch = 1
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.validation = validation
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def _make_checkpoint(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.checkpoint_dir, f"_crnn_checkpoint_{self.epoch}.pth")
        )

    def _train_step_(self, batch: tuple[torch.Tensor, ...]):
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

    def _validation_step_(self, batch: tuple[torch.Tensor, ...]):
        self.model.eval()
        with torch.no_grad():
            images, targets, input_lengths, target_lengths = batch
            images = images.to(self.device)
            input_lengths = input_lengths.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            outputs: torch.Tensor = self.model(images)
            outputs = outputs.permute(1, 0, 2)
            loss: torch.Tensor = self.criterion(outputs, targets, input_lengths, target_lengths)

            predicted_labels = ctc_greedy_decoder(outputs.cpu())
            targets_list = targets.cpu().tolist()

            idx = 0
            actual_labels: list[str] = []
            for tlen in target_lengths.tolist():
                actual_labels.append(''.join(decode_text(targets_list[idx:idx+tlen])))
                idx += tlen

            print("\nSample predictions:")
            for i in range(min(3, len(predicted_labels))):
                print(f"Prediction: `{predicted_labels[i]}`, Actual: `{actual_labels[i]}`")

        wer = fastwer.score(actual_labels, predicted_labels)
        cer = fastwer.score(actual_labels, predicted_labels, char_level=True)
        return loss.item(), wer, cer

    def train_model(self):
        """
        Starts the Training.

        Make sure to set the ``loader`` variable before starting the Training.
        """

        assert self.loader is not None, "loader not set"
        num_batches = len(self.loader) - (self.batch_size if self.validation else 0)

        while self.epoch <= self.num_epochs:
            # Training
            total_loss = 0.0
            epoch_start = time.time_ns()
            for batch_idx, batch in enumerate(self.loader):
                batch_start = time.time_ns()
                loss = self._train_step_(batch)
                total_loss += loss
                self.train_history.append((self.epoch, batch_idx, loss))
                if batch_idx % 5 == 0:
                    time_spent = time.time_ns() - batch_start
                    print(
                        f"\rEpoch [{self.epoch}/{self.num_epochs}],",
                        f"Step [{batch_idx}/{num_batches}],",
                        f"{to_time_string(time_spent)}/step,",
                        f"Loss: {loss:.4f}",
                        end=''
                    )

            time_spent = time.time_ns() - epoch_start
            print(
                f"\nEpoch [{self.epoch}/{self.num_epochs}] finished with an avg loss of",
                round(total_loss / len(self.loader), 4),
                "in", to_time_string(time_spent)
            )

            if self.validation:
                loss, wer, cer = self._validation_step_(next(iter(self.loader)))
                self.val_history.append((self.epoch, loss, wer, cer))
                print(
                    f"Epoch [{self.epoch}/{self.num_epochs}]",
                    f"Validation Loss: {loss:.4f},",
                    f"WER: {wer:.4f}, CER: {cer:.4f}"
                )
            self._make_checkpoint()
            self.epoch += 1
            print()

    def start(
        self,
        save_to: torch.serialization.FILE_LIKE = "weights/crnn_model.pth",
        dataset: Literal["iamwords", "iamlines", "nist"] = 'iamwords',
        **kwargs
    ):
        """
        Starts the Training on the specified ``dataset``

        Parameters
        ----------
        save_to
            Path to save the Trained Model to. Defaults to ``weights/crnn_model.pth``.
        dataset
            Dataset to train the Model on. Defaults to ``iamwords``.
        labels_path
            Path to load the Labels from.
        images_dir
            Path to load the Images from.
        verify_images
            Whether to verify and remove corrupted Images. Defaults to False.
        """

        print("Setting up Dataset")
        if dataset == 'iamwords':
            self.dataset = IAMWordsDataset(base_transform=base_transform, **kwargs)
        elif dataset == 'iamlines':
            self.dataset = IAMLinesDataset(base_transform=base_transform, **kwargs)
        elif dataset == 'nist':
            self.dataset = NISTDataset(transform=base_transform, **kwargs)
        self.loader = DataLoader(self.dataset, self.batch_size, True, collate_fn=collate_fn)

        print("\nStarting Training...\n")
        self.train_model()

        print("\nTraining Completed. Saving Model...")
        torch.save(self.model.state_dict(), save_to)
        print("\rModel Saved.")
