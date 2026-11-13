"""CPU smoke test for neuralforecast's masked-target training contract."""

from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(
    importlib.util.find_spec("neuralforecast"),
    "neuralforecast is installed only in the approved WSL analysis environment",
)
class PatchTSTMissingMaskTest(unittest.TestCase):
    def test_missing_targets_are_masked_and_full_horizon_is_predicted(self) -> None:
        import numpy as np
        import pandas as pd
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast.losses.pytorch import MAE
        from neuralforecast.models import PatchTST

        loss_calls: list[tuple[str, int, int]] = []

        class RecordingMAE(MAE):
            def __init__(self, role: str) -> None:
                super().__init__()
                self.role = role

            def __call__(self, y, y_hat, mask=None, y_insample=None):
                if mask is not None:
                    loss_calls.append(
                        (
                            self.role,
                            int((mask == 0).sum().item()),
                            int((mask != 0).sum().item()),
                        )
                    )
                return super().__call__(
                    y=y, y_hat=y_hat, mask=mask, y_insample=y_insample
                )

        n = 120
        idx = np.arange(n)
        observed = np.ones(n, dtype=np.float32)
        missing = np.array([30, 31, 57, 91, 114])
        observed[missing] = 0

        raw = 50 + np.sin(idx / 5) * 7 + np.cos(idx / 13) * 3
        # PatchTST consumes the filled y values directly; available_mask marks
        # which target positions remain eligible for training and validation loss.
        filled = raw.copy()
        for point in missing:
            filled[point] = filled[point - 1]

        frame = pd.DataFrame(
            {
                "unique_id": "rt_price",
                "ds": pd.date_range("2025-01-01", periods=n, freq="15min"),
                "y": filled.astype(np.float32),
                "available_mask": observed,
            }
        )

        model = PatchTST(
            h=4,
            input_size=32,
            encoder_layers=1,
            n_heads=2,
            hidden_size=8,
            linear_hidden_size=16,
            patch_len=8,
            stride=4,
            revin=False,
            scaler_type="identity",
            loss=RecordingMAE("train"),
            valid_loss=RecordingMAE("validation"),
            max_steps=2,
            batch_size=8,
            windows_batch_size=16,
            random_seed=7,
            val_check_steps=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            accelerator="cpu",
            devices=1,
        )
        forecast = NeuralForecast(models=[model], freq="15min")
        forecast.fit(df=frame, val_size=8, verbose=False)
        predictions = forecast.predict(df=frame)

        for role in ("train", "validation"):
            self.assertTrue(
                any(name == role and masked > 0 and used > 0 for name, masked, used in loss_calls),
                f"{role} loss did not receive both masked and observed targets: {loss_calls}",
            )
        self.assertEqual(len(predictions), 4)
        self.assertTrue(predictions["PatchTST"].notna().all())

        point_loss = MAE()
        target = torch.tensor([[[1.0], [9999.0]]])
        estimate = torch.zeros_like(target)
        mask = torch.tensor([[[1.0], [0.0]]])
        loss_a = point_loss(y=target, y_hat=estimate, mask=mask)
        target[0, 1, 0] = -9999.0
        loss_b = point_loss(y=target, y_hat=estimate, mask=mask)
        torch.testing.assert_close(loss_a, loss_b)


if __name__ == "__main__":
    unittest.main()
