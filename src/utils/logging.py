"""Logging + W&B wrapper.

Two responsibilities:
1. ``setup_console_logger(name)`` — one-shot stdlib logger with consistent
   format. Used by Phase 0 scripts (generate_checksums, unk_token_scan,
   build_label_vocab). No external deps.

2. ``WandbRun`` — context-manager wrapper around ``wandb.init`` / ``wandb.log``
   / ``wandb.finish``. Honors cfg.logging.{wandb_enabled, wandb_mode}. When
   wandb is disabled or unreachable, all log calls become no-ops so trainer
   code does not need to branch.

Design choice — no global wandb state:
    The trainer holds a ``WandbRun`` instance and calls ``run.log(...)``.
    No ``import wandb; wandb.log(...)`` scattered through the codebase.
    Easier to mock in tests; easier to disable for ad-hoc runs.
"""
from __future__ import annotations

import logging
import sys
from contextlib import AbstractContextManager
from typing import Any


# ---------------------------------------------------------------------------
# Console logger
# ---------------------------------------------------------------------------


def setup_console_logger(
    name: str = "mcis",
    level: int = logging.INFO,
    fmt: str = "%(asctime)s %(levelname)s %(name)s: %(message)s",
) -> logging.Logger:
    """Get-or-create a stdlib logger with consistent formatting.

    Idempotent: calling twice with the same name returns the same logger
    without duplicating handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(stream=sys.stderr)
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
    logger.setLevel(level)
    logger.propagate = False  # avoid double-printing under pytest capture
    return logger


# ---------------------------------------------------------------------------
# W&B wrapper
# ---------------------------------------------------------------------------


class WandbRun(AbstractContextManager):
    """Lazy W&B run wrapper. Safe no-op when wandb is disabled or unavailable.

    Usage:
        with WandbRun(project="MCIS_AI", run_name=cfg.experiment_name,
                      config=cfg.to_dict(), enabled=cfg.logging.wandb_enabled,
                      mode=cfg.logging.wandb_mode) as run:
            run.log({"train/loss": 0.5}, step=42)
            run.summary["best_f1_macro"] = 0.82
    """

    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        enabled: bool = True,
        mode: str = "online",  # online | offline | disabled
        tags: list[str] | None = None,
        group: str | None = None,
        notes: str | None = None,
    ) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self.enabled = enabled and mode != "disabled"
        self.mode = mode
        self.tags = tags
        self.group = group
        self.notes = notes
        self._run: Any = None
        self._wandb: Any = None

    def __enter__(self) -> "WandbRun":
        if not self.enabled:
            return self
        try:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config,
                mode=self.mode,
                tags=self.tags,
                group=self.group,
                notes=self.notes,
                reinit=True,
            )
        except ImportError:
            self.enabled = False
        except Exception:
            # wandb online could fail if no internet / bad key; fall back silently.
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self._run is not None:
            try:
                self._wandb.finish(exit_code=1 if exc_type else 0)
            except Exception:
                pass

    # ---- public API (no-op when disabled) ----

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.log(data, step=step)

    @property
    def summary(self) -> Any:
        """W&B summary dict. Returns a no-op shim when disabled."""
        if self.enabled and self._run is not None:
            return self._run.summary
        return _SummaryShim()

    @property
    def run_id(self) -> str | None:
        if self.enabled and self._run is not None:
            return self._run.id
        return None

    @property
    def url(self) -> str | None:
        if self.enabled and self._run is not None:
            return self._run.get_url()
        return None


class _SummaryShim:
    """Drop-in replacement for ``wandb.Run.summary`` when wandb is off."""

    def __setitem__(self, key: str, value: Any) -> None:
        pass

    def __getitem__(self, key: str) -> Any:
        return None

    def update(self, d: dict[str, Any]) -> None:
        pass
