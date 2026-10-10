"""ワークフロー実行中の DRAM / 専用DRAM / 共有DRAM のピークを、終了後にログへ出す。"""

from __future__ import annotations

import logging
import time

from .wddm import PeakSampler, startup_line

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_LOG = "[PeakDRAM]"


def _install() -> None:
    import execution

    original = execution.PromptExecutor.execute_async
    if getattr(original, "_peak_dram", False):
        return

    async def execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        sampler = PeakSampler()
        started = time.perf_counter()
        status = "ok"
        try:
            sampler.start()
        except Exception:
            logging.exception("%s 計測を開始できませんでした", _LOG)
        try:
            result = await original(self, prompt, prompt_id, extra_data=extra_data, execute_outputs=execute_outputs)
            if not getattr(self, "success", True):
                status = "error"
            return result
        except Exception:
            status = "error"
            raise
        finally:
            elapsed = time.perf_counter() - started
            try:
                report = sampler.stop()
                logging.info(report.line(str(prompt_id), status, elapsed))
            except Exception:
                logging.exception("%s 計測結果を出せませんでした prompt=%s", _LOG, prompt_id)

    execute_async._peak_dram = True
    execution.PromptExecutor.execute_async = execute_async
    logging.info(
        "%s 実行中のピークを記録し、終了後に DRAM・専用DRAM・共有DRAM をログへ出します",
        _LOG,
    )
    logging.info(startup_line())


_install()
