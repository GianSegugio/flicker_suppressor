from __future__ import annotations
import os, threading, traceback, uuid
from pathlib import Path
from PySide6.QtCore import QObject, QRunnable, Signal, Slot
from .engine import CancelledError


class WorkerSignals(QObject):
    log = Signal(str, str)
    finished = Signal(str, str, object)
    error = Signal(str, str)
    cancelled = Signal(str)


class InferenceWorker(QRunnable):
    def __init__(self, task_id, engine, src: Path, dst: Path, settings: dict, masks: dict|None=None):
        super().__init__(); self.task_id=task_id; self.engine=engine; self.src=Path(src); self.dst=Path(dst); self.settings=dict(settings); self.masks={str(k):Path(v) for k,v in (masks or {}).items()}; self.cancel_event=threading.Event(); self.signals=WorkerSignals()
    def cancel(self): self.cancel_event.set()
    @Slot()
    def run(self):
        try:
            stats=self.engine.process_file(self.src,self.dst,self.settings,masks=self.masks,callback=lambda x:self.signals.log.emit(self.task_id,x),cancel_event=self.cancel_event)
            self.signals.finished.emit(self.task_id,str(self.dst),stats)
        except CancelledError:
            self.signals.cancelled.emit(self.task_id)
        except Exception:
            self.signals.error.emit(self.task_id,traceback.format_exc())


class AutoSettingsWorker(QRunnable):
    """Estimate band settings for one image, off the UI thread.

    Runs numpy/scipy only -- no torch, no GPU -- so it is safe to queue one per
    image during a large import. Failures are reported rather than raised: a
    broken estimate must never stop an image from being imported.
    """

    def __init__(self, doc_id: str, src: Path):
        super().__init__()
        self.doc_id = str(doc_id)
        self.src = Path(src)
        self.signals = WorkerSignals()

    @Slot()
    def run(self):
        try:
            from autosettings import estimate
            result = estimate(self.src)
            self.signals.finished.emit(self.doc_id, str(self.src), result)
        except Exception:
            self.signals.error.emit(self.doc_id, traceback.format_exc())


class CopyWorker(QRunnable):
    """Cancelable, atomic file copy used by GUI exports.

    Data is written to a temporary sibling of the destination and moved into
    place only after the copy succeeds.  Cancelling therefore never leaves a
    partial exported file at the final path.
    """
    def __init__(self, task_id, src: Path, dst: Path, chunk_size: int=4*1024*1024):
        super().__init__(); self.task_id=task_id; self.src=Path(src); self.dst=Path(dst); self.chunk_size=max(64*1024,int(chunk_size)); self.cancel_event=threading.Event(); self.signals=WorkerSignals()

    def cancel(self):
        self.cancel_event.set()

    @Slot()
    def run(self):
        tmp=None
        try:
            self.dst.parent.mkdir(parents=True,exist_ok=True)
            tmp=self.dst.with_name(f'.{self.dst.name}.{uuid.uuid4().hex}.flicker-tmp')
            with self.src.open('rb') as fin, tmp.open('wb') as fout:
                while True:
                    if self.cancel_event.is_set():
                        raise CancelledError()
                    block=fin.read(self.chunk_size)
                    if not block:
                        break
                    fout.write(block)
                fout.flush()
                os.fsync(fout.fileno())
            if self.cancel_event.is_set():
                raise CancelledError()
            try:
                stat=self.src.stat()
                os.utime(tmp,(stat.st_atime,stat.st_mtime))
            except OSError:
                pass
            os.replace(tmp,self.dst)
            tmp=None
            self.signals.finished.emit(self.task_id,str(self.dst),None)
        except CancelledError:
            if tmp is not None:
                try: tmp.unlink(missing_ok=True)
                except OSError: pass
            self.signals.cancelled.emit(self.task_id)
        except Exception:
            if tmp is not None:
                try: tmp.unlink(missing_ok=True)
                except OSError: pass
            self.signals.error.emit(self.task_id,traceback.format_exc())
