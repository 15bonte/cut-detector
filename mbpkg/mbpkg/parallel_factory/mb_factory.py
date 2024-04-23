import threading
import math
from typing import Union, Optional, Literal

import numpy as np

from cut_detector.factories.mid_body_detection_factory import MidBodyDetectionFactory
from cut_detector.utils.mid_body_spot import MidBodySpot
import cut_detector.utils.mb_support.detection as mbd

class ParallelFactory(MidBodyDetectionFactory):
    DETECTION_PARALLELIZATION_METHOD = Union[
        bool,
        Literal[
            "pool",
            "thread",
            "np_thread",
            "max_thread",
        ]
    ]

    SPOT_DETECTION_METHOD = MidBodyDetectionFactory.SPOT_DETECTION_METHOD

    def detect_mid_body_spots(
            self,
            mitosis_movie: np.ndarray,
            mask_movie: Optional[np.ndarray] = None,
            mid_body_channel=1,
            sir_channel=0,
            mode: SPOT_DETECTION_METHOD = mbd.cur_dog,
            log_blob_spot: bool = False,
            parallelization: DETECTION_PARALLELIZATION_METHOD = False,
            ) -> dict[int, list[MidBodySpot]]:
        """
        Parameters
        ----------
        mitosis_movie: TYXC
        mask_movie: TYX

        Returns
        ----------
        spots_dictionary: dict[int, list[MidBodySpot]]
        """

        # Default mask is all ones
        if mask_movie is None:
            mask_movie = np.ones(mitosis_movie.shape[:-1])


        if isinstance(parallelization, bool):
            if parallelization:
                return self.thread_pool_detect_mid_body_spots(
                    mitosis_movie,
                    mask_movie,
                    mid_body_channel,
                    sir_channel,
                    mode
                )
            else:
                return self.serial_detect_mid_body_spots(
                    mitosis_movie,
                    mask_movie,
                    mid_body_channel,
                    sir_channel,
                    mode,
                    log_blob_spot
                )

        elif isinstance(parallelization, str):
            if parallelization == "pool":
                return self.thread_pool_detect_mid_body_spots(
                    mitosis_movie,
                    mask_movie,
                    mid_body_channel,
                    sir_channel,
                    mode
                )
            elif parallelization == "thread":
                return self.std_parallel_detect_mid_body_spots(
                    mitosis_movie,
                    mask_movie,
                    mid_body_channel,
                    sir_channel,
                    mode,
                )
            elif parallelization == "np_thread":
                return self.std_np_parallel_detect_mid_body_spots(
                    mitosis_movie,
                    mask_movie,
                    mid_body_channel,
                    sir_channel,
                    mode,
                )
            elif parallelization == "max_thread":
                return self.std_max_parallel_detect_mid_body_spots(
                    mitosis_movie,
                    mask_movie,
                    mid_body_channel,
                    sir_channel,
                    mode,
                )
            else:
                raise RuntimeError(f"parallelization str must be either 'pool'/'thread'/'np_thread': found {parallelization}")

        else:
            raise RuntimeError("parallelization must be either a str or bool")
        
    def std_parallel_detect_mid_body_spots(
            self,
            mitosis_movie: np.ndarray,
            mask_movie:    np.ndarray,
            mid_body_channel = 1,
            sir_channel      = 0,
            method: SPOT_DETECTION_METHOD = mbd.cur_dog,
            ) -> dict[int, list[MidBodySpot]]:

        nb_frames = mitosis_movie.shape[0] # TYXC

        def ret_writer(
                slots: list,
                inc_frame_beg: int,
                ex_frame_end: int,
                ) -> None:

            # print(f"t: [{inc_frame_beg}-{ex_frame_end}[")

            for frame in range(inc_frame_beg, ex_frame_end):
                mitosis_frame = mitosis_movie[frame]
                mask_frame = mask_movie[frame]
                slots[frame] = self._spot_detection(
                    mitosis_frame,
                    mask_frame,
                    mid_body_channel,
                    sir_channel,
                    method,
                    frame,
                    log_blob_spot=False
                )

        # print("nb_frames:", nb_frames)

        slots = [None] * nb_frames
        target_thread_count = 25 if nb_frames >= 25 else (math.floor(nb_frames / 4))
        frames_per_thread, rem = divmod(nb_frames, target_thread_count)

        all_threads = [
            threading.Thread(
                target=ret_writer,
                args=[
                    slots,
                    frames_per_thread*t,
                    frames_per_thread*(t+1),
                ]
            )
            for t in range(target_thread_count)
        ]
        all_threads.append(threading.Thread(
            target=ret_writer,
            args=[
                slots,
                frames_per_thread*target_thread_count,
                nb_frames,
            ]
        ))

        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()

        return {f: slots[f] for f in range(nb_frames)}

    def std_np_parallel_detect_mid_body_spots(
            self,
            mitosis_movie: np.ndarray,
            mask_movie:    np.ndarray,
            mid_body_channel = 1,
            sir_channel      = 0,
            method: SPOT_DETECTION_METHOD = mbd.cur_dog,
            ) -> dict[int, list[MidBodySpot]]:

        nb_frames = mitosis_movie.shape[0] # TYXC
        slots = {k: [] for k in range(nb_frames)}

        def slot_writer(
                frames: list[int],
                ) -> None:

            # print(f"frames:", frames)

            for frame in frames:
                mitosis_frame = mitosis_movie[frame]
                mask_frame = mask_movie[frame]
                slots[frame] = self._spot_detection(
                    mitosis_frame,
                    mask_frame,
                    mid_body_channel,
                    sir_channel,
                    method,
                    frame,
                    log_blob_spot=False
                )

        # print("nb_frames:", nb_frames)

        target_thread_count = 25 if nb_frames >= 25 else nb_frames
        frames = np.arange(nb_frames)
        splits = np.array_split(frames, target_thread_count)

        all_threads = [
            threading.Thread(
                target=slot_writer,
                args=[
                    frame_split
                ]
            )
            for frame_split in splits
        ]

        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()

        return slots

    def std_max_parallel_detect_mid_body_spots(
            self,
            mitosis_movie: np.ndarray,
            mask_movie:    np.ndarray,
            mid_body_channel = 1,
            sir_channel      = 0,
            method: SPOT_DETECTION_METHOD = mbd.cur_dog,
            ) -> dict[int, list[MidBodySpot]]:

        nb_frames = mitosis_movie.shape[0] # TYXC
        slots = {k: [] for k in range(nb_frames)}

        def slot_writer(
                frames: list[int],
                ) -> None:

            # print(f"frames:", frames)

            for frame in frames:
                mitosis_frame = mitosis_movie[frame]
                mask_frame = mask_movie[frame]
                slots[frame] = self._spot_detection(
                    mitosis_frame,
                    mask_frame,
                    mid_body_channel,
                    sir_channel,
                    method,
                    frame,
                    log_blob_spot=False
                )

        # print("nb_frames:", nb_frames)

        target_thread_count = nb_frames
        frames = np.arange(nb_frames)
        splits = np.array_split(frames, target_thread_count)

        all_threads = [
            threading.Thread(
                target=slot_writer,
                args=[
                    frame_split
                ]
            )
            for frame_split in splits
        ]

        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()

        return slots



