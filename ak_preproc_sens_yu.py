#%%
import mne
from pathlib import Path
import numpy as np
from almkanal import (
    #ICA,
    AlmKanal,
    Filter,
    StimulusSpanSpec,
    #MultiBlockMaxwell,
    EpochTRF,
    Resample
)
from cluster_jobs.meta_job import Job
from natsort import natsorted
import joblib
from almkanal.almkanal_steps.eog_utils import EOGCleaner
#from almkanal.almkanal_steps.trf_utils import TRFSpanSpec
#from scipy.io import savemat


from almkanal.eye_utils.eog_utils import (
    DEFAULT_PEAK_THRESHOLD,
    DEFAULT_BLINK_REL_HEIGHT,
    DEFAULT_FIND_BLINKS,
    DEFAULT_INTERPOLATE_BLINKS,
    DEFAULT_CHECK_POLARITY,
    DEFAULT_CHECK_CHANNEL_ASSIGNMENT,
    DEFAULT_FLAT_THRESHOLD_V,
    DEFAULT_MAX_BAD_FRACTION,
)

# Map audio onset triggers to stimulus files.
TRIGGERS = {
    21: "bodyscan.wav",
    22: "safeplace.wav",
    23: "podcast.wav",
}

#%%
class PreprocSensPipe(Job):
    job_data_folder = 'sd_therapy_eog_meg_sens_epoch_matfiles'

    def run(
        self,
        subject_id: str = '20050629hibe_krbe29',
        feature: str = 'envelope',
        # drift correction parameters
        hw_delay_s: float = 0.0165,
        fallback_drift_us_per_s: float = 499.0,
        # epoching parameter
        epoch_len_s: float = 5.0,
        # EOG preprocessing parameters
        peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
        blink_rel_height: float = DEFAULT_BLINK_REL_HEIGHT,
        find_blinks: bool = DEFAULT_FIND_BLINKS,
        interpolate_blinks: bool = DEFAULT_INTERPOLATE_BLINKS,
        check_and_correct_eog_polarity: bool = DEFAULT_CHECK_POLARITY,
        check_and_correct_eog_channel_assignment: bool = DEFAULT_CHECK_CHANNEL_ASSIGNMENT,
        flat_threshold_v: float = DEFAULT_FLAT_THRESHOLD_V,
        max_bad_fraction: float = DEFAULT_MAX_BAD_FRACTION,
    ) -> None:
                
        
        base_path = Path(
            '/mnt/ceph/groups_hdd/SCCGroup/salzburg_brain_dynamics/staff/yuwen/sd_therapy'
        )

        data_path = base_path / 'sinuhe'
        audio_path = base_path / 'audio_data'
        
        path_list = natsorted(
            data_path.glob(f"*/{subject_id}_condition[1-3]_*_strgfix.fif")
        )

        print(subject_id, path_list)

        raw_list = []

        for cur_path in path_list:
            cur_raw = mne.io.read_raw_fif(cur_path, preload=True)

            for old, new, channel_type in [
                ("BIO001", "EOG001", "eog"),
                ("BIO002", "EOG002", "eog"),
                ("BIO003", "ECG003", "ecg"),
            ]:
                if old in cur_raw.ch_names:
                    cur_raw.set_channel_types({old: channel_type})
                    cur_raw.rename_channels({old: new})

            cur_raw.set_channel_types({
                "EOG001": "eog",
                "EOG002": "eog",
            })

            raw_list.append(cur_raw)

        if not raw_list:
            raise ValueError(f"No MEG blocks found for {subject_id}.")

        channel_order = raw_list[0].ch_names.copy()

        for block in raw_list:
            block.reorder_channels(channel_order)
        
        # 21/41 = Bodyscan, 22/42 = Safe Place, 23/43 = Podcast. The end codes
        # mark the transition to questions.
        def make_spans(raw):
                return StimulusSpanSpec.from_events(
                    raw,
                    onset_trigger_to_wav=TRIGGERS,
                    end_triggers=(41, 42, 43),
                    stim_channel="STI101",
                    )

                
        ak = AlmKanal(
            steps=[
                #MultiBlockMaxwell(),

                # meg filter 0.5-40
                Filter(0.1, 40),

                # ICA(
                #     train=True,
                #     train_thresh=1,
                #     eog=True,
                #     ecg=True,
                #     ecg_corr_thresh=0.4,
                #     eog_corr_thresh=0.4,
                #     emg=False,
                #     resample_freq=200,
                # ),

                EOGCleaner(
                    eog_channels=('EOG001', 'EOG002'),
                    find_blinks=find_blinks,
                    interpolate_blinks=interpolate_blinks,
                    peak_threshold=peak_threshold,
                    blink_rel_height=blink_rel_height,
                    check_and_correct_eog_polarity=check_and_correct_eog_polarity,
                    check_and_correct_eog_channel_assignment=check_and_correct_eog_channel_assignment,
                    flat_threshold_v=flat_threshold_v,
                    max_bad_fraction=max_bad_fraction,
                ),

                Resample(100),

                EpochTRF(gen_span_spec=make_spans,
                        base_audio_path=audio_path,
                        audio_channels=None,
                        realign_without_audio=True,
                        fallback_drift_us_per_s=fallback_drift_us_per_s,
                        feature=feature,
                        hw_delay_s=hw_delay_s,
                        on_alignment_error='raise',
                        audio_cutoff_hz=40,
                        epoch_len_s=epoch_len_s),
            ],
        )
        epochs, report = ak.run(raw_list)

        epo_data = {'epochs': epochs}
        # joblib.dump(epo_data, self.full_output_path)

        # print(f"Saved Joblib file: {self.full_output_path}")

        print(f"Finished EOG preprocessing for {subject_id}")
        
# %%
