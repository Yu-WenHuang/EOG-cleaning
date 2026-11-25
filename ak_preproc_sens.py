#%%
import mne
from pathlib import Path
import numpy as np
from almkanal import (
    ICA,
    AlmKanal,
    Filter,
    MultiBlockMaxwell,
    EpochTRF,
    Resample
)
from cluster_jobs.meta_job import Job
from natsort import natsorted
import joblib
from almkanal.almkanal_steps.eog_utils import EOGCleaner
from almkanal.almkanal_steps.trf_utils import TRFSpanSpec
from scipy.io import savemat


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

#%%
class PreprocSensPipe(Job):
    job_data_folder = 'data_eog_meg_sens_epoch_newEOGCleaner_matfiles'

    def run(
        self,
        subject_id: str = '19650121taso',
        feature: str = 'envelope',
        # drift correction parameters
        hw_delay_s: float = 0.0165,
        fallback_drift_us_per_s: float = 499.0,
        # epoching parameter
        epoch_len_s: float = 6.0,
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
                
        
        data_path = Path(
            '/mnt/ceph/groups_hdd/SCCGroup/salzburg_brain_dynamics/shared_data/vocoding_dyslexia/'
        )

        audio_path = Path(
            '/mnt/ceph/groups_hdd/SCCGroup/salzburg_brain_dynamics/lreitinger/vocoding_dyslexia/data/audio_data'
        )
        
        path_list = natsorted(
            list(data_path.glob(f"*/{subject_id}_vocoding_blk[1-8].fif"))
            + list(data_path.glob(f"*/{subject_id}_vocoding_block[1-8].fif"))
            + list(data_path.glob(f"*/{subject_id}_block[1-8].fif"))
            + list(data_path.glob(f"*/{subject_id}_blk[1-8].fif"))
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


        def generate_event_spans(raw):

            events = mne.find_events(raw, stim_channel='STI101', shortest_event=1)
            all_good_events = np.concatenate([np.arange(51,75), np.arange(81, 105), np.arange(141, 165)])
            filtered_events = events[[True if event[2] in all_good_events else False for event in events]]

            event_dict = {}
            for event in filtered_events:
                if event[2] in range(51,75):
                    key = 'clean/' + str(event[2] - 50) 
                elif event[2] in range(81, 105):
                    key = '7ch/' + str(event[2] - 80) 
                elif event[2] in range(141, 165):
                    key = '3ch/' + str(event[2] - 140) 
                    
                if key in event_dict:
                    raise ValueError(f"Repeated stimulus label: {key}")
                
                event_dict.update({key: (event[0], None)})

            return TRFSpanSpec(spans_by_label=event_dict, metadata_by_label={})
                
        ak = AlmKanal(
            steps=[
                MultiBlockMaxwell(),
                Filter(0.1, 40),
                ICA(
                    train=True,
                    train_thresh=1,
                    eog=True,
                    ecg=True,
                    ecg_corr_thresh=0.4,
                    eog_corr_thresh=0.4,
                    emg=False,
                    resample_freq=200,
                ),
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
                EpochTRF(gen_span_spec=generate_event_spans,
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


        # export .mat file for coherence in ft
        n_epochs = len(epochs)

        trial = np.empty((1, n_epochs), dtype=object)
        time = np.empty((1, n_epochs), dtype=object)

        for i in range(n_epochs):

            epoch_data = epochs.get_data(item=i)[0]

            trial[0, i] = epoch_data
            time[0, i] = epochs.times.copy()

        label = np.asarray(epochs.ch_names, dtype=object).reshape(-1, 1)

        # recover original condition trigger codes
        event_id_to_name = {v: k for k, v in epochs.event_id.items()}

        trialinfo = np.array([
            int(event_id_to_name[event_id].split("/")[1]) +
            (50 if event_id_to_name[event_id].startswith("clean/") else
            80 if event_id_to_name[event_id].startswith("7ch/") else 140)
            for event_id in epochs.events[:, 2]
        ]).reshape(-1, 1)

        ft_data = {
            'trial': trial,
            'time': time,
            'label': label,
            'fsample': float(epochs.info['sfreq']),
            'trialinfo': trialinfo,
        }

        mat_path = Path(self.full_output_path).with_suffix('.mat')

        savemat(
            mat_path,
            {'data': ft_data},
            do_compression=True,
            long_field_names=True,
        )

        print(f"Saved .mat-file for {subject_id}")

        # # save ak report & json
        # ak.generate_json(f'/mnt/ceph/groups_hdd/SCCGroup/salzburg_brain_dynamics/lreitinger/vocoding_dyslexia/data/preproc_jsons/{subject_id}.json')

        # Path(f'/mnt/ceph/groups_hdd/SCCGroup/salzburg_brain_dynamics/lreitinger/vocoding_dyslexia/data/reports/{self.job_data_folder}/{subject_id}').mkdir(parents=True, exist_ok=True)

        # report.save(f'/mnt/ceph/groups_hdd/SCCGroup/salzburg_brain_dynamics/lreitinger/vocoding_dyslexia/data/reports/{self.job_data_folder}/{subject_id}/{subject_id}_report.html')


        print(f"Finished EOG preprocessing for {subject_id}")
        