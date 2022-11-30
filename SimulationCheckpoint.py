import os,pickle
import re
from typing import Optional

class SimulationResultsCheckPoint:
    accumilatted_run_statistics:tuple
    system_state_at_end = None
    name_base:str
    progress: Optional[int]

    stored_name: Optional[str]

    cp_dir_name = "CheckpointFiles"

    def __init__(self,name_base, progress, system_state_at_end, *accumilatted_run_statistics):
        self.name_base = name_base
        self.progress = progress
        self.system_state_at_end = system_state_at_end
        self.accumilatted_run_statistics = accumilatted_run_statistics
        self.stored_name = None

    def unload(self):
        return self.system_state_at_end, *self.accumilatted_run_statistics

    def store(self):
        file_name = SimulationResultsCheckPoint.get_file_name(self.name_base, self.progress)
        if os.path.exists(file_name):
            new_name = self.find_vacent_name()
            # move old checkpoint file to new place.
            os.rename(file_name, new_name)
        with open(file_name, "wb") as f:
            pickle.dump(self, f)

    def delete_self_from_disk(self):
        file_name = SimulationResultsCheckPoint.get_file_name(self.name_base, self.progress)
        assert os.path.exists(file_name)
        os.remove(file_name)

    @staticmethod
    def load(name_base, progress=None):
        file_name = SimulationResultsCheckPoint.get_file_name(name_base, progress)
        with open(file_name, "rb") as f:
            a = pickle.load(f)
        return a

    @staticmethod
    def b_is_cp_exists(name_base, progress=None):
        file_name = SimulationResultsCheckPoint.get_file_name(name_base, progress)
        return os.path.exists(file_name)

    @staticmethod
    def b_find_less_advanced_cp(name_base, progress):
        assert (progress is not None)
        # todo this implementation is wierd but works, rewrite this with one for loop.
        files = os.listdir(SimulationResultsCheckPoint.cp_dir_name)
        files_with_same_base = [f for f in files if SimulationResultsCheckPoint.extract_base_name_from_cp_file_name(f) == name_base]
        progresses = [SimulationResultsCheckPoint.extract_progress_from_cp_file_name(f) for f in files_with_same_base]
        most_advanced = None
        for i in range(len(progresses)):
            if progresses[i] < progress and ((most_advanced is None) or (progresses[i] > most_advanced)):
                most_advanced = progresses[i]
        return most_advanced

    @staticmethod
    def get_file_name(name_base, progress):
        if progress is not None:
            if progress == float("inf"):
                progress = "inf"
            else:
                progress = int(progress)
            return os.path.join(SimulationResultsCheckPoint.cp_dir_name, f"{name_base}.prg={progress}.checkpoint")
        else:
            return os.path.join(SimulationResultsCheckPoint.cp_dir_name, f"{name_base}.checkpoint")

    @staticmethod
    def extract_base_name_from_cp_file_name(cp_file_name):
        cp_file_name_no_path = os.path.split(cp_file_name)[1]
        base_name = os.path.splitext(os.path.splitext(cp_file_name_no_path)[0])[0]
        return base_name

    @staticmethod
    def extract_progress_from_cp_file_name(cp_file_name):
        cp_file_name_no_path = os.path.split(cp_file_name)[1]
        prog_text_all = re.findall("\\.prg=([0-9]+)",cp_file_name_no_path)
        assert len(prog_text_all) == 1
        return int(prog_text_all[0])

    def find_vacent_name(self):
        i = 0
        while True:
            i += 1
            name_out = SimulationResultsCheckPoint.get_file_name(f"{self.name_base}.old={i}", self.progress)
            if not os.path.exists(name_out):
                break
        return name_out